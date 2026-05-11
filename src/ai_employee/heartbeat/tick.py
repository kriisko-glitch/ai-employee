"""The tick loop — what actually makes an agent 'alive'.

Each tick:
  1. Check STOP / FREEZE flags. STOP exits the loop. FREEZE skips writes.
  2. Check daily budget. Exit if exceeded.
  3. Load state, build prompt, recall memory.
  4. Call the runner.
  5. Parse a [STATE intensity] header and optional [next_tick_seconds: N].
  6. Update state.json atomically.
  7. Optionally ingest the response as a memory chunk.
  8. Post via the configured transport.
  9. Sleep next_delay (clamped by cadence + modulated by user activity).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from ..config import AgentConfig
from ..prompt import build_prompt
from ..runner import build_runner, Runner
from ..transport import build_transport
from .budget import BudgetTracker
from .modulation import compute_activity_multiplier
from .state import load_state, update_state, write_state_atomic

log = logging.getLogger(__name__)

# Matches: [STATE_NAME intensity]   (intensity = 0..3 typically)
_STATE_HEADER_RE = re.compile(r"\[([A-Z_]+)\s+(\d+)\]")
# Matches: [next_tick_seconds: N]
_NEXT_TICK_RE = re.compile(r"\[next_tick_seconds:\s*(\d+)\]", re.IGNORECASE)


def _parse_response(text: str, default_state: str, default_intensity: int,
                    default_delay: int) -> tuple[str, int, int]:
    """Extract (state, intensity, next_delay) from the model's response."""
    state, intensity = default_state, default_intensity
    m = _STATE_HEADER_RE.search(text)
    if m:
        state = m.group(1)
        intensity = int(m.group(2))

    next_delay = default_delay
    m2 = _NEXT_TICK_RE.search(text)
    if m2:
        next_delay = int(m2.group(1))

    return state, intensity, next_delay


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def tick_once(
    config: AgentConfig,
    runner: Optional[Runner] = None,
    extra_observation: Optional[str] = None,
    persist_memory: bool = True,
) -> dict:
    """Execute one tick. Returns the new state dict.

    Useful for `aie tick <name>` (one-shot) and inside run_loop().
    """
    # STOP/FREEZE checks.
    if config.stop_flag_path.exists():
        log.info("STOP flag present; refusing to tick.")
        return load_state(config.state_file, config.heartbeat.cadence.default_seconds)

    frozen = config.freeze_flag_path.exists()

    # Build the runner if not provided.
    if runner is None:
        runner = build_runner(config.model)

    # Budget gate.
    tracker = BudgetTracker(config.budget_file, config.heartbeat.budget)
    if config.heartbeat.enabled and tracker.exceeded():
        log.warning("Daily budget exceeded; skipping tick.")
        return load_state(config.state_file, config.heartbeat.cadence.default_seconds)

    # Load state + recall.
    state = load_state(config.state_file, config.heartbeat.cadence.default_seconds)

    recall_chunks: list[dict] = []
    if config.memory.enabled and config.memory.storage.backend == "sqlite_vec":
        try:
            from ..memory.retrieve import retrieve
            query = state.get("last_observation") or config.name
            recall_chunks = retrieve(config.memory_db_file, query, config.memory)
        except Exception as e:
            log.warning("Memory recall failed: %s", e)

    # Build prompt and call the model.
    system, user = build_prompt(config, state, recall_chunks, extra_observation)
    result = runner.run(system, user)

    # Track spend.
    tracker.record(result.input_tokens, result.output_tokens)

    # Parse + update state.
    new_state_name, new_intensity, next_delay = _parse_response(
        result.text,
        default_state=state.get("state", "BOREDOM"),
        default_intensity=state.get("intensity", 1),
        default_delay=config.heartbeat.cadence.default_seconds,
    )

    # Modulate by user activity.
    multiplier = compute_activity_multiplier(config.workspace, config.heartbeat.modulation)
    next_delay = int(next_delay * multiplier)
    next_delay = _clamp(
        next_delay,
        config.heartbeat.cadence.min_seconds,
        config.heartbeat.cadence.max_seconds,
    )

    new_state = update_state(
        state,
        new_state=new_state_name,
        new_intensity=new_intensity,
        next_delay=next_delay,
        observation=extra_observation or "tick",
    )

    if not frozen:
        write_state_atomic(config.state_file, new_state)

        if persist_memory and config.memory.enabled and config.memory.storage.backend == "sqlite_vec":
            try:
                from ..memory.score import remember
                remember(config.memory_db_file, result.text, config.memory,
                         source=f"tick:{new_state_name}")
            except Exception as e:
                log.warning("Memory ingest failed: %s", e)

    # Post.
    transport = build_transport(config.transport, name=config.name)
    transport.post(result.text)

    return new_state


def run_loop(config: AgentConfig) -> None:
    """Long-running heartbeat. Exits on STOP flag, budget cap, or KeyboardInterrupt."""
    if not config.heartbeat.enabled:
        raise RuntimeError(
            f"Heartbeat is disabled in agent.yaml for {config.name!r}. "
            f"Set heartbeat.enabled: true to use `aie run`."
        )

    runner = build_runner(config.model)
    log.info("Heartbeat starting for %s", config.name)

    try:
        while True:
            if config.stop_flag_path.exists():
                log.info("STOP flag detected; exiting cleanly.")
                return

            tracker = BudgetTracker(config.budget_file, config.heartbeat.budget)
            if tracker.exceeded():
                log.warning("Daily budget cap reached; exiting loop.")
                return

            state = tick_once(config, runner=runner)
            delay = state.get("next_tick_delay_seconds", config.heartbeat.cadence.default_seconds)
            log.info("Tick complete; sleeping %ss", delay)
            # Sleep in 1s slices so STOP is responsive.
            for _ in range(int(delay)):
                if config.stop_flag_path.exists():
                    return
                time.sleep(1)
    except KeyboardInterrupt:
        log.info("Interrupted; exiting.")
