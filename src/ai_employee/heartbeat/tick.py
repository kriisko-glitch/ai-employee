"""The tick loop — what makes an agent 'alive'.

Each tick (in full v0.2 form):

  1. Read STOP/FREEZE flags. STOP exits the loop.
  2. Read state.json.
  3. If conversation history > threshold → this is a DREAMING tick.
     Otherwise if state == "WAKING" → this is a WAKING tick (one grace tick
     after dreaming). Otherwise → normal tick.
  4. Check daily budget. Skip if exceeded.
  5. Fetch recent channel messages if discord listener enabled.
  6. Recall valence × SDT × decay weighted chunks (current state biases).
  7. Build the prompt (system + user + optional conversation history).
  8. Call the runner.
  9. If DREAMING → parse moments, ingest as scored chunks, append MEMORY.md,
     clear conversation history, mark wake_pending.
     Else → append user/assistant turn to conversation.jsonl (if enabled),
     ingest assistant response as a chunk.
 10. Atomically write state.json.
 11. Post the cleaned response via transport.
 12. Compute next-tick delay (model-chosen × modulation × discord-active).
 13. Sleep.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from ..config import AgentConfig
from ..conversation import (
    append_turn, load_recent, size_bytes, clear, to_message_list,
)
from ..prompt import build_prompt
from ..runner import build_runner, Runner
from ..transport import build_transport
from .budget import BudgetTracker
from .cadence import discord_active_delay
from .modulation import compute_activity_multiplier
from .state import load_state, update_state, write_state_atomic

log = logging.getLogger(__name__)

_STATE_HEADER_RE = re.compile(r"\[([A-Z_]+)\s+(\d+)\]")
_NEXT_TICK_RE = re.compile(r"\[next_tick_seconds:\s*(\d+)\]", re.IGNORECASE)


def _parse_response(text: str, default_state: str, default_intensity: int,
                    default_delay: int) -> tuple[str, int, int]:
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


def _is_dreaming_tick(config: AgentConfig) -> bool:
    if not config.conversation.enabled:
        return False
    threshold = config.conversation.compact.threshold_bytes
    return size_bytes(config.conversation_file) >= threshold


def _is_waking_tick(state: dict) -> bool:
    return bool(state.get("wake_pending"))


def _fetch_channel_context(config: AgentConfig) -> tuple[str, Optional[float], Optional[str]]:
    """If listener enabled, return (formatted_block, kris_age_seconds, newest_message_id)."""
    if not config.discord_listener.enabled:
        return "", None, None
    try:
        from ..listeners.discord_channel import (
            DiscordChannelListener, most_recent_kris_age_seconds,
        )
        listener = DiscordChannelListener(
            channel_id=config.discord_listener.channel_id,
            bot_token_env=config.discord_listener.bot_token_env,
            user_id_to_watch=config.discord_listener.user_id_to_watch,
        )
        messages = listener.fetch_recent(limit=config.discord_listener.recent_message_count)
        if not messages:
            return "(channel quiet)", None, None
        formatted = listener.format_for_prompt(messages)
        age = most_recent_kris_age_seconds(messages, config.discord_listener.user_id_to_watch)
        newest_id = messages[0].id if messages else None
        return formatted, age, newest_id
    except Exception as e:
        log.warning("channel fetch failed: %s", e)
        return "", None, None


def tick_once(
    config: AgentConfig,
    runner: Optional[Runner] = None,
    extra_observation: Optional[str] = None,
    persist_memory: bool = True,
) -> dict:
    """Execute one tick. Returns the new state dict."""
    if config.stop_flag_path.exists():
        log.info("STOP flag present; refusing to tick.")
        return load_state(config.state_file, config.heartbeat.cadence.default_seconds)

    frozen = config.freeze_flag_path.exists()

    if runner is None:
        runner = build_runner(config.model)

    tracker = BudgetTracker(config.budget_file, config.heartbeat.budget)
    if config.heartbeat.enabled and tracker.exceeded():
        log.warning("Daily budget exceeded; skipping tick.")
        return load_state(config.state_file, config.heartbeat.cadence.default_seconds)

    state = load_state(config.state_file, config.heartbeat.cadence.default_seconds)

    # --- mode selection ---
    is_dream = _is_dreaming_tick(config)
    is_wake = _is_waking_tick(state) and not is_dream

    if is_dream:
        return _run_dream_tick(config, runner, tracker, state, frozen)
    if is_wake:
        return _run_wake_tick(config, runner, tracker, state, frozen, extra_observation)
    return _run_normal_tick(config, runner, tracker, state, frozen,
                            extra_observation, persist_memory)


def _run_normal_tick(config, runner, tracker, state, frozen,
                     extra_observation, persist_memory) -> dict:
    # Channel context.
    channel_block, kris_age, newest_msg_id = _fetch_channel_context(config)

    # Recall — drive-biased by current state.
    recall_chunks: list[dict] = []
    if config.memory.enabled and config.memory.storage.backend == "sqlite_vec":
        try:
            from ..memory.retrieve import retrieve
            query = state.get("last_observation") or config.name
            recall_chunks = retrieve(
                config.memory_db_file, query, config.memory,
                current_state=state.get("state"),
            )
        except Exception as e:
            log.warning("Memory recall failed: %s", e)

    # Build prompt + optional conversation history.
    system, user = build_prompt(
        config, state, recall_chunks, extra_observation,
        channel_context=channel_block or None,
    )

    history = None
    if config.conversation.enabled:
        recent_turns = load_recent(config.conversation_file,
                                    config.conversation.max_turns_in_prompt)
        history = to_message_list(recent_turns) if recent_turns else None

    result = runner.run(system, user, history=history)
    tracker.record(result.input_tokens, result.output_tokens)

    new_state_name, new_intensity, next_delay = _parse_response(
        result.text,
        default_state=state.get("state", "BOREDOM"),
        default_intensity=state.get("intensity", 1),
        default_delay=config.heartbeat.cadence.default_seconds,
    )

    # Modulation cascade.
    fs_mult = compute_activity_multiplier(config.workspace, config.heartbeat.modulation)
    next_delay = int(next_delay * fs_mult)
    next_delay = discord_active_delay(kris_age, config.discord_cadence, next_delay)
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
    new_state["wake_pending"] = False  # explicit; this is a normal tick

    if not frozen:
        write_state_atomic(config.state_file, new_state)

        # Working memory: append user + assistant turn.
        if config.conversation.enabled:
            user_for_log = extra_observation or "(tick — no operator input)"
            append_turn(config.conversation_file, "user", user_for_log, source="tick")
            append_turn(config.conversation_file, "assistant", result.text, source="tick")

        # Long-term memory: ingest the response as a chunk (v0.1 behavior).
        if persist_memory and config.memory.enabled and config.memory.storage.backend == "sqlite_vec":
            try:
                from ..memory.score import remember
                remember(config.memory_db_file, result.text, config.memory,
                         source=f"tick:{new_state_name}")
            except Exception as e:
                log.warning("Memory ingest failed: %s", e)

        # Track last-seen Discord message id.
        if newest_msg_id and config.discord_listener.enabled:
            try:
                from ..listeners.discord_channel import save_last_seen
                save_last_seen(config.last_seen_file, newest_msg_id)
            except Exception as e:
                log.warning("save_last_seen failed: %s", e)

    transport = build_transport(config.transport, name=config.name)
    transport.post(result.text)
    return new_state


def _run_dream_tick(config, runner, tracker, state, frozen) -> dict:
    """DREAMING tick: distill conversation into SDT-scored chunks + MEMORY.md."""
    from ..memory.sleep import (
        build_sleep_prompt, parse_dream, ingest_dream, post_strip_dream,
    )

    history_bytes = size_bytes(config.conversation_file)
    threshold = config.conversation.compact.threshold_bytes
    log.info("DREAMING tick — history=%sb threshold=%sb", history_bytes, threshold)

    system = build_sleep_prompt(history_bytes, threshold)
    user = (
        "Reflect on the just-ended session, output your moments + memory "
        "carry-forward + state-update in the format specified."
    )

    history = None
    if config.conversation.enabled:
        recent_turns = load_recent(
            config.conversation_file,
            config.conversation.max_turns_in_prompt * 2,  # see more during dreaming
        )
        history = to_message_list(recent_turns) if recent_turns else None

    result = runner.run(system, user, history=history)
    tracker.record(result.input_tokens, result.output_tokens)

    dream = parse_dream(result.text)
    moments_stored, chunk_ids = (0, [])
    if not frozen:
        moments_stored, chunk_ids = ingest_dream(config, dream, history_bytes)

    log.info("dream ingested %s moments (chunk ids: %s)",
             moments_stored, chunk_ids[:5])

    # Update state to DREAMING for this post; set wake_pending so next tick is WAKING.
    new_state = update_state(
        state,
        new_state="DREAMING",
        new_intensity=0,
        next_delay=config.heartbeat.cadence.min_seconds,
        observation=f"dreaming — {moments_stored} moments stored",
    )
    new_state["wake_pending"] = True

    if not frozen:
        write_state_atomic(config.state_file, new_state)
        # Clear conversation — content is now in the vector DB.
        clear(config.conversation_file)

    # Post the sleep message (stripped of MOMENTS/MEMORY/STATE blocks).
    post_text = post_strip_dream(result.text)
    if post_text.strip():
        try:
            transport = build_transport(config.transport, name=config.name)
            transport.post(post_text)
        except Exception as e:
            log.warning("dream post failed: %s", e)

    return new_state


def _run_wake_tick(config, runner, tracker, state, frozen, extra_observation) -> dict:
    """WAKING tick: one grace tick after dreaming. Read wake context, post brief."""
    from ..memory.wake import build_wake_context, build_wake_prompt_block

    wake_ctx = build_wake_context(config)
    wake_block = build_wake_prompt_block(wake_ctx)
    log.info("WAKING tick — wake context built (%d chars)", len(wake_ctx))

    # No conversation history yet — we just cleared it.
    # No channel context either; this tick is for orienting.
    system, user = build_prompt(
        config, state, recall_chunks=None,
        extra_observation=extra_observation or "you are waking",
        wake_context_block=wake_block,
    )

    result = runner.run(system, user, history=None)
    tracker.record(result.input_tokens, result.output_tokens)

    new_state_name, new_intensity, next_delay = _parse_response(
        result.text,
        default_state="REFLECTING",
        default_intensity=1,
        default_delay=240,
    )
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
        observation="just woke",
    )
    new_state["wake_pending"] = False  # consumed

    if not frozen:
        write_state_atomic(config.state_file, new_state)
        # Seed the new conversation with this wake turn.
        if config.conversation.enabled:
            append_turn(config.conversation_file, "user",
                        "wake from sleep", source="tick")
            append_turn(config.conversation_file, "assistant",
                        result.text, source="tick")

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
    log.info("  conversation.enabled    = %s", config.conversation.enabled)
    log.info("  discord_listener.enabled = %s", config.discord_listener.enabled)
    log.info("  memory.enabled          = %s", config.memory.enabled)

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
            log.info("Tick complete; state=%s/%s; sleeping %ss",
                     state.get("state"), state.get("intensity"), delay)
            for _ in range(int(delay)):
                if config.stop_flag_path.exists():
                    return
                time.sleep(1)
    except KeyboardInterrupt:
        log.info("Interrupted; exiting.")
