"""aie — the AI Employee CLI.

Commands:
    aie init <name>          scaffold a new agent dir
    aie tick <name>          run one stateless tick
    aie run <name>           start the heartbeat loop
    aie status <name>        show state + budget
    aie remember <name> <text>            add a memory chunk
    aie recall <name> <query>             search memory
    aie tag <name> <chunk_id> <valence>   override a chunk's valence
    aie attempt <name> --task ... --approach ... --outcome hit|miss|walkback
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import AgentConfig, load_agent_config


def _find_repo_root() -> Path:
    """Walk up from cwd looking for agents/ + pyproject.toml. Fallback to cwd."""
    env_home = os.environ.get("AIE_HOME")
    if env_home:
        return Path(env_home).resolve()
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "agents").is_dir() and (p / "pyproject.toml").is_file():
            return p
    return cur


def _load(name: str) -> AgentConfig:
    repo_root = _find_repo_root()
    cfg_path = repo_root / "agents" / name / "agent.yaml"
    if not cfg_path.exists():
        sys.exit(f"error: no agent.yaml at {cfg_path}\n"
                 f"Run `aie init {name}` to scaffold one, or set AIE_HOME.")
    return load_agent_config(cfg_path, repo_root=repo_root)


def _setup_logging() -> None:
    level = os.environ.get("AIE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# --- commands ---------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    repo_root = _find_repo_root()
    target = repo_root / "agents" / args.name
    if target.exists():
        print(f"agent already exists: {target}", file=sys.stderr)
        return 1
    example = repo_root / "agents" / "example"
    if example.exists():
        shutil.copytree(example, target)
    else:
        target.mkdir(parents=True)
    # Copy/customize agent.yaml.example.
    example_yaml = repo_root / "agent.yaml.example"
    if example_yaml.exists() and not (target / "agent.yaml").exists():
        content = example_yaml.read_text(encoding="utf-8")
        content = content.replace("name: example", f"name: {args.name}")
        content = content.replace(
            "workspace: agents/example",
            f"workspace: agents/{args.name}",
        )
        (target / "agent.yaml").write_text(content, encoding="utf-8")
    print(f"scaffolded {target}")
    print(f"next: edit {target / 'agent.yaml'} and {target / 'SOUL.md'}")
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    cfg = _load(args.name)
    from .heartbeat.tick import tick_once
    new_state = tick_once(cfg, extra_observation=args.observation,
                          persist_memory=not args.no_memory)
    print(f"[ok] state: {new_state['state']} {new_state['intensity']}  "
          f"next: {new_state['next_tick_delay_seconds']}s")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args.name)
    from .heartbeat.tick import run_loop
    run_loop(cfg)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _load(args.name)
    from .heartbeat.state import load_state
    from .heartbeat.budget import BudgetTracker

    state = load_state(cfg.state_file, cfg.heartbeat.cadence.default_seconds)
    budget = BudgetTracker(cfg.budget_file, cfg.heartbeat.budget).status()

    print(f"agent:      {cfg.name}")
    print(f"workspace:  {cfg.workspace}")
    print(f"model:      {cfg.model.provider}:{cfg.model.model_id}")
    print(f"heartbeat:  {'on' if cfg.heartbeat.enabled else 'off'}")
    print(f"memory:     {'on' if cfg.memory.enabled else 'off'} "
          f"({cfg.memory.storage.backend})")
    print(f"transport:  {cfg.transport.kind}")
    print(f"tier:       T{cfg.safety.tier}")
    print()
    print(f"state:      {state['state']} {state['intensity']}")
    print(f"since:      {state['since_ts']}")
    print(f"last tick:  {state.get('last_tick_ts') or '(never)'}")
    print(f"next delay: {state['next_tick_delay_seconds']}s")
    print()
    print(f"budget today ({budget.date}):  "
          f"${budget.spend_usd:.4f} / ${cfg.heartbeat.budget.daily_usd_cap:.2f}")
    print(f"tokens today:  in={budget.input_tokens}  out={budget.output_tokens}")

    if cfg.stop_flag_path.exists():
        print("\n⚠  STOP flag is set — heartbeat will not tick.")
    if cfg.freeze_flag_path.exists():
        print("\n⚠  FREEZE flag is set — heartbeat will tick but not write.")
    return 0


def cmd_remember(args: argparse.Namespace) -> int:
    cfg = _load(args.name)
    if not cfg.memory.enabled:
        sys.exit("error: memory is disabled in agent.yaml")
    from .memory.score import remember
    chunk_id = remember(
        cfg.memory_db_file, args.text, cfg.memory,
        source=args.source, valence=args.valence,
    )
    print(f"stored chunk #{chunk_id}")
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    cfg = _load(args.name)
    if not cfg.memory.enabled:
        sys.exit("error: memory is disabled in agent.yaml")
    from .memory.retrieve import retrieve
    chunks = retrieve(cfg.memory_db_file, args.query, cfg.memory)
    if not chunks:
        print("(no results)")
        return 0
    for c in chunks:
        body = c["body"].replace("\n", " ")
        if len(body) > 140:
            body = body[:137] + "..."
        print(f"#{c['id']} [{c['valence']}] score={c['score']:.3f}  {body}")
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    cfg = _load(args.name)
    if not cfg.memory.enabled:
        sys.exit("error: memory is disabled in agent.yaml")
    valid = set(cfg.memory.parameterization.valence_labels)
    if args.valence not in valid:
        sys.exit(f"error: valence must be one of {sorted(valid)}")
    from .memory.db import connect, init_schema, set_valence
    conn = connect(cfg.memory_db_file)
    try:
        init_schema(conn, cfg.memory.storage.embedding.dim)
        set_valence(conn, args.chunk_id, args.valence, reason=args.reason)
    finally:
        conn.close()
    print(f"chunk #{args.chunk_id} → {args.valence}")
    return 0


def cmd_attempt(args: argparse.Namespace) -> int:
    cfg = _load(args.name)
    if not cfg.memory.enabled:
        sys.exit("error: memory is disabled in agent.yaml")
    from .memory.score import log_attempt
    chunk_id = log_attempt(
        cfg.memory_db_file,
        task=args.task,
        approach=args.approach,
        outcome=args.outcome,
        config=cfg.memory,
        lesson=args.lesson,
    )
    print(f"attempt logged; paired chunk #{chunk_id}")
    return 0


# --- entry point ------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # read .env from cwd or upward
    _setup_logging()

    parser = argparse.ArgumentParser(prog="aie", description="AI Employee CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="scaffold a new agent")
    p_init.add_argument("name")
    p_init.set_defaults(func=cmd_init)

    p_tick = sub.add_parser("tick", help="run one stateless tick")
    p_tick.add_argument("name")
    p_tick.add_argument("--observation", "-o", default=None,
                        help="extra observation to feed the prompt")
    p_tick.add_argument("--no-memory", action="store_true",
                        help="don't persist the response as a memory chunk")
    p_tick.set_defaults(func=cmd_tick)

    p_run = sub.add_parser("run", help="start the heartbeat loop")
    p_run.add_argument("name")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="show state + budget")
    p_status.add_argument("name")
    p_status.set_defaults(func=cmd_status)

    p_remember = sub.add_parser("remember", help="add a memory chunk")
    p_remember.add_argument("name")
    p_remember.add_argument("text")
    p_remember.add_argument("--source", default=None)
    p_remember.add_argument("--valence", default=None,
                            help="hit | miss | walkback | unmarked (auto-tagged if omitted)")
    p_remember.set_defaults(func=cmd_remember)

    p_recall = sub.add_parser("recall", help="search memory")
    p_recall.add_argument("name")
    p_recall.add_argument("query")
    p_recall.set_defaults(func=cmd_recall)

    p_tag = sub.add_parser("tag", help="override a chunk's valence")
    p_tag.add_argument("name")
    p_tag.add_argument("chunk_id", type=int)
    p_tag.add_argument("valence")
    p_tag.add_argument("--reason", default=None)
    p_tag.set_defaults(func=cmd_tag)

    p_attempt = sub.add_parser("attempt", help="record a solution_attempt")
    p_attempt.add_argument("name")
    p_attempt.add_argument("--task", required=True)
    p_attempt.add_argument("--approach", required=True)
    p_attempt.add_argument("--outcome", required=True,
                           choices=["hit", "miss", "walkback"])
    p_attempt.add_argument("--lesson", default=None)
    p_attempt.set_defaults(func=cmd_attempt)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
