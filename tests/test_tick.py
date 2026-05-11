"""Tick loop — with a mock runner so no real API is called."""
from pathlib import Path

from ai_employee.config import load_agent_config
from ai_employee.heartbeat.tick import tick_once, _parse_response
from ai_employee.runner.base import RunResult


def test_parse_response_state_header():
    state, intensity, delay = _parse_response(
        "[BUILDING 2]\n\nSome body text.", "X", 0, 300
    )
    assert state == "BUILDING"
    assert intensity == 2
    assert delay == 300  # default — no next_tick_seconds


def test_parse_response_with_next_tick():
    state, intensity, delay = _parse_response(
        "[REFLECTING 3]\n\nBody.\n\n[next_tick_seconds: 90]", "X", 0, 300
    )
    assert state == "REFLECTING"
    assert intensity == 3
    assert delay == 90


def test_parse_response_no_header_falls_back():
    state, intensity, delay = _parse_response(
        "Just a plain response with no header.", "DEFAULT", 1, 300
    )
    assert state == "DEFAULT"
    assert intensity == 1


class MockRunner:
    """Inert runner — no network calls."""

    def __init__(self, response: str = "[REFLECTING 1]\n\nMocked tick."):
        self.response = response
        self.called = 0

    def run(self, system: str, user: str) -> RunResult:
        self.called += 1
        return RunResult(
            text=self.response,
            input_tokens=100,
            output_tokens=50,
            model_id="mock",
        )


def _make_agent(tmp_path: Path, memory_enabled: bool = False):
    repo = tmp_path
    (repo / "pyproject.toml").touch()
    agent_dir = repo / "agents" / "t"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("test agent")
    (agent_dir / "MEMORY.md").write_text("none")
    (agent_dir / "agent.yaml").write_text(f"""\
name: t
workspace: agents/t
model:
  provider: openai
  model_id: gpt-test
  api_key_env: FAKE_KEY
heartbeat:
  enabled: false
  budget:
    daily_usd_cap: 100.0
memory:
  enabled: {str(memory_enabled).lower()}
transport:
  kind: stdout
""")
    return load_agent_config(agent_dir / "agent.yaml", repo_root=repo)


def test_tick_once_writes_state(tmp_path, capsys):
    cfg = _make_agent(tmp_path, memory_enabled=False)
    runner = MockRunner("[BUILDING 2]\n\nWorked on the thing.")
    new = tick_once(cfg, runner=runner, persist_memory=False)
    assert runner.called == 1
    assert new["state"] == "BUILDING"
    assert new["intensity"] == 2
    assert cfg.state_file.exists()


def test_tick_respects_stop_flag(tmp_path):
    cfg = _make_agent(tmp_path)
    cfg.stop_flag_path.write_text("halt")
    runner = MockRunner()
    tick_once(cfg, runner=runner, persist_memory=False)
    # Runner should not have been called.
    assert runner.called == 0
