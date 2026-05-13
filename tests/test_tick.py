"""Tick loop — with a mock runner so no real API is called.

Parser-specific tests now live in test_response_parse.py since the parser
was extracted to a dedicated module.
"""
from pathlib import Path

from ai_employee.config import load_agent_config
from ai_employee.heartbeat.tick import tick_once
from ai_employee.runner.base import RunResult


class MockRunner:
    """Inert runner — no network calls."""

    def __init__(self, response: str = "[REFLECTING 1]\n\nMocked tick."):
        self.response = response
        self.called = 0
        self.last_history = None

    def run(self, system: str, user: str, history=None) -> RunResult:
        self.called += 1
        self.last_history = history
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
    # Use the new structured format the prompt now requests.
    runner = MockRunner(
        "[THINKING]\nweighing options\n[/THINKING]\n"
        "[POST]\nworked on the thing.\n[/POST]\n"
        "[BUILDING 2]\n[next_tick_seconds: 300]\n"
    )
    new = tick_once(cfg, runner=runner, persist_memory=False)
    assert runner.called == 1
    assert new["state"] == "BUILDING"
    assert new["intensity"] == 2
    assert cfg.state_file.exists()


def test_tick_silent_response_does_not_post(tmp_path, capsys):
    """When the model chooses SILENT, no transport.post should fire."""
    cfg = _make_agent(tmp_path, memory_enabled=False)
    runner = MockRunner(
        "[THINKING]\nnothing to say\n[/THINKING]\n"
        "[POST]\nSILENT\n[/POST]\n"
        "[BOREDOM 1]\n[next_tick_seconds: 1800]\n"
    )
    tick_once(cfg, runner=runner, persist_memory=False)
    out = capsys.readouterr().out
    # Stdout transport posts a "── name @ ts ──" banner; ensure none appeared.
    assert "── t @" not in out


def test_tick_legacy_format_still_works(tmp_path, capsys):
    """A response with no THINKING/POST blocks should still produce a post —
    backward compatibility for legacy v0.1/early-v0.2 agents."""
    cfg = _make_agent(tmp_path, memory_enabled=False)
    runner = MockRunner("[REFLECTING 1]\n\nplain legacy body.")
    tick_once(cfg, runner=runner, persist_memory=False)
    out = capsys.readouterr().out
    assert "plain legacy body" in out


def test_tick_respects_stop_flag(tmp_path):
    cfg = _make_agent(tmp_path)
    cfg.stop_flag_path.write_text("halt")
    runner = MockRunner()
    tick_once(cfg, runner=runner, persist_memory=False)
    # Runner should not have been called.
    assert runner.called == 0
