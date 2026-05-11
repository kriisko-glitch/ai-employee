"""Config loader — YAML parsing, defaults, provider base URL resolution."""
from pathlib import Path

from ai_employee.config import (
    PROVIDER_DEFAULT_BASE_URL, ModelConfig, load_agent_config,
)


def test_provider_base_urls_known():
    assert PROVIDER_DEFAULT_BASE_URL["openai"] is None
    assert "anthropic.com" in PROVIDER_DEFAULT_BASE_URL["anthropic"]
    assert "deepseek.com" in PROVIDER_DEFAULT_BASE_URL["deepseek"]
    assert "groq.com" in PROVIDER_DEFAULT_BASE_URL["groq"]


def test_model_config_resolved_base_url_uses_provider_default():
    m = ModelConfig(provider="anthropic", model_id="claude-sonnet-4-6",
                    api_key_env="ANTHROPIC_API_KEY")
    assert m.resolved_base_url == PROVIDER_DEFAULT_BASE_URL["anthropic"]


def test_model_config_explicit_override():
    m = ModelConfig(provider="openai", base_url="http://localhost:8080/v1")
    assert m.resolved_base_url == "http://localhost:8080/v1"


def test_load_agent_config(tmp_path: Path):
    # Build a tiny repo layout.
    repo = tmp_path
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    (repo / "agents" / "demo").mkdir(parents=True)
    cfg_yaml = repo / "agents" / "demo" / "agent.yaml"
    cfg_yaml.write_text("""\
name: demo
workspace: agents/demo
model:
  provider: deepseek
  model_id: deepseek-chat
  api_key_env: DEEPSEEK_API_KEY
  temperature: 0.3
heartbeat:
  enabled: true
  cadence:
    min_seconds: 30
    max_seconds: 600
memory:
  enabled: true
  parameterization:
    retrieval:
      boost_hits: 2.0
transport:
  kind: discord
safety:
  tier: 2
""")
    cfg = load_agent_config(cfg_yaml, repo_root=repo)
    assert cfg.name == "demo"
    assert cfg.model.provider == "deepseek"
    assert cfg.model.temperature == 0.3
    assert cfg.heartbeat.enabled is True
    assert cfg.heartbeat.cadence.min_seconds == 30
    assert cfg.memory.parameterization.retrieval.boost_hits == 2.0
    assert cfg.transport.kind == "discord"
    assert cfg.safety.tier == 2
    # Default fall-through for keys not in the yaml.
    assert cfg.memory.parameterization.retrieval.suppress_misses == 0.3


def test_workspace_paths_resolve(tmp_path: Path):
    repo = tmp_path
    (repo / "pyproject.toml").touch()
    (repo / "agents" / "demo").mkdir(parents=True)
    cfg_yaml = repo / "agents" / "demo" / "agent.yaml"
    cfg_yaml.write_text("name: demo\nworkspace: agents/demo\n")
    cfg = load_agent_config(cfg_yaml, repo_root=repo)
    assert cfg.soul_file == repo / "agents" / "demo" / "SOUL.md"
    assert cfg.state_file == repo / "agents" / "demo" / "state.json"
    assert cfg.memory_db_file == repo / "agents" / "demo" / "memory.db"
