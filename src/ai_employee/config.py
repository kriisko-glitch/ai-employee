"""Load and validate agent.yaml into typed config dataclasses.

One AgentConfig per agent. Every toggle lives here; the rest of the package
reads from this object rather than re-parsing YAML.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# --- model ------------------------------------------------------------------

# Default base_urls for known providers. `null` means "use the OpenAI SDK
# default" (which is OpenAI's own endpoint).
PROVIDER_DEFAULT_BASE_URL: dict[str, Optional[str]] = {
    "openai": None,
    "anthropic": "https://api.anthropic.com/v1/",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "local": "http://localhost:11434/v1",  # Ollama default
    "custom": None,
}


@dataclass
class ModelConfig:
    provider: str = "openai"
    model_id: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_seconds: int = 120

    @property
    def resolved_base_url(self) -> Optional[str]:
        if self.base_url:
            return self.base_url
        return PROVIDER_DEFAULT_BASE_URL.get(self.provider)


# --- heartbeat --------------------------------------------------------------

@dataclass
class CadenceConfig:
    min_seconds: int = 60
    max_seconds: int = 1800
    default_seconds: int = 300


@dataclass
class ModulationConfig:
    enabled: bool = True
    signals: list[str] = field(default_factory=lambda: ["filesystem"])
    active_window_minutes: int = 5
    idle_window_minutes: int = 60
    active_multiplier: float = 0.5
    idle_multiplier: float = 2.0


@dataclass
class PricingConfig:
    input_per_1m_usd: float = 0.0
    output_per_1m_usd: float = 0.0


@dataclass
class BudgetConfig:
    daily_usd_cap: float = 5.00
    per_tick_token_cap: int = 4000
    pricing: PricingConfig = field(default_factory=PricingConfig)


@dataclass
class HeartbeatConfig:
    enabled: bool = False
    cadence: CadenceConfig = field(default_factory=CadenceConfig)
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)


# --- memory -----------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    device: str = "auto"


@dataclass
class StorageConfig:
    backend: str = "sqlite_vec"  # sqlite_vec | none
    db_filename: str = "memory.db"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class AutoTagConfig:
    enabled: bool = True
    mode: str = "keyword"  # keyword | llm | hybrid


@dataclass
class RetrievalConfig:
    top_k: int = 8
    boost_hits: float = 1.5
    suppress_misses: float = 0.3
    walkback_weight: float = 1.2


@dataclass
class DecayConfig:
    enabled: bool = True
    half_life_days: float = 30.0


@dataclass
class ParameterizationConfig:
    valence_labels: list[str] = field(
        default_factory=lambda: ["hit", "miss", "walkback", "unmarked"]
    )
    auto_tag: AutoTagConfig = field(default_factory=AutoTagConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)


@dataclass
class ConsolidationConfig:
    enabled: bool = True
    trigger_session_bytes: int = 3_000_000
    keep_diff_log: bool = True


@dataclass
class MemoryConfig:
    enabled: bool = True
    storage: StorageConfig = field(default_factory=StorageConfig)
    parameterization: ParameterizationConfig = field(default_factory=ParameterizationConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)


# --- transport --------------------------------------------------------------

@dataclass
class DiscordTransportConfig:
    webhook_url_env: str = "DISCORD_WEBHOOK_URL"
    username: Optional[str] = None


@dataclass
class WebhookTransportConfig:
    url_env: str = "GENERIC_WEBHOOK_URL"
    format: str = "json"  # json | text


@dataclass
class TransportConfig:
    kind: str = "stdout"  # stdout | discord | webhook
    discord: DiscordTransportConfig = field(default_factory=DiscordTransportConfig)
    webhook: WebhookTransportConfig = field(default_factory=WebhookTransportConfig)


# --- safety -----------------------------------------------------------------

@dataclass
class SafetyConfig:
    tier: int = 1
    drafts_dir: str = "drafts"
    stop_flag: str = "STOP"
    freeze_flag: str = "FREEZE"


# --- top-level --------------------------------------------------------------

@dataclass
class AgentConfig:
    name: str
    workspace: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Derived workspace paths
    @property
    def soul_file(self) -> Path:
        return self.workspace / "SOUL.md"

    @property
    def memory_file(self) -> Path:
        return self.workspace / "MEMORY.md"

    @property
    def state_file(self) -> Path:
        return self.workspace / "state.json"

    @property
    def memory_db_file(self) -> Path:
        return self.workspace / self.memory.storage.db_filename

    @property
    def stop_flag_path(self) -> Path:
        return self.workspace / self.safety.stop_flag

    @property
    def freeze_flag_path(self) -> Path:
        return self.workspace / self.safety.freeze_flag

    @property
    def drafts_path(self) -> Path:
        return self.workspace / self.safety.drafts_dir

    @property
    def budget_file(self) -> Path:
        return self.workspace / "budget.json"


# --- loader -----------------------------------------------------------------

def _merge_dataclass(cls, data: Optional[dict]) -> Any:
    """Build a (possibly nested) dataclass from a dict, ignoring unknown keys.

    Nested dataclass fields are recursed into when the YAML key is a dict.
    """
    import dataclasses

    if data is None:
        return cls()
    if not dataclasses.is_dataclass(cls):
        return data

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if dataclasses.is_dataclass(f.type):
            kwargs[f.name] = _merge_dataclass(f.type, value)
        elif isinstance(f.default_factory, type) and dataclasses.is_dataclass(  # type: ignore[misc]
            f.default_factory
        ):
            kwargs[f.name] = _merge_dataclass(f.default_factory, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_agent_config(config_path: Path, repo_root: Optional[Path] = None) -> AgentConfig:
    """Load an agent.yaml file into an AgentConfig.

    Workspace paths are resolved relative to `repo_root` (defaults to the
    config file's grandparent, e.g. `agents/example/agent.yaml` → repo root
    is two levels up).
    """
    config_path = Path(config_path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    if repo_root is None:
        repo_root = config_path.parent.parent.parent

    name = data.get("name") or config_path.parent.name
    workspace_raw = data.get("workspace") or f"agents/{name}"
    workspace = Path(workspace_raw)
    if not workspace.is_absolute():
        workspace = (repo_root / workspace).resolve()

    # Build nested configs manually so dataclass nesting works robustly.
    import dataclasses

    def build(cls, key):
        return _merge_dataclass(cls, data.get(key) or {})

    cfg = AgentConfig(
        name=name,
        workspace=workspace,
        model=build(ModelConfig, "model"),
        heartbeat=_build_heartbeat(data.get("heartbeat") or {}),
        memory=_build_memory(data.get("memory") or {}),
        transport=_build_transport(data.get("transport") or {}),
        safety=build(SafetyConfig, "safety"),
    )
    return cfg


def _build_heartbeat(d: dict) -> HeartbeatConfig:
    return HeartbeatConfig(
        enabled=d.get("enabled", False),
        cadence=_merge_dataclass(CadenceConfig, d.get("cadence")),
        modulation=_merge_dataclass(ModulationConfig, d.get("modulation")),
        budget=_build_budget(d.get("budget") or {}),
    )


def _build_budget(d: dict) -> BudgetConfig:
    return BudgetConfig(
        daily_usd_cap=d.get("daily_usd_cap", 5.00),
        per_tick_token_cap=d.get("per_tick_token_cap", 4000),
        pricing=_merge_dataclass(PricingConfig, d.get("pricing")),
    )


def _build_memory(d: dict) -> MemoryConfig:
    storage_d = d.get("storage") or {}
    storage = StorageConfig(
        backend=storage_d.get("backend", "sqlite_vec"),
        db_filename=storage_d.get("db_filename", "memory.db"),
        embedding=_merge_dataclass(EmbeddingConfig, storage_d.get("embedding")),
    )
    param_d = d.get("parameterization") or {}
    parameterization = ParameterizationConfig(
        valence_labels=param_d.get("valence_labels")
        or ["hit", "miss", "walkback", "unmarked"],
        auto_tag=_merge_dataclass(AutoTagConfig, param_d.get("auto_tag")),
        retrieval=_merge_dataclass(RetrievalConfig, param_d.get("retrieval")),
        decay=_merge_dataclass(DecayConfig, param_d.get("decay")),
    )
    return MemoryConfig(
        enabled=d.get("enabled", True),
        storage=storage,
        parameterization=parameterization,
        consolidation=_merge_dataclass(ConsolidationConfig, d.get("consolidation")),
    )


def _build_transport(d: dict) -> TransportConfig:
    return TransportConfig(
        kind=d.get("kind", "stdout"),
        discord=_merge_dataclass(DiscordTransportConfig, d.get("discord")),
        webhook=_merge_dataclass(WebhookTransportConfig, d.get("webhook")),
    )


def get_api_key(env_var: str) -> str:
    """Read a required API key from the environment. Raises if missing."""
    val = os.environ.get(env_var)
    if not val:
        raise RuntimeError(
            f"Environment variable {env_var!r} is not set. "
            f"Add it to your .env file."
        )
    return val
