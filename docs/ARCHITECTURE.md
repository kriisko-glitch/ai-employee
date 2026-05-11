# Architecture

A bird's-eye view of how `ai-employee` is wired.

```
                  ┌──────────────────────────────────────┐
                  │            agent.yaml                │
                  │  (one config file, every toggle)     │
                  └────────────────┬─────────────────────┘
                                   │ load_agent_config()
                                   ▼
                  ┌──────────────────────────────────────┐
                  │            AgentConfig               │
                  │  ModelConfig | HeartbeatConfig |     │
                  │  MemoryConfig | TransportConfig |    │
                  │  SafetyConfig                        │
                  └────────────────┬─────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌────────────────┐        ┌────────────────┐         ┌────────────────┐
│ Runner         │        │ Heartbeat      │         │ Transport      │
│                │        │                │         │                │
│ OpenAI-compat  │◄──────►│ tick loop      │────────►│ stdout         │
│ (openai SDK    │        │ state.json     │         │ discord        │
│  with base_url │        │ budget         │         │ webhook        │
│  override)     │        │ modulation     │         │                │
└────────────────┘        └───────┬────────┘         └────────────────┘
                                  │
                                  ▼
                          ┌────────────────┐
                          │ Memory         │
                          │                │
                          │ sqlite + vec0  │
                          │ valence tagger │
                          │ decay-weighted │
                          │ retrieval      │
                          └────────────────┘
```

## Modules

### `config`
- Dataclasses for every section of `agent.yaml`.
- `load_agent_config(path)` → `AgentConfig`.
- `PROVIDER_DEFAULT_BASE_URL` table maps friendly names → endpoint URLs.

### `runner`
- `Runner` protocol: one method, `run(system, user) → RunResult`.
- `OpenAICompatibleRunner` wraps the `openai` Python SDK with a `base_url` override. Works for every OpenAI-shaped API.
- `build_runner(ModelConfig)` reads the API key env var and constructs an instance.

### `memory`
SQLite + sqlite-vec, one DB per agent.

**Tables:**

| Table | Purpose |
|---|---|
| `chunk` | Every memory unit. Fields: `body`, `valence`, `weight`, `ts`, `embedding_model`, `recall_count`, `last_recalled_ts`. |
| `chunk_vec` | `vec0` virtual table — embeddings (default 384-dim, BAAI/bge-small-en-v1.5). |
| `solution_attempt` | Explicit `(task, approach, outcome, lesson)` rows. The structured half of the "hit/miss" model. |
| `valence_override` | Manual re-tags from `aie tag`. |

**Retrieval:**

```
score = similarity × valence_weight × decay_factor
```

- `similarity = max(0, 1 - L2_distance)` (sqlite-vec normalized)
- `valence_weight`: hit ×1.5, walkback ×1.2, miss ×0.3, unmarked ×1.0 (configurable)
- `decay_factor = 0.5 ^ (age_days / half_life_days)`

Top-K candidates are pulled with kNN, then re-ranked by the formula above. Hit chunks surface higher; miss chunks surface as warnings.

### `heartbeat`
Off by default. When `heartbeat.enabled: true`:

```
each tick:
  if STOP flag exists: exit
  if FREEZE flag exists: tick but don't write
  if budget.exceeded(): exit
  state ← load state.json
  recall ← retrieve(query=state.last_observation)
  prompt ← build_prompt(SOUL, MEMORY, state, recall)
  result ← runner.run(prompt)
  budget.record(result.tokens)
  state ← parse + atomic write
  memory ← remember(result.text)
  transport.post(result.text)
  sleep(state.next_tick_delay × activity_multiplier)
```

**Modulation:** the multiplier is < 1.0 when the user is active (recent file edits or git commits in the workspace) and > 1.0 when idle. The agent breathes with you.

**Atomic writes:** every state.json update writes a temp file and `os.replace`s. Crash-safe.

### `transport`
- `StdoutTransport` — default; prints to terminal.
- `DiscordTransport` — POSTs to a webhook URL (splits long messages).
- `WebhookTransport` — generic POST for Slack-incoming, n8n, Zapier.

Adding a new transport: implement the `Transport` protocol (one method, `post(text)`).

## Safety

- **Tiers (T0–T4)** in `safety.tier`. Default T1 means drafts-only.
- **`STOP` flag**: presence of a `STOP` file in workspace → loop exits.
- **`FREEZE` flag**: presence of `FREEZE` → loop continues, no writes.
- **Daily budget**: `BudgetTracker` hard-stops the loop when today's spend exceeds the cap.
- **No telemetry**: zero outbound calls except to your configured provider and transport.

## What's NOT included (by design)

This is the minimum viable kernel. Out-of-scope for v0.1:

- Multi-agent coordination
- Function/tool calling (use the model's native tool-use if you need it)
- Web UI / dashboard
- Distributed/cloud deployment
- Embodiment / sensor layers
