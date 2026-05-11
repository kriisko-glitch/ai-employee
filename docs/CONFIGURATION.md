# Configuration reference

Every knob, what it does, default value.

## `name` (string, required)

Display name and folder identifier for this agent.

## `workspace` (path, default `agents/<name>`)

Where this agent's files live. Relative paths resolve against the repo root. Absolute paths are taken as-is. The workspace contains:

- `SOUL.md`, `MEMORY.md` — markdown identity + persistent notes
- `agent.yaml` — this file
- `state.json` — auto-managed heartbeat state
- `memory.db` — sqlite memory DB (if `memory.enabled`)
- `budget.json` — daily spend tracker
- `drafts/` — where draft outputs land (tier T1)

---

## `model`

| Field | Type | Default | Notes |
|---|---|---|---|
| `provider` | str | `openai` | See [PROVIDERS.md](PROVIDERS.md) |
| `model_id` | str | `gpt-4o-mini` | Provider-specific |
| `api_key_env` | str | `OPENAI_API_KEY` | Name of env var in `.env` |
| `base_url` | str/null | null | Override the provider default |
| `temperature` | float | 0.7 | |
| `max_tokens` | int | 2048 | Per-tick cap on output |
| `timeout_seconds` | int | 120 | HTTP timeout |

---

## `heartbeat`

Off by default. When enabled, the agent runs an autonomous tick loop.

### `heartbeat.enabled` (bool, default `false`)

Off → agent is stateless / on-demand only (use `aie tick`).
On → `aie run` starts the loop.

### `heartbeat.cadence`

| Field | Default | Notes |
|---|---|---|
| `min_seconds` | 60 | Floor on tick delay |
| `max_seconds` | 1800 | Ceiling on tick delay |
| `default_seconds` | 300 | Initial delay at boot |

### `heartbeat.modulation`

Adjust tick delay based on user activity around the workspace.

| Field | Default | Notes |
|---|---|---|
| `enabled` | true | |
| `signals` | `[filesystem]` | `filesystem`, `git`, or both |
| `active_window_minutes` | 5 | Activity within this window = "active" |
| `idle_window_minutes` | 60 | Nothing for this long = "idle" |
| `active_multiplier` | 0.5 | Tick faster when user is around |
| `idle_multiplier` | 2.0 | Tick slower when user is gone |

### `heartbeat.budget`

| Field | Default | Notes |
|---|---|---|
| `daily_usd_cap` | 5.00 | Hard kill-switch. 0 = unlimited (not recommended). |
| `per_tick_token_cap` | 4000 | Advisory; the model's `max_tokens` is the hard limit. |
| `pricing.input_per_1m_usd` | 0.0 | USD per 1M input tokens — fill from your provider's rate card |
| `pricing.output_per_1m_usd` | 0.0 | USD per 1M output tokens |

---

## `memory`

### `memory.enabled` (bool, default `true`)

Off → only `MEMORY.md` is used (no vector DB, no decay, no valence retrieval).
On → sqlite + sqlite-vec backs long-term memory.

### `memory.storage`

| Field | Default | Notes |
|---|---|---|
| `backend` | `sqlite_vec` | `sqlite_vec` or `none` |
| `db_filename` | `memory.db` | Relative to workspace |
| `embedding.model` | `BAAI/bge-small-en-v1.5` | Any sentence-transformers model |
| `embedding.dim` | 384 | Must match the chosen model |
| `embedding.device` | `auto` | `auto`/`cpu`/`cuda`/`mps` |

### `memory.parameterization`

Controls how memories are labeled and retrieved.

| Field | Default | Notes |
|---|---|---|
| `valence_labels` | `[hit, miss, walkback, unmarked]` | Allowed labels for `aie tag` |
| `auto_tag.enabled` | true | Heuristic tagger runs on ingest |
| `auto_tag.mode` | `keyword` | `keyword`, `llm`, or `hybrid` |
| `retrieval.top_k` | 8 | How many chunks the recall query returns |
| `retrieval.boost_hits` | 1.5 | Score multiplier for `hit` chunks |
| `retrieval.suppress_misses` | 0.3 | Multiplier for `miss` chunks |
| `retrieval.walkback_weight` | 1.2 | Multiplier for `walkback` chunks |
| `decay.enabled` | true | Exponential decay applied at recall |
| `decay.half_life_days` | 30 | Retrieval weight halves every N days |

### `memory.consolidation`

Periodic "sleep" — distill long session logs into `MEMORY.md`.

| Field | Default | Notes |
|---|---|---|
| `enabled` | true | |
| `trigger_session_bytes` | 3,000,000 | Consolidate when session log exceeds |
| `keep_diff_log` | true | Log what was kept vs dropped |

---

## `transport`

### `transport.kind` (default `stdout`)

- `stdout` — print to terminal. Default. Good for testing.
- `discord` — POST to a webhook URL. Requires `DISCORD_WEBHOOK_URL` in `.env`.
- `webhook` — generic POST. Works for Slack-incoming-webhooks, n8n, Zapier.

### `transport.discord`

| Field | Default | Notes |
|---|---|---|
| `webhook_url_env` | `DISCORD_WEBHOOK_URL` | Name of env var in `.env` |
| `username` | (agent name) | Display name on Discord |

### `transport.webhook`

| Field | Default | Notes |
|---|---|---|
| `url_env` | `GENERIC_WEBHOOK_URL` | Name of env var in `.env` |
| `format` | `json` | `json` (body = `{"text": ...}`) or `text` (raw body) |

---

## `safety`

| Field | Default | Notes |
|---|---|---|
| `tier` | 1 | 0 read-only, 1 draft-only, 2 supervised, 3 trusted, 4 autonomous |
| `drafts_dir` | `drafts` | Where T1 outputs land |
| `stop_flag` | `STOP` | Filename; presence halts loop |
| `freeze_flag` | `FREEZE` | Filename; presence pauses writes |

To stop a running heartbeat cleanly: `touch agents/<name>/STOP`.
