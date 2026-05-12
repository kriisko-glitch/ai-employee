# AI Employee

A minimal, OpenAI-compatible framework for building stateful, optionally-autonomous AI agents.

Toggle a heartbeat on or off. Plug any OpenAI-compatible API (OpenAI, Anthropic, DeepSeek, Groq, Cerebras, OpenRouter, Ollama, llama.cpp, etc.). Give the agent long-term memory with valence-tagged retrieval — so it learns what worked vs. what didn't.

Everything lives in one `agent.yaml` per agent. No hidden state, no vendor lock-in.

---

## Quick start

```bash
git clone https://github.com/kriisko-glitch/ai-employee.git
cd ai-employee

# Install with the memory extras (sqlite-vec + sentence-transformers).
# Memory is on by default, so the [memory] extra is the recommended path.
pip install -e ".[memory]"               # or: uv sync --extra memory

# Configure
cp .env.example .env                     # paste API keys

# The example agent is already scaffolded at agents/example/.
# Edit agents/example/agent.yaml to pick your provider, then:

aie tick example                         # one-shot stateless tick
aie status example                       # state + budget snapshot
aie recall example "what did I learn?"   # search memory

# Scaffold a new agent of your own:
aie init my-agent
# Edit agents/my-agent/agent.yaml and agents/my-agent/SOUL.md, then:
aie tick my-agent
```

**Windows users:** if you see a `UnicodeEncodeError` on output, you're on the cp1252 default console. Recent versions of `aie` reconfigure stdout to UTF-8 automatically, but if you're on an older clone, run `chcp 65001` once per session or set `PYTHONIOENCODING=utf-8`.

---

## What's in the box

| Feature | Default | Toggle |
|---|---|---|
| **Stateless one-shot** | always available | `aie tick <name>` |
| **Heartbeat loop** | off | `heartbeat.enabled: true` |
| **User-activity modulation** | off | `heartbeat.modulation.enabled: true` |
| **Daily budget cap** | $5 USD | `heartbeat.budget.daily_usd_cap` |
| **Long-term memory (sqlite-vec)** | on | `memory.enabled: false` |
| **Valence tagging (hit/miss/walkback)** | on | `memory.parameterization.*` |
| **Memory decay** | 30-day half-life | `memory.parameterization.decay.*` |
| **Sleep consolidation** | on | `memory.consolidation.enabled` |
| **Transport (stdout/Discord/webhook)** | stdout | `transport.kind` |
| **Safety tiers (T0–T4)** | T1 draft-only | `safety.tier` |

Every toggle has a sensible default. Start with `aie tick example`, add features as you need them.

---

## The agent.yaml — one config, every knob

See [`agent.yaml.example`](agent.yaml.example) for the fully-commented version. Short tour:

```yaml
name: example
workspace: agents/example

model:
  provider: openai          # openai | anthropic | deepseek | groq | cerebras | openrouter | local | custom
  model_id: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
  # base_url: null          # override if you want a non-default endpoint

heartbeat:
  enabled: false            # off = stateless, invoke-only
  cadence:
    min_seconds: 60
    max_seconds: 1800
  modulation:
    enabled: true           # tick faster when user is active
  budget:
    daily_usd_cap: 5.00

memory:
  enabled: true
  storage:
    backend: sqlite_vec
    embedding:
      model: BAAI/bge-small-en-v1.5
  parameterization:
    valence_labels: [hit, miss, walkback, unmarked]
    retrieval:
      boost_hits: 1.5       # solutions that worked surface higher
      suppress_misses: 0.3  # failed approaches surface as warnings

transport:
  kind: stdout              # stdout | discord | webhook
```

---

## Providers — any OpenAI-compatible endpoint

The OpenAI Chat Completions API has become the universal interface. This framework leans on it. Set `provider`, `model_id`, and `api_key_env` in `agent.yaml`; paste the matching key in `.env`. Done.

| Provider | `provider` | `base_url` | Key env |
|---|---|---|---|
| OpenAI | `openai` | (default) | `OPENAI_API_KEY` |
| Anthropic (Claude) | `anthropic` | `https://api.anthropic.com/v1/` | `ANTHROPIC_API_KEY` |
| DeepSeek | `deepseek` | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| Groq | `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| Cerebras | `cerebras` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` |
| OpenRouter | `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| Ollama (local) | `local` | `http://localhost:11434/v1` | (any string) |
| llama.cpp / LM Studio | `local` | `http://localhost:1234/v1` | (any string) |
| Custom | `custom` | (your URL) | (your env var) |

See [`docs/PROVIDERS.md`](docs/PROVIDERS.md) for full per-provider notes.

---

## The valence-tagged memory model

Long-term memory is a `sqlite` database with `sqlite-vec` for embeddings. Every memory chunk carries a `valence` label:

| Label | Meaning | Retrieval effect |
|---|---|---|
| `hit` | Approach worked. Validated. | Boosted in retrieval (default ×1.5) |
| `miss` | Approach failed. Don't repeat. | Suppressed but surfaced as a warning (×0.3) |
| `walkback` | Earlier claim retracted; calibration moment. | Weighted high (×1.2) — learning gold |
| `unmarked` | Neutral / not yet labeled. | Default weight |

Tags are applied three ways:

1. **Heuristic auto-tagger** — regex on common phrases ("turned out wrong", "this worked", etc.)
2. **Explicit tag** — `aie tag <name> <chunk_id> hit` from the CLI
3. **Solution attempts** — a separate `solution_attempt` table for explicit task→approach→outcome rows

At recall time, the retriever does kNN over embeddings, then re-scores by valence × decay. The agent sees prior wins boosted and prior losses surfaced as cautionary context.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full schema.

---

## Heartbeat & user-activity modulation

When `heartbeat.enabled: true`, the agent runs a tick loop independent of any human invocation. Each tick:

1. Check `STOP` / `FREEZE` flags in workspace → bail if set.
2. Check daily budget → bail if exceeded.
3. Read state, build prompt (SOUL + MEMORY + recent observations + valence-weighted recall).
4. Call the model.
5. Parse a `[STATE intensity]` header from the response → update `state.json` atomically.
6. Post via the configured transport.
7. Sleep `next_tick_delay_seconds` (clamped by `min_seconds`/`max_seconds`).

With `modulation.enabled: true`, the delay is shortened when the user is active (recent workspace edits, git commits) and lengthened when idle. The agent breathes with you.

---

## Safety

- **Tier system** (T0 read-only → T4 autonomous). Default T1 routes all external actions to `drafts/`.
- **`STOP` flag**: drop a file named `STOP` in the workspace → loop exits cleanly.
- **`FREEZE` flag**: drop `FREEZE` → loop continues but writes nothing.
- **Budget cap**: hard kill if today's spend exceeds `daily_usd_cap`.
- **Atomic state writes**: crash-safe `state.json` (write-temp + `os.replace`).
- **No telemetry**: this framework phones home to nobody.

---

## Install

```bash
# Recommended (memory is on by default, so install with extras):
pip install -e ".[memory]"       # + sqlite-vec + sentence-transformers
pip install -e ".[memory,dev]"   # + pytest

# uv equivalents:
uv sync --extra memory
uv sync                          # core + memory + dev (uses [all])

# Bare-metal core only — only useful if you've also flipped
# `memory.enabled: false` in every agent.yaml.
pip install -e .
```

Python 3.11+. Cross-platform (macOS, Linux, Windows). Memory backend uses `sqlite-vec` and `sentence-transformers` (CPU works fine; Apple Silicon gets MPS acceleration automatically).

---

## Layout

```
ai-employee/
├── agent.yaml.example          # main config — copy + customize
├── .env.example                # API keys go here
├── agents/                     # one directory per agent
│   └── example/
│       ├── agent.yaml          # this agent's config
│       ├── SOUL.md             # identity, voice, values
│       ├── MEMORY.md           # human-readable persistent notes
│       ├── state.json          # auto-managed by heartbeat
│       └── memory.db           # auto-created if memory.enabled
├── src/ai_employee/
│   ├── cli.py                  # `aie` entry point
│   ├── config.py               # agent.yaml loader
│   ├── prompt.py               # prompt assembly
│   ├── runner/                 # model abstraction (OpenAI-compatible)
│   ├── memory/                 # sqlite-vec, valence, retrieval, decay
│   ├── heartbeat/              # tick loop, modulation, budget, state
│   └── transport/              # stdout, discord, webhook
└── tests/                      # pytest
```

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Status

v0.1 — minimum viable, designed to be forked. Issues and PRs welcome.
