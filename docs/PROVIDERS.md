# Providers

Every provider listed here speaks the OpenAI Chat Completions API. Set `provider`, `model_id`, and `api_key_env` in `agent.yaml`; paste the matching key in `.env`.

> Pricing changes constantly. Verify rates against the provider's current rate card and update `heartbeat.budget.pricing.*` in your `agent.yaml` accordingly.

## OpenAI

```yaml
model:
  provider: openai
  model_id: gpt-4o-mini       # or gpt-4o, o1-mini, etc.
  api_key_env: OPENAI_API_KEY
```

## Anthropic (Claude)

Anthropic offers an OpenAI-compatible endpoint at `/v1/`.

```yaml
model:
  provider: anthropic
  model_id: claude-sonnet-4-6   # or claude-opus-4-7, claude-haiku-4-5
  api_key_env: ANTHROPIC_API_KEY
```

## DeepSeek

```yaml
model:
  provider: deepseek
  model_id: deepseek-chat
  api_key_env: DEEPSEEK_API_KEY
```

## Groq

Fast inference for open-weight models.

```yaml
model:
  provider: groq
  model_id: llama-3.3-70b-versatile
  api_key_env: GROQ_API_KEY
```

## Cerebras

```yaml
model:
  provider: cerebras
  model_id: llama-3.3-70b
  api_key_env: CEREBRAS_API_KEY
```

## OpenRouter

A multi-provider proxy. Use `<vendor>/<model>` syntax.

```yaml
model:
  provider: openrouter
  model_id: anthropic/claude-sonnet-4
  api_key_env: OPENROUTER_API_KEY
```

## Ollama (local)

Run any open model on your own machine. Default port 11434.

```yaml
model:
  provider: local
  model_id: llama3.3
  api_key_env: LOCAL_API_KEY   # any non-empty string; Ollama ignores it
  # base_url: http://localhost:11434/v1   # already the default for `local`
```

## LM Studio / llama.cpp (local)

These also expose an OpenAI-compatible server.

```yaml
model:
  provider: custom
  model_id: your-model-name
  base_url: http://localhost:1234/v1
  api_key_env: LOCAL_API_KEY
```

## Anything else

If your provider speaks Chat Completions, use `provider: custom` and set `base_url` explicitly.

```yaml
model:
  provider: custom
  model_id: some-model
  base_url: https://api.example.com/v1
  api_key_env: EXAMPLE_API_KEY
```

## Adding a new "named" provider

Edit `PROVIDER_DEFAULT_BASE_URL` in `src/ai_employee/config.py`. One line per provider. The OpenAI SDK does the rest.

## Pricing reference (volatile — verify before relying)

| Model | Input $/1M | Output $/1M |
|---|---:|---:|
| OpenAI gpt-4o-mini | 0.15 | 0.60 |
| OpenAI gpt-4o | 2.50 | 10.00 |
| Anthropic Claude Sonnet 4.6 | 3.00 | 15.00 |
| Anthropic Claude Haiku 4.5 | 0.80 | 4.00 |
| DeepSeek deepseek-chat | 0.27 | 1.10 |
| Groq llama-3.3-70b | 0.59 | 0.79 |

These figures are approximate as of early 2026. **Always check the provider's current rate card** before setting your budget config.
