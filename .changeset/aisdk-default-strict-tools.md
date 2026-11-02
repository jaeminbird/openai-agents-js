---
'@openai/agents-extensions': patch
---

fix: only forward the SDK's default `strict: true` tool flag to OpenAI AI SDK providers, so Anthropic (strict grammar limits) and DeepSeek (beta-only strict tools) agents run again; add an `aisdk(model, { strictTools })` option to opt in or out
