# LLM Generation Service

This project can use an OpenAI-compatible LLM service for synthetic data generation and verification.

## Endpoint

- Base URL: `${LLM_BASE_URL}`
- API key environment variable: `OPENAI_API_KEY`
- Recommended client environment:

```bash
export OPENAI_BASE_URL="${LLM_BASE_URL}"
export OPENAI_API_KEY="<virtual-key>"
```

Do not commit the virtual key into repository files. Keep it in the shell environment or a local ignored secret file.

## Available Models

- `gpt-5.5`
- `glm-5.1`
- `qwen3.7-max`
- `deepseek-v4-pro`
- `doubao-seed-2.0-pro`
- `doubao-seed-2.0-lite`
- `gemini-3-flash-preview`
- `gemini-3.5-flash`
- `text-embedding-v4`
- `qwen3-rerank`
- `google-rerank`

## Smoke Test

```bash
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.5-flash",
    "messages": [
      {"role": "user", "content": "请用一句中文回答：接口连通了吗？"}
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

## Current Connectivity Check

- Checked at `2026-05-30 21:53 +0800` from `/mnt/disk/gaojun/research/progressive-ee`.
- Incorrect old endpoint `an internal endpoint` timed out from this environment.
- Correct endpoint `${LLM_BASE_URL}` is reachable:
  - `/v1/models` returned the expected model list.
  - `gemini-3.5-flash` chat completion returned `{"status":"ok"}` with `max_tokens=512`.
- Note: small `max_tokens` values such as `16` or `64` can be consumed by Gemini reasoning tokens and return `content: null`. Use a larger generation budget for smoke tests and JSON generation.

## Intended Project Use

Use the service for seed-to-balanced dataset reconstruction:

1. Planner call: convert a seed example and target quota into a structured generation plan.
2. Generator call: produce a new passage, gold event list, and optional reasoning trace.
3. Verifier call: check that triggers, event types, arguments, and role assignments are supported by the passage.
4. Rule filter: reject invalid schemas, missing trigger/argument surface strings, malformed JSON, and unsupported roles.
5. Model-disagreement filter: keep high-confidence examples that current Direct/E32 models find difficult.

The service should not be used as an unstructured paraphraser only. The goal is to rebuild a controlled event-extraction training distribution.
