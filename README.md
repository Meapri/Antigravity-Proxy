# Antigravity Proxy

OpenAI-compatible FastAPI proxy for Antigravity / Cloud Code models.

This project lets OpenAI-compatible clients talk to Antigravity-backed models
through familiar endpoints such as `/v1/models`, `/v1/chat/completions`, and
`/v1/images/generations`. It also includes a SearXNG-compatible `/search`
endpoint backed by Google Search grounding.

> This is an unofficial compatibility proxy. Use it on localhost, a private VPN,
> or another trusted network. If you expose it outside a trusted network, set
> `ANTIGRAVITY_PROXY_API_KEY` and put it behind TLS.

## What It Supports

- OpenAI-compatible model listing: `GET /v1/models`
- OpenAI-compatible chat completions: `POST /v1/chat/completions`
- OpenAI-compatible Responses API shim: `POST /v1/responses`
- Streaming chat responses, including tool-call deltas
- OpenAI-style function calling / tool calls
- Vision input through OpenAI `image_url` content parts
- OpenAI-compatible image generation: `POST /v1/images/generations`
- SearXNG-compatible grounded search: `GET /search?q=...&format=json`
- Admin model refresh without restart: `POST /admin/models/refresh`
- Optional API key protection
- OpenAI-style error responses

## Repository Safety

This repository intentionally does **not** include:

- OAuth client secrets
- access tokens
- refresh tokens
- `.env` files
- local caches
- logs
- databases
- runtime state

Keep those files outside git. `.gitignore` already blocks common secret and
runtime file names.

## Requirements

- Python 3.11+
- Valid local Antigravity OAuth credentials
- Network access to the Antigravity / Cloud Code endpoints

The proxy expects two local credential files:

- `ANTIGRAVITY_AUTH_FILE`: JSON file with `access`, `refresh`, `expires`, and optionally `project_id`
- `ANTIGRAVITY_CLIENT_FILE`: JSON file with OAuth `client_id` and `client_secret`

The exact way you obtain these files depends on your local Antigravity setup.
This project does not ship credentials and does not bypass OAuth.

## Quick Start

```bash
git clone https://github.com/Meapri/Antigravity-Proxy.git
cd Antigravity-Proxy

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
python antigravity_proxy.py
```

On Windows PowerShell:

```powershell
git clone https://github.com/Meapri/Antigravity-Proxy.git
cd Antigravity-Proxy

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
python antigravity_proxy.py
```

By default, the server listens on:

```text
http://127.0.0.1:8765
http://0.0.0.0:8765
```

## Configuration

Copy `.env.example` to `.env` and edit paths for your machine.

### Required

```bash
ANTIGRAVITY_AUTH_FILE=~/.hermes/auth/google_antigravity.json
ANTIGRAVITY_CLIENT_FILE=~/.hermes/auth/google_antigravity_client.json
```

### Common Optional Settings

```bash
ANTIGRAVITY_PROXY_MODEL=gemini-3.5-flash-high
ANTIGRAVITY_PROXY_IMAGE_MODEL=gemini-3.1-flash-image
ANTIGRAVITY_RESPONSES_DB=data/responses.sqlite3
ANTIGRAVITY_PROJECT_ID=
```

### Optional API Key Protection

If this variable is set, every route except `/health` requires either
`Authorization: Bearer <key>`, `X-API-Key: <key>`,
`X-Goog-API-Key: <key>`, or a Gemini-style `?key=<key>` query parameter.
Gemini Live WebSocket endpoints also accept `?key=<key>` and
`X-Goog-API-Key`.
Gemini REST routes return Gemini-style `UNAUTHENTICATED` errors on failed auth,
while OpenAI/admin routes keep OpenAI-compatible authentication errors.

```bash
ANTIGRAVITY_PROXY_API_KEY=change-this-long-random-value
```

Example:

```bash
curl http://127.0.0.1:8765/v1/models \
  -H "Authorization: Bearer change-this-long-random-value"
```

### Internal Model Visibility

Internal `tab_*` / `chat_*` models are hidden by default.

```bash
ANTIGRAVITY_PROXY_INCLUDE_INTERNAL_MODELS=1
```

### Grounded Search Tuning

The `/search` endpoint uses a lightweight grounded model by default.

```bash
ANTIGRAVITY_GROUNDING_MODEL=gemini-3.1-flash-lite
ANTIGRAVITY_GROUNDING_THINKING=-1
ANTIGRAVITY_GROUNDING_MAXTOK=0
```

## Smoke Tests

Health:

```bash
curl http://127.0.0.1:8765/health
```

Models:

```bash
curl http://127.0.0.1:8765/v1/models
```

Chat:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemini 3.5 Flash (High)",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'
```

Responses:

```bash
curl http://127.0.0.1:8765/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemini 3.5 Flash (High)",
    "input": "Say hello in one short sentence."
  }'
```

Responses streaming:

```bash
curl http://127.0.0.1:8765/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemini 3.5 Flash (High)",
    "input": "Stream one sentence.",
    "stream": true
  }'
```

Responses with Gemini-grounded web search and JSON schema output:

```bash
curl http://127.0.0.1:8765/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemini 3.5 Flash (High)",
    "input": "Find the latest supported Gemini API output schema controls.",
    "tools": [{"type": "web_search_preview"}],
    "text": {
      "format": {
        "type": "json_schema",
        "schema": {
          "type": "object",
          "properties": {
            "summary": {"type": "string"}
          },
          "required": ["summary"]
        }
      }
    },
    "reasoning": {"effort": "low"}
  }'
```

Retrieve stored response:

```bash
curl http://127.0.0.1:8765/v1/responses/resp_your_response_id
```

List response input items:

```bash
curl http://127.0.0.1:8765/v1/responses/resp_your_response_id/input_items
```

Count response input tokens:

```bash
curl http://127.0.0.1:8765/v1/responses/input_tokens \
  -H "Content-Type: application/json" \
  -d '{"input": "Count these tokens approximately."}'
```

Tool call:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Gemini 3.5 Flash (High)",
    "messages": [{"role": "user", "content": "What is the weather in Seoul?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "tool_choice": {"type": "function", "function": {"name": "get_current_weather"}}
  }'
```

Grounded search:

```bash
curl "http://127.0.0.1:8765/search?q=NVIDIA%20latest%20GPU&format=json"
```

Image generation:

```bash
curl http://127.0.0.1:8765/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.1-flash-image",
    "prompt": "A small robot reading a book at a wooden desk",
    "size": "1024x1024"
  }'
```

## OpenAI SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="not-used",  # Use your proxy API key here if ANTIGRAVITY_PROXY_API_KEY is set.
)

response = client.chat.completions.create(
    model="Gemini 3.5 Flash (High)",
    messages=[{"role": "user", "content": "Explain this proxy in one sentence."}],
)

print(response.choices[0].message.content)
```

Responses API example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="not-used",
)

response = client.responses.create(
    model="Gemini 3.5 Flash (High)",
    input="Explain this proxy in one sentence.",
)

print(response.output_text)
```

## Gemini API Compatibility

The proxy also exposes a Gemini REST-compatible surface for clients that can
use a custom Gemini base URL, such as a Hermes Gemini provider:

```text
http://127.0.0.1:8765/v1beta
```

Remote Tailscale example:

```text
http://your-host.ts.net:8765/v1beta
```

Gemini stable-version aliases are also accepted for Gemini-specific routes,
such as `/v1/models/{model}:generateContent`, `/v1/files:register`,
`/v1/cachedContents`, `/v1/batches`, and `/v1/live`. OpenAI-compatible routes
that already live under `/v1` keep their OpenAI behavior, so `/v1/models`,
`/v1/chat/completions`, `/v1/responses`, and `/v1/images/generations` are not
rewritten.

Common SDK spelling variants are accepted for query parameters: `page_size`,
`page_token`, `update_mask`, `upload_type`, `display_name`, and
`return_partial_success` are normalized to the Gemini REST camelCase forms.
`generateContent?alt=sse` and
`generateContent?stream=true` are treated as streaming Gemini SSE responses.
Streaming fallback errors use the same Gemini `error` payload and
`google.rpc.ErrorInfo` details as non-streaming Gemini errors.
`interactions.create` accepts the same SDK-style `config` object used by
`generateContent`, including `systemInstruction`, `generationConfig` scalar
coercion, `safetySettings`, `functionDeclarations`, `toolConfig`, and
response-format aliases.

Implemented Gemini-compatible routes:

- `GET /v1/models/{model}`
- `GET /v1/models/{model}/operations`
- `GET /v1/models/{model}/operations/{operation}`
- `POST /v1/models/{model}/operations/{operation}:wait`
- `POST /v1/models/{model}/operations/{operation}:cancel`
- `DELETE /v1/models/{model}/operations/{operation}`
- `POST /v1/models/{model}:generateContent`
- `POST /v1/models/{model}:streamGenerateContent`
- `POST /v1/models/{model}:countTokens`
- `POST /v1/models/{model}:computeTokens`
- `POST /v1/models/{model}:countTextTokens`
- `POST /v1/models/{model}:countMessageTokens`
- `POST /v1/models/{model}:embedContent`
- `POST /v1/models/{model}:batchEmbedContents`
- `POST /v1/models/{model}:embedText`
- `POST /v1/models/{model}:batchEmbedText`
- `POST /v1/models/{model}:asyncBatchEmbedContent`
- `POST /v1/models/{model}:batchGenerateContent`
- `POST /v1/models/{model}:generateText`
- `POST /v1/models/{model}:generateMessage`
- `POST /v1/models/{model}:generateAnswer`
- `POST /v1/models/{model}:generateImages`
- `POST /v1/models/{model}:generateVideos`
- `POST /v1/models/{model}:predict`
- `POST /v1/models/{model}:predictLongRunning`
- `GET /v1beta/models`
- `GET /v1beta/models/{model}`
- `GET /v1beta/models/{model}/operations`
- `GET /v1beta/models/{model}/operations/{operation}`
- `POST /v1beta/models/{model}/operations/{operation}:wait`
- `POST /v1beta/models/{model}/operations/{operation}:cancel`
- `DELETE /v1beta/models/{model}/operations/{operation}`
- `POST /v1beta/models/{model}:generateContent`
- `POST /v1beta/models/{model}:streamGenerateContent`
- `POST /v1beta/models/{model}:countTokens`
- `POST /v1beta/models/{model}:computeTokens`
- `POST /v1beta/models/{model}:countTextTokens`
- `POST /v1beta/models/{model}:countMessageTokens`
- `POST /v1beta/models/{model}:embedContent`
- `POST /v1beta/models/{model}:batchEmbedContents`
- `POST /v1beta/models/{model}:embedText`
- `POST /v1beta/models/{model}:batchEmbedText`
- `POST /v1beta/models/{model}:asyncBatchEmbedContent`
- `POST /v1beta/models/{model}:batchGenerateContent`
- `POST /v1beta/models/{model}:generateText`
- `POST /v1beta/models/{model}:generateMessage`
- `POST /v1beta/models/{model}:generateAnswer`
- `POST /v1beta/models/{model}:generateImages`
- `POST /v1beta/models/{model}:generateVideos`
- `POST /v1beta/models/{model}:predict`
- `POST /v1beta/models/{model}:predictLongRunning`
- `POST /v1/interactions`
- `GET /v1/interactions/{interaction}`
- `POST /v1/interactions/{interaction}/cancel`
- `POST /v1/interactions/{interaction}:cancel`
- `DELETE /v1/interactions/{interaction}`
- `POST /v1beta/interactions`
- `GET /v1beta/interactions/{interaction}`
- `POST /v1beta/interactions/{interaction}/cancel`
- `POST /v1beta/interactions/{interaction}:cancel`
- `DELETE /v1beta/interactions/{interaction}`
- `WS /ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`
- `WS /v1beta/live`
- `POST /v1/batches`
- `GET /v1/batches`
- `GET /v1/batches/{batch}`
- `POST /v1/batches/{batch}:cancel`
- `PATCH /v1/batches/{batch}:updateGenerateContentBatch`
- `PATCH /v1/batches/{batch}:updateEmbedContentBatch`
- `DELETE /v1/batches/{batch}`
- `POST /v1beta/batches`
- `GET /v1beta/batches`
- `GET /v1beta/batches/{batch}`
- `POST /v1beta/batches/{batch}:cancel`
- `PATCH /v1beta/batches/{batch}:updateGenerateContentBatch`
- `PATCH /v1beta/batches/{batch}:updateEmbedContentBatch`
- `DELETE /v1beta/batches/{batch}`
- `POST /v1/webhooks`
- `GET /v1/webhooks`
- `GET /v1/webhooks/{webhook}`
- `PATCH /v1/webhooks/{webhook}`
- `DELETE /v1/webhooks/{webhook}`
- `POST /v1/webhooks/{webhook}:ping`
- `POST /v1/webhooks/{webhook}:rotateSigningSecret`
- `POST /v1beta/webhooks`
- `GET /v1beta/webhooks`
- `GET /v1beta/webhooks/{webhook}`
- `PATCH /v1beta/webhooks/{webhook}`
- `DELETE /v1beta/webhooks/{webhook}`
- `POST /v1beta/webhooks/{webhook}:ping`
- `POST /v1beta/webhooks/{webhook}:rotateSigningSecret`
- `POST /v1/files:register`
- `POST /upload/v1/files`
- `POST /v1/files`
- `GET /v1/files`
- `GET /v1/files/{file}`
- `GET /v1/files/{file}:download`
- `DELETE /v1/files/{file}`
- `POST /v1beta/files:register`
- `POST /upload/v1beta/files`
- `POST /v1beta/files`
- `GET /v1beta/files`
- `GET /v1beta/files/{file}`
- `GET /v1beta/files/{file}:download`
- `DELETE /v1beta/files/{file}`
- `GET /v1/generatedFiles`
- `GET /v1/generatedFiles/{generated_file}`
- `GET /v1/generatedFiles/{generated_file}:download`
- `DELETE /v1/generatedFiles/{generated_file}`
- `GET /v1/generatedFiles/operations`
- `GET /v1/generatedFiles/operations/{operation}`
- `POST /v1/generatedFiles/operations/{operation}:wait`
- `POST /v1/generatedFiles/operations/{operation}:cancel`
- `DELETE /v1/generatedFiles/operations/{operation}`
- `GET /v1beta/generatedFiles`
- `GET /v1beta/generatedFiles/{generated_file}`
- `GET /v1beta/generatedFiles/{generated_file}:download`
- `DELETE /v1beta/generatedFiles/{generated_file}`
- `GET /v1beta/generatedFiles/operations`
- `GET /v1beta/generatedFiles/operations/{operation}`
- `POST /v1beta/generatedFiles/operations/{operation}:wait`
- `POST /v1beta/generatedFiles/operations/{operation}:cancel`
- `DELETE /v1beta/generatedFiles/operations/{operation}`
- `POST /v1/cachedContents`
- `GET /v1/cachedContents`
- `GET /v1/cachedContents/{cached_content}`
- `PATCH /v1/cachedContents/{cached_content}`
- `DELETE /v1/cachedContents/{cached_content}`
- `POST /v1beta/cachedContents`
- `GET /v1beta/cachedContents`
- `GET /v1beta/cachedContents/{cached_content}`
- `PATCH /v1beta/cachedContents/{cached_content}`
- `DELETE /v1beta/cachedContents/{cached_content}`
- `POST /v1/corpora`
- `GET /v1/corpora`
- `GET /v1/corpora/{corpus}`
- `PATCH /v1/corpora/{corpus}`
- `POST /v1/corpora/{corpus}:query`
- `DELETE /v1/corpora/{corpus}`
- `POST /v1/corpora/{corpus}/documents`
- `GET /v1/corpora/{corpus}/documents`
- `GET /v1/corpora/{corpus}/documents/{document}`
- `PATCH /v1/corpora/{corpus}/documents/{document}`
- `POST /v1/corpora/{corpus}/documents/{document}:query`
- `DELETE /v1/corpora/{corpus}/documents/{document}`
- `POST /v1/corpora/{corpus}/documents/{document}/chunks`
- `GET /v1/corpora/{corpus}/documents/{document}/chunks`
- `GET /v1/corpora/{corpus}/documents/{document}/chunks/{chunk}`
- `PATCH /v1/corpora/{corpus}/documents/{document}/chunks/{chunk}`
- `DELETE /v1/corpora/{corpus}/documents/{document}/chunks/{chunk}`
- `POST /v1/corpora/{corpus}/documents/{document}/chunks:batchCreate`
- `POST /v1/corpora/{corpus}/documents/{document}/chunks:batchUpdate`
- `POST /v1/corpora/{corpus}/documents/{document}/chunks:batchDelete`
- `GET /v1/corpora/{corpus}/permissions`
- `POST /v1/corpora/{corpus}/permissions`
- `GET /v1/corpora/{corpus}/permissions/{permission}`
- `PATCH /v1/corpora/{corpus}/permissions/{permission}`
- `DELETE /v1/corpora/{corpus}/permissions/{permission}`
- `POST /v1beta/corpora`
- `GET /v1beta/corpora`
- `GET /v1beta/corpora/{corpus}`
- `PATCH /v1beta/corpora/{corpus}`
- `POST /v1beta/corpora/{corpus}:query`
- `DELETE /v1beta/corpora/{corpus}`
- `POST /v1beta/corpora/{corpus}/documents`
- `GET /v1beta/corpora/{corpus}/documents`
- `GET /v1beta/corpora/{corpus}/documents/{document}`
- `PATCH /v1beta/corpora/{corpus}/documents/{document}`
- `POST /v1beta/corpora/{corpus}/documents/{document}:query`
- `DELETE /v1beta/corpora/{corpus}/documents/{document}`
- `POST /v1beta/corpora/{corpus}/documents/{document}/chunks`
- `GET /v1beta/corpora/{corpus}/documents/{document}/chunks`
- `GET /v1beta/corpora/{corpus}/documents/{document}/chunks/{chunk}`
- `PATCH /v1beta/corpora/{corpus}/documents/{document}/chunks/{chunk}`
- `DELETE /v1beta/corpora/{corpus}/documents/{document}/chunks/{chunk}`
- `POST /v1beta/corpora/{corpus}/documents/{document}/chunks:batchCreate`
- `POST /v1beta/corpora/{corpus}/documents/{document}/chunks:batchUpdate`
- `POST /v1beta/corpora/{corpus}/documents/{document}/chunks:batchDelete`
- `GET /v1beta/corpora/{corpus}/permissions`
- `POST /v1beta/corpora/{corpus}/permissions`
- `GET /v1beta/corpora/{corpus}/permissions/{permission}`
- `PATCH /v1beta/corpora/{corpus}/permissions/{permission}`
- `DELETE /v1beta/corpora/{corpus}/permissions/{permission}`
- `GET /v1/operations`
- `GET /v1/operations/{operation}`
- `POST /v1/operations/{operation}:wait`
- `POST /v1/operations/{operation}:cancel`
- `DELETE /v1/operations/{operation}`
- `GET /v1beta/operations`
- `GET /v1beta/operations/{operation}`
- `POST /v1beta/operations/{operation}:wait`
- `POST /v1beta/operations/{operation}:cancel`
- `DELETE /v1beta/operations/{operation}`
- `POST /v1/fileSearchStores`
- `GET /v1/fileSearchStores`
- `GET /v1/fileSearchStores/{store}`
- `DELETE /v1/fileSearchStores/{store}`
- `POST /v1/fileSearchStores/{store}:importFile`
- `POST /upload/v1/fileSearchStores/{store}:uploadToFileSearchStore`
- `POST /v1/fileSearchStores/{store}:uploadToFileSearchStore`
- `GET /v1/fileSearchStores/{store}/documents`
- `GET /v1/fileSearchStores/{store}/documents/{document}`
- `GET /v1/fileSearchStores/{store}/media/{document}`
- `GET /v1/fileSearchStores/{store}/operations`
- `GET /v1/fileSearchStores/{store}/operations/{operation}`
- `GET /v1/fileSearchStores/{store}/upload/operations/{operation}`
- `POST /v1/fileSearchStores/{store}/operations/{operation}:wait`
- `POST /v1/fileSearchStores/{store}/upload/operations/{operation}:wait`
- `POST /v1/fileSearchStores/{store}/operations/{operation}:cancel`
- `POST /v1/fileSearchStores/{store}/upload/operations/{operation}:cancel`
- `DELETE /v1/fileSearchStores/{store}/operations/{operation}`
- `DELETE /v1/fileSearchStores/{store}/upload/operations/{operation}`
- `DELETE /v1/fileSearchStores/{store}/documents/{document}`
- `POST /v1beta/fileSearchStores`
- `GET /v1beta/fileSearchStores`
- `GET /v1beta/fileSearchStores/{store}`
- `DELETE /v1beta/fileSearchStores/{store}`
- `POST /v1beta/fileSearchStores/{store}:importFile`
- `POST /upload/v1beta/fileSearchStores/{store}:uploadToFileSearchStore`
- `GET /v1beta/fileSearchStores/{store}/documents`
- `GET /v1beta/fileSearchStores/{store}/documents/{document}`
- `GET /v1beta/fileSearchStores/{store}/media/{document}`
- `GET /v1beta/fileSearchStores/{store}/operations`
- `GET /v1beta/fileSearchStores/{store}/operations/{operation}`
- `GET /v1beta/fileSearchStores/{store}/upload/operations/{operation}`
- `POST /v1beta/fileSearchStores/{store}/operations/{operation}:wait`
- `POST /v1beta/fileSearchStores/{store}/upload/operations/{operation}:wait`
- `POST /v1beta/fileSearchStores/{store}/operations/{operation}:cancel`
- `POST /v1beta/fileSearchStores/{store}/upload/operations/{operation}:cancel`
- `DELETE /v1beta/fileSearchStores/{store}/operations/{operation}`
- `DELETE /v1beta/fileSearchStores/{store}/upload/operations/{operation}`
- `DELETE /v1beta/fileSearchStores/{store}/documents/{document}`
- `POST /v1/tunedModels`
- `GET /v1/tunedModels`
- `GET /v1/tunedModels/{tuned_model}`
- `GET /v1/tunedModels/{tuned_model}/operations`
- `GET /v1/tunedModels/{tuned_model}/operations/{operation}`
- `POST /v1/tunedModels/{tuned_model}/operations/{operation}:wait`
- `POST /v1/tunedModels/{tuned_model}/operations/{operation}:cancel`
- `DELETE /v1/tunedModels/{tuned_model}/operations/{operation}`
- `PATCH /v1/tunedModels/{tuned_model}`
- `DELETE /v1/tunedModels/{tuned_model}`
- `POST /v1/tunedModels/{tuned_model}:generateContent`
- `POST /v1/tunedModels/{tuned_model}:streamGenerateContent`
- `POST /v1/tunedModels/{tuned_model}:countTokens`
- `POST /v1/tunedModels/{tuned_model}:computeTokens`
- `POST /v1/tunedModels/{tuned_model}:embedContent`
- `POST /v1/tunedModels/{tuned_model}:batchEmbedContents`
- `GET /v1/tunedModels/{tuned_model}/permissions`
- `POST /v1/tunedModels/{tuned_model}/permissions`
- `GET /v1/tunedModels/{tuned_model}/permissions/{permission}`
- `PATCH /v1/tunedModels/{tuned_model}/permissions/{permission}`
- `POST /v1/tunedModels/{tuned_model}/permissions/{permission}:transferOwnership`
- `DELETE /v1/tunedModels/{tuned_model}/permissions/{permission}`
- `POST /v1beta/tunedModels`
- `GET /v1beta/tunedModels`
- `GET /v1beta/tunedModels/{tuned_model}`
- `GET /v1beta/tunedModels/{tuned_model}/operations`
- `GET /v1beta/tunedModels/{tuned_model}/operations/{operation}`
- `POST /v1beta/tunedModels/{tuned_model}/operations/{operation}:wait`
- `POST /v1beta/tunedModels/{tuned_model}/operations/{operation}:cancel`
- `DELETE /v1beta/tunedModels/{tuned_model}/operations/{operation}`
- `PATCH /v1beta/tunedModels/{tuned_model}`
- `DELETE /v1beta/tunedModels/{tuned_model}`
- `POST /v1beta/tunedModels/{tuned_model}:generateContent`
- `POST /v1beta/tunedModels/{tuned_model}:streamGenerateContent`
- `POST /v1beta/tunedModels/{tuned_model}:countTokens`
- `POST /v1beta/tunedModels/{tuned_model}:computeTokens`
- `POST /v1beta/tunedModels/{tuned_model}:embedContent`
- `POST /v1beta/tunedModels/{tuned_model}:batchEmbedContents`
- `GET /v1beta/tunedModels/{tuned_model}/permissions`
- `POST /v1beta/tunedModels/{tuned_model}/permissions`
- `GET /v1beta/tunedModels/{tuned_model}/permissions/{permission}`
- `PATCH /v1beta/tunedModels/{tuned_model}/permissions/{permission}`
- `POST /v1beta/tunedModels/{tuned_model}/permissions/{permission}:transferOwnership`
- `DELETE /v1beta/tunedModels/{tuned_model}/permissions/{permission}`

Example:

```bash
curl http://127.0.0.1:8765/v1beta/models/gemini-3-flash-agent:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{
      "role": "user",
      "parts": [{"text": "Say hello in one short sentence."}]
    }],
    "generationConfig": {
      "temperature": 0.3,
      "maxOutputTokens": 128
    }
  }'
```

The Gemini compatibility layer forwards native Gemini request fields such as
`contents`, `systemInstruction`, `generationConfig`, `safetySettings`,
`tools`, and `toolConfig` to Antigravity. It accepts both common camelCase and
snake_case SDK spellings for fields such as `generation_config`,
`response_mime_type`, `inline_data`, `file_data`, `googleSearch`,
`googleSearchRetrieval`, `urlContext`, and `codeExecution`. `googleSearch` and
legacy `googleSearchRetrieval` are forwarded as `google_search`, and local
`file_search` tools are converted into retrieved context. `urlContext` and
`codeExecution` are recognized but return
`UNIMPLEMENTED` because the current Antigravity backend does not expose those
hosted tools. `toolConfig.functionCallingConfig.mode` and
`allowedFunctionNames` are normalized from common SDK spellings.
`tools` may be a single object or a list, and top-level
`functionDeclarations` / `function_declarations` are wrapped into Gemini tool
objects before forwarding. Function declaration schema aliases such as
`parametersJsonSchema` / `parameters_json_schema` and `responseJsonSchema` /
`response_json_schema` are normalized to Gemini `parameters` and `response`.
`safetySettings` may be supplied as a single object, and shortcut
`toolConfig` forms such as `{mode, allowedFunctionNames}` are expanded into
`functionCallingConfig`. Safety settings accept common short category and
threshold aliases such as `harassment`, `dangerous`, `only_high`,
`medium_and_above`, and `none`, plus `harm_block_method`.
`generateContent`, `streamGenerateContent`, `batchGenerateContent`, and
`countTokens` also normalize SDK content-union inputs such as string
`contents`, part dictionaries, part arrays, and string `systemInstruction`
values into Gemini REST `Content` objects before forwarding.
They also accept the common SDK-style top-level `config` object and merge it
into Gemini REST fields such as `generationConfig`, `systemInstruction`,
`toolConfig`, `safetySettings`, `tools`, `cachedContent`, and `labels`.
Generation config aliases include `responseSchema` / `response_schema`,
`responseJsonSchema` / `response_json_schema`, `thinkingConfig` /
`thinking_config`, `responseModalities` / `response_modalities`, and
`enableEnhancedCivicAnswers` / `enable_enhanced_civic_answers`.
`service_tier` / `serviceTier` and `store` are preserved as top-level
GenerateContent request fields.
Google provider wrappers such as `providerOptions.google` and
`provider_options.google` are merged through the same path, with direct request
and `config` fields taking precedence.
`processingOptions` / `processing_options` and aliases such as `start_offset`
and `end_offset` are accepted for current Gemini SDK compatibility; because
Antigravity's internal endpoint does not expose that field yet, the proxy
removes it before forwarding so requests do not fail with unknown-field errors.
Content parts preserve SDK-style aliases such as `function_call`,
`function_response`, `executable_code`, `code_execution_result`, and
`thought_signature` as canonical Gemini REST fields.
String `stopSequences` and `responseModalities` values are normalized to the
Gemini REST list form. Numeric and boolean generation config values such as
`maxOutputTokens`, `temperature`, `topK`, and `responseLogprobs` are coerced
from common string forms when possible.
SDK transport-only options such as `httpOptions`, `requestOptions`,
`apiVersion`, and `baseUrl` are ignored rather than forwarded as API payload.
`response_format` / `responseFormat` wrappers and direct
`generationConfig.responseSchema` / SDK `config.response_schema` values are
mapped into Gemini-compatible JSON output controls.
Common schema aliases such as `property_ordering`, `any_of`, `min_items`, and
`min_length` are normalized to Gemini/OpenAPI-style camelCase fields.
Generate responses are normalized with Gemini-style `modelVersion`,
`responseId`, candidate `index`, `finishReason`, model-role content parts, and
`usageMetadata` token counts when the upstream response omits or partially
spells them differently. Usage metadata preserves tool-use and thought token
counts, and computed `totalTokenCount` includes prompt, candidate, tool-use
prompt, and thought tokens. Candidate metadata aliases such as
`safety_ratings`, `grounding_metadata`, and `avg_logprobs`, plus top-level
aliases such as `prompt_feedback`, `model_version`, and `response_id`, are also
returned in Gemini REST camelCase form.
Gemini compatibility errors use the `google.rpc.Status`-style
`error.code` / `error.message` / `error.status` shape. `INVALID_ARGUMENT`
responses include `google.rpc.BadRequest` field-violation details, and
unsupported hosted features include `google.rpc.ErrorInfo` details so SDKs can
distinguish unsupported proxy features from malformed requests.
Gemini request validation failures, including invalid query parameters, return
`400 INVALID_ARGUMENT` rather than OpenAI-style validation errors.
Unmatched Gemini routes and uncaught Gemini `HTTPException`s also return this
Gemini error shape, while OpenAI-compatible routes keep OpenAI-style errors.
Quota, timeout, and server-side failures map to Gemini statuses such as
`RESOURCE_EXHAUSTED`, `DEADLINE_EXCEEDED`, `UNAVAILABLE`, or `INTERNAL`.

Files API example:

```bash
curl "http://127.0.0.1:8765/upload/v1beta/files?uploadType=media&displayName=note.txt" \
  -H "Content-Type: text/plain" \
  --data-binary @note.txt
```

The upload endpoint also supports Gemini resumable upload starts through
`uploadType=resumable` or `X-Goog-Upload-Protocol: resumable`, followed by
`X-Goog-Upload-Command: query`, `upload`, and `upload, finalize` with
`X-Goog-Upload-Offset`. Upload metadata accepts both official `file` objects
and SDK-style `config` wrappers such as `{"config": {"mimeType": "text/plain"}}`.
Uploaded local files include a Gemini-style `expirationTime` 48 hours after
creation; metadata-only registered external files preserve an explicit
`expirationTime` only when supplied.

Metadata-only File creation is also available through the official metadata
URI:

```bash
curl http://127.0.0.1:8765/v1beta/files \
  -H "Content-Type: application/json" \
  -d '{"file":{"displayName":"external.txt","mimeType":"text/plain","uri":"gs://bucket/external.txt"}}'
```

`files:register` accepts Gemini's current `uris` array form and returns
`files`; common `config` fields such as `mimeType`, `source`, and
`customMetadata` are applied to each URI, while an optional `files` array can
provide per-URI metadata such as `displayName` or `customMetadata`. The older
proxy `{"file": ...}` form remains available as a compatibility alias. File
`state` and `source` values are normalized to Gemini-style enum names such as
`ACTIVE`, `FAILED`, `UPLOADED`, and `REGISTERED` across create, list, and get
responses. Metadata-only registered files receive a deterministic base64
`sha256Hash` when one is not supplied.

Then pass the returned `file.uri` in `fileData.fileUri`:

```json
{
  "contents": [{
    "role": "user",
    "parts": [
      {"text": "Summarize this file."},
      {"fileData": {"mimeType": "text/plain", "fileUri": "files/file_..."}}
    ]
  }]
}
```

Uploaded local file references are automatically converted to Gemini
`inlineData` before being forwarded to Antigravity, which keeps Gemini SDK-style
file upload flows usable even though Antigravity's internal endpoint does not
expose a native public Files API. The proxy also accepts Gemini SDK-style File
resource objects directly inside `contents[].parts`, for example
`{"name": "files/...", "uri": "files/...", "mimeType": "text/plain"}`.
Bytes-style SDK parts such as `{"data": "...", "mimeType": "image/png"}` are
normalized to Gemini `inlineData`.

Cached content example:

```bash
curl http://127.0.0.1:8765/v1beta/cachedContents \
  -H "Content-Type: application/json" \
  -d '{
    "model": "models/gemini-3-flash-agent",
    "contents": [{
      "role": "user",
      "parts": [{"text": "Reusable context"}]
    }]
  }'
```

Then pass the returned `name` as `cachedContent` in `generateContent`. The proxy
merges local cached content into the outgoing Antigravity request because the
upstream internal endpoint does not expose public Gemini cache objects.
Create and patch calls also accept SDK wrapper bodies like
`{"cachedContent": {"ttl": "3600s"}}`; patch supports Gemini-style `ttl` or
`expireTime` updates with `update_mask=ttl` / `updateMask=expireTime`, or with
`updateMask` / `update_mask` inside the SDK wrapper body.
Cached content creation uses the same SDK-friendly normalization as
`generateContent`, including string `contents`, `config`, single
`safetySettings`, and shortcut `toolConfig` forms. When a `cachedContent`
wrapper is present, sibling fields such as `model`, `displayName`, `ttl`,
`expireTime`, and `config` are merged into the wrapped cache resource if the
wrapped object did not already provide them. Cache `usageMetadata` is computed
with the same local token-estimation path used by `countTokens`.

Embeddings and batch operations:

- `embedContent` and `batchEmbedContents` return deterministic local embedding
  vectors for Gemini SDK compatibility. They accept current SDK-style
  `config` / `embedContentConfig`, string or list `contents`, and the legacy
  top-level `outputDimensionality`, `taskType`, and `title` fields. They are
  normalized from either camelCase or snake_case, including string numeric
  `outputDimensionality`, common `taskType` spellings such as
  `retrieval document`, and boolean config fields such as `autoTruncate`,
  `documentOcr`, and `audioTrackExtraction`. They are stable and shaped like
  Gemini embeddings, but they are not semantic Google embedding model outputs
  because Antigravity does not expose a public embedding RPC. Single and batch
  embed requests accept SDK wrapper bodies such as `{"request": {...}}`,
  `{"embedContentRequest": {...}}`, and `providerOptions.google` /
  `provider_options.google` embedding config.
- `asyncBatchEmbedContent` stores the deterministic batch embedding result as
  an immediately completed local operation and batch resource.
- `batchGenerateContent` runs requests synchronously through Antigravity and
  stores immediately completed `operations/*` and `batches/*` results with
  Gemini `BATCH_STATE_*` status values and `stats` counters. Inline batch
  items may be plain request objects or SDK/REST wrapper items such as
  `{"request": {...}}` / `{"generateContentRequest": {...}}`.
- `batches.create` accepts inline `requests` plus `model` and returns a
  completed Gemini operation named `batches/*`; the local batch resource is
  preserved under `metadata.batchResource`. It also accepts common SDK wrapper bodies
  such as `{"batch": {...}}`, `{"generateContentBatch": {...}}`,
  `{"generate_content_batch": {...}}`, `{"embedContentBatch": {...}}`, and
  `{"embed_content_batch": {...}}`. Top-level `inputConfig` / `input_config`,
  `outputConfig` / `output_config`, and `priority` fields are preserved on the
  stored batch resource. Embed wrappers are completed with local deterministic
  embeddings, and embed request items may use
  `{"request": {...}}` / `{"embedContentRequest": {...}}`. It is intended for
  Gemini SDK/REST management compatibility; it does not run true asynchronous
  Batch Mode jobs.
- `batches.updateGenerateContentBatch` and `batches.updateEmbedContentBatch`
  accept wrapper bodies plus `updateMask` / `update_mask` for `displayName`
  and `priority`, and return the same completed `batches/*` operation view.
- `batches.list` supports `filter` terms for common operation and batch fields
  such as `done`, `displayName`, `state`, `model`, and
  `metadata.batchResource.*`; `returnPartialSuccess` /
  `return_partial_success` returns an empty `unreachable` list for local stores.
- Scoped operation routes under models, generated files, file search stores,
  and tuned models validate that the operation belongs to the requested parent.
  `:cancel` on scoped routes stores a Gemini-style `CANCELLED` operation state
  for pending operations before returning the empty cancel response.
- `predict` and `predictLongRunning` are mapped to Gemini `generateContent`
  requests and return prediction/operation-shaped compatibility responses.
  Vertex-style `instances` requests preserve the same SDK `config`,
  `providerOptions.google`, and `processingOptions` compatibility path used by
  `generateContent`. `parameters` may carry Gemini generation config fields or
  top-level generate fields such as `safetySettings` / `safety_settings` and
  `toolConfig` / `tool_config`; these are split and normalized before
  forwarding.
- Legacy `generateText`, `generateMessage`, `generateAnswer`, `embedText`,
  `batchEmbedText`, `countTextTokens`, and `countMessageTokens` are accepted and
  mapped onto the newer local `generateContent`, embedding, and token-count
  compatibility paths. Legacy text/message generation also accepts SDK
  `config`, `providerOptions.google`, `response_format`, and
  `processingOptions` wrappers before forwarding to the generate path.

File search stores:

- `fileSearchStores` and document management are implemented as a local
  compatibility layer under `data/gemini_file_search_stores`.
- `fileSearchStores.create`, `importFile`, and `uploadToFileSearchStore` accept
  current SDK-style `config` wrappers as well as direct REST fields.
- `fileSearchStores.create` also accepts `fileSearchStore` /
  `file_search_store` resource wrappers and normalizes aliases such as
  `embedding_model`, `chunking_config`, `white_space_config`,
  `max_tokens_per_chunk`, and `max_overlap_tokens`.
- File search store resources include Gemini-style document counters and
  `sizeBytes`; deleting a non-empty store requires `force=true`.
- `importFile` imports files previously uploaded through the local Files API and
  preserves document display names, custom metadata, and document
  `chunkingConfig` when supplied. It accepts `fileMetadata` /
  `file_metadata` wrappers in addition to direct fields.
- `uploadToFileSearchStore` accepts raw, multipart, or JSON direct uploads and
  stores documents locally while preserving custom metadata and
  `chunkingConfig`. JSON uploads may provide document metadata through
  `file`, `fileMetadata`, or `file_metadata` wrappers.
- `fileSearchStores.documents.list` accepts `pageSize` / `pageToken` and
  snake_case `page_size` / `page_token`, with the Gemini REST page size capped
  at 20.
- `tools.file_search` / `tools.fileSearch` performs local lexical retrieval
  against these stores and injects the best matching document snippets into the
  outgoing Gemini request context.

Corpora and semantic retriever:

- Legacy `corpora`, `documents`, `chunks`, and corpus `permissions` are
  implemented as a local compatibility layer under `data/gemini_corpora`.
- Corpus and document `:query` perform local lexical chunk matching. They return
  Gemini-shaped `relevantChunks`, but they are not semantic Google retriever
  scores because Antigravity does not expose that service.
- Corpus, document, and chunk patch calls honor `updateMask` / `update_mask`
  for supported mutable fields, including wrapper paths such as
  `corpus.displayName`, `document.customMetadata`, and `chunk.data`.
- Document and chunk create/update calls accept SDK-style `document` / `chunk`
  wrapper bodies, query/body `documentId` / `chunkId` aliases, and per-request
  `updateMask` values in `chunks:batchUpdate`.

Tuned models and permissions:

- `tunedModels` are implemented as local aliases over a base Antigravity model.
- Creating a tuned model stores metadata and returns an immediately completed
  operation; it does not run real model training.
- `tunedModels.create` and patch accept SDK-style `config` / `tunedModel`
  wrappers and preserve Gemini tuning metadata such as `tuningTask`,
  `hyperparameters`, `trainingData`, `validationData`, `readerProjectNumbers`,
  and `tunedModelSource`.
- `tunedModels.patch` honors `updateMask` / `update_mask` from either the query
  string or SDK wrapper body for mutable metadata fields such as `displayName`,
  `description`, `baseModel`, `tuningTask`, and `readerProjectNumbers`.
- `tunedModels/{id}:generateContent`, `:streamGenerateContent`,
  `:countTokens`, `:computeTokens`, `:embedContent`, and
  `:batchEmbedContents` forward to or reuse the configured `baseModel`
  compatibility path.
- `permissions` are stored locally for Gemini SDK compatibility; corpus and
  tuned-model permission create/patch accept `permission` wrappers plus
  snake_case aliases such as `email_address` and `grantee_type`, normalizing
  role/grantee enums to Gemini-style uppercase values. Permission patch honors
  `updateMask` / `update_mask` for `role`, `granteeType`, and `emailAddress`,
  and tuned-model permission routes also accept full permission resource names
  in the `{permission}` path segment.

Generated files:

- OpenAI-compatible image generation also stores each image as a local Gemini
  `generatedFiles/*` resource so Gemini-style clients can list, fetch, and
  download generated media.
- Gemini image model calls through `generateContent`, `generateImages`, and
  `predict` are mapped to Antigravity image generation and return inline/base64
  image payloads plus local `generatedFiles/*` metadata. Image options are
  accepted from top-level fields, `config`, `generationConfig`, `parameters`,
  or nested `imageConfig`, including snake_case aliases for `aspectRatio`,
  `imageSize`, `numberOfImages`, and `sampleCount`. `generateImages` repeats
  the local generation call up to 8 images when `numberOfImages` or
  `sampleCount` is provided.
- Generated files are stored locally under `data/gemini_generated_files`;
  override with `ANTIGRAVITY_GEMINI_GENERATED_FILES_DIR`. Generated file
  resources expose Gemini File-like metadata including `downloadUri`,
  base64-encoded `sha256Hash`, `state: ACTIVE`, and `source: GENERATED`.

Video generation:

- Veo-style model names such as `veo-*` are recognized as Gemini video model
  resources and expose `generateVideos` plus `predictLongRunning`.
- Video generation requests are stored as Gemini long-running `operations/*`
  and can be retrieved through the normal operations endpoints.
- Because the current Antigravity backend does not expose native video
  generation, video operations currently complete with an explicit
  `UNIMPLEMENTED` error instead of returning a 404 or silently mapping to text.

Live API:

- The Gemini Live WebSocket envelope is implemented for text turn flows:
  `setup` returns `setupComplete`, and `clientContent` with `turnComplete`
  produces `serverContent.modelTurn`.
- Live `setup` forwards `systemInstruction`, `generationConfig`,
  `safetySettings`, `tools`, and `toolConfig`; `response_format` wrappers are
  normalized into `generationConfig` before generation.
- Realtime audio/video `realtimeInput` is explicitly rejected with
  `UNIMPLEMENTED` because the current Antigravity backend is request/response
  oriented and does not expose native bidirectional media streaming.
- Live protocol errors use the same Gemini `error` payload and `google.rpc`
  details as the REST and SSE compatibility routes.

Notes:

- Model names are exposed as Gemini resources like
  `models/gemini-3-flash-agent`; pass `gemini-3-flash-agent` in the path.
- Public-style aliases such as `gemini-flash-latest`, `gemini-pro-latest`,
  and `gemini-image-latest` resolve to matching Antigravity-backed models.
- Model resources include `supportedGenerationMethods`, token limits, and
  capability metadata for the local Gemini compatibility surface.
- `models.list` supports Gemini `pageSize` / `pageToken` pagination with the
  official default page size of 50 and maximum page size of 1000. `/v1/models`
  keeps the OpenAI model-list shape by default, but returns the Gemini
  `{"models": ...}` shape when Gemini pagination query parameters are present.
- Gemini list-style routes accept both REST-style `pageSize` / `pageToken` and
  SDK-style `page_size` / `page_token` query aliases. Batch and operation list
  routes also accept `return_partial_success` as an alias for
  `returnPartialSuccess`.
- `countTokens` and `computeTokens` are approximate because Antigravity's internal endpoint does not
  expose a separate Gemini token-count RPC. Responses include `totalTokens`,
  `promptTokensDetails`, `cachedContentTokenCount`, and `cacheTokensDetails`
  fields for Gemini SDK compatibility. `computeTokens` returns deterministic
  `tokensInfo` with stable local token IDs and base64-encoded token bytes.
  `generateContentRequest` wrappers, string `contents`, and local
  `cachedContent` / `file_search` context are expanded before counting. Tool
  declarations and `toolConfig` are also included in the local prompt estimate.
- Files are stored locally under `data/gemini_files` by default; override with
  `ANTIGRAVITY_GEMINI_FILES_DIR`. File resources include Gemini-style
  `downloadUri`, `source`, base64 `sha256Hash`, and video metadata fields when
  available. `POST /v1beta/files` supports official metadata-only File
  creation, the same Files API surface is also available under `/v1`, and
  `files:register` supports Gemini's `uris` array shape. `files.list` uses
  Gemini's default page size of 10 and maximum page size of 100.
- Cached contents are stored locally under `data/gemini_cached_contents` by
  default; override with `ANTIGRAVITY_GEMINI_CACHED_CONTENTS_DIR`.
  `cachedContents` is available under both `/v1` and `/v1beta`;
  `cachedContents.list` supports Gemini `pageSize` / `pageToken` pagination and
  coerces page sizes above 1000 down to 1000.
- Corpora are stored locally under `data/gemini_corpora` by default; override
  with `ANTIGRAVITY_GEMINI_CORPORA_DIR`.
- Batch operations are stored locally under `data/gemini_operations` by default;
  override with `ANTIGRAVITY_GEMINI_OPERATIONS_DIR`. The top-level
  `operations.list` endpoint supports `filter` terms for common fields such as
  `done`, `name`, `metadata.*`, and `error.status`, plus
  `returnPartialSuccess` / `return_partial_success`.
- Batch resources are stored locally under `data/gemini_batches` by default;
  override with `ANTIGRAVITY_GEMINI_BATCHES_DIR`.
- Interactions are stored locally under `data/gemini_interactions` by default;
  override with `ANTIGRAVITY_GEMINI_INTERACTIONS_DIR`.
- Interactions accept Gemini-style content items such as `{"type":"text"}`,
  `{"type":"image","image_url":...}`, `inline_data`, and `file_data`;
  snake_case Gemini SDK fields are normalized to REST casing before forwarding.
- Interaction responses include `steps` with `model_output` content blocks, and
  streaming emits `interaction.step.completed` events for step-aware clients.
- Interaction create accepts SDK-style `config` / `interaction` wrappers,
  exposes `object: "interaction"`, `created` / `updated`,
  `created_at` / `updated_at`, `outputText` / `output_text`, legacy
  `createTime` / `updateTime` / `usageMetadata`, and both camelCase and
  snake_case token usage fields such as `totalTokens` / `total_tokens`,
  `total_input_tokens`, `total_output_tokens`, and
  `input_tokens_by_modality` / `output_tokens_by_modality`.
- `background=true` creates a stored `in_progress` interaction resource without
  starting generation; it can be retrieved or cancelled by name.
- Interaction cancel endpoints return the updated interaction resource.
- Remote `http`/`https` media URLs in `file_data` or `image_url` are fetched by
  the proxy and forwarded as `inlineData`; limit with
  `ANTIGRAVITY_GEMINI_REMOTE_FILE_MAX_BYTES` (default 20 MiB).
- Image-capable Gemini models can be used through Interactions and return a
  Gemini candidate containing `inlineData`, plus a local `generatedFile`.
- Generated files are stored locally under `data/gemini_generated_files` by
  default; override with `ANTIGRAVITY_GEMINI_GENERATED_FILES_DIR`.
- File search stores are stored locally under `data/gemini_file_search_stores`;
  override with `ANTIGRAVITY_GEMINI_FILE_SEARCH_STORES_DIR`.
- Tuned model metadata and permissions are stored locally under
  `data/gemini_tuned_models`; override with `ANTIGRAVITY_GEMINI_TUNED_MODELS_DIR`.
- Webhook configurations are stored locally under `data/gemini_webhooks`;
  override with `ANTIGRAVITY_GEMINI_WEBHOOKS_DIR`. The proxy delivers
  `webhooks.ping`, `batch.succeeded`, and `interaction.completed` callback
  events to enabled webhooks, stores delivery attempts, and signs callbacks
  with `X-Goog-Webhook-Signature` when a signing secret exists. Legacy
  `batches.*` / `interactions.*` subscriptions are matched as aliases.
- `webhooks.create` and patch accept SDK-style `config` / `webhook` wrappers,
  snake_case aliases, and update masks for `displayName`, `uri`, `targetUri`,
  `eventTypes`, `subscribedEvents`, and `state`.
- Native Veo video generation, realtime Live audio/video, real model
  tuning/training, true async long-running jobs, semantic Google embeddings,
  and semantic/vector `tools.file_search` retrieval are not fully implemented
  yet.

## Responses API Compatibility

Implemented:

- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `DELETE /v1/responses/{response_id}`
- `POST /v1/responses/{response_id}/cancel`
- `GET /v1/responses/{response_id}/input_items`
- `POST /v1/responses/input_tokens`
- `POST /v1/responses/{response_id}/compact`
- durable response storage through SQLite
- `previous_response_id`
- `store=false`
- text, image input, custom function tools, and streaming response events
- `text.format` / chat `response_format` mapped to Gemini JSON output controls
- `reasoning` mapped to Gemini thinking configuration when the selected model supports it
- Responses `web_search_preview` mapped to Gemini Google Search grounding

Unsupported OpenAI-hosted tools are rejected with an OpenAI-style 400 error,
including file search, code interpreter, computer use, MCP, shell, and
apply-patch style tools. `web_search_preview` is supported through Gemini
grounding, so its result shape is compatible for text output but not a byte-for-
byte OpenAI hosted web-search transcript.

## Hermes Custom Provider Example

Use this as an OpenAI-compatible provider:

```yaml
providers:
  antigravity:
    name: antigravity
    api: http://127.0.0.1:8765/v1
    api_key: not-used
    default_model: Gemini 3.5 Flash (High)
```

If `ANTIGRAVITY_PROXY_API_KEY` is set, use that value as `api_key`.

For a Hermes Gemini provider with a custom Base URL field, use:

```text
http://127.0.0.1:8765/v1beta
```

or your remote proxy host:

```text
http://your-host.ts.net:8765/v1beta
```

For Hermes web search, point the SearXNG backend at:

```text
http://127.0.0.1:8765
```

The proxy answers:

```text
GET /search?q=...&format=json&pageno=1
```

## Admin Model Refresh

The model catalog is refreshed at startup. You can refresh without restarting:

```bash
curl -X POST http://127.0.0.1:8765/admin/models/refresh \
  -H "Authorization: Bearer $ANTIGRAVITY_PROXY_API_KEY"
```

This endpoint is disabled unless `ANTIGRAVITY_PROXY_API_KEY` is configured.

## Run As A User Service

Example systemd user unit:

```ini
[Unit]
Description=Antigravity OpenAI Proxy
After=network-online.target

[Service]
WorkingDirectory=/opt/Antigravity-Proxy
ExecStart=/opt/Antigravity-Proxy/.venv/bin/python antigravity_proxy.py
Restart=always
Environment=ANTIGRAVITY_PROXY_ENV_FILE=/opt/Antigravity-Proxy/.env

[Install]
WantedBy=default.target
```

Install:

```bash
mkdir -p ~/.config/systemd/user
cp antigravity-proxy.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now antigravity-proxy.service
systemctl --user status antigravity-proxy.service
```

## Security Checklist Before Public Exposure

- Set `ANTIGRAVITY_PROXY_API_KEY`
- Use HTTPS/TLS
- Prefer localhost, Tailscale, WireGuard, or another private network
- Do not commit `.env`
- Do not commit OAuth credential JSON files
- Do not commit generated images, logs, caches, databases, or token files

## Troubleshooting

### `401 Invalid or missing API key`

`ANTIGRAVITY_PROXY_API_KEY` is set. Send one of:

```text
Authorization: Bearer <key>
X-API-Key: <key>
```

### `Could not resolve Antigravity project id`

Your auth file may not include `project_id`, and automatic project discovery did
not succeed. Set:

```bash
ANTIGRAVITY_PROJECT_ID=your-project-id
```

### `Antigravity refresh token is missing`

Your `ANTIGRAVITY_AUTH_FILE` does not contain a usable refresh token. Re-run your
local Antigravity authentication flow and update the auth file path.

### `Non-TTY environment detected`

When running under systemd, the proxy intentionally avoids interactive OAuth
fallbacks. Refresh credentials from an interactive terminal, then restart the
service.

### `/v1/models` does not show internal models

This is the default. To show internal `tab_*` / `chat_*` models:

```bash
ANTIGRAVITY_PROXY_INCLUDE_INTERNAL_MODELS=1
```

### Korean or non-ASCII prompts look broken in manual tests

Make sure your terminal and HTTP client send UTF-8 JSON:

```bash
curl -H "Content-Type: application/json; charset=utf-8" ...
```

## Development

```bash
pip install -r requirements.txt
pytest -q
```

The test suite avoids real upstream calls for proxy behavior.
