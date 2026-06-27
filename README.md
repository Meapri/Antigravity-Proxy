# Antigravity Proxy

Gemini REST-compatible FastAPI proxy for Antigravity / Cloud Code models.

This project lets Gemini-compatible clients talk to Antigravity-backed models
through the native Gemini REST surface, such as
`/v1beta/models/{model}:generateContent`, `streamGenerateContent`,
`countTokens`, files, cached contents, batches, and generated files. It also
includes a SearXNG-compatible `/search` endpoint backed by Google Search
grounding.

> This is an unofficial compatibility proxy. Use it on localhost, a private VPN,
> or another trusted network. If you expose it outside a trusted network, set
> `ANTIGRAVITY_PROXY_API_KEY` and put it behind TLS.

## What It Supports

- Gemini-compatible model listing: `GET /v1beta/models`
- Gemini-compatible content generation: `POST /v1beta/models/{model}:generateContent`
- Gemini-compatible streaming: `POST /v1beta/models/{model}:streamGenerateContent`
- Gemini-compatible token counting, embeddings, files, cached contents, batches, and generated files
- Gemini function calling / tools
- Vision input through Gemini `inlineData` and `fileData` content parts
- Gemini-compatible image generation through image models and `generateImages`
- SearXNG-compatible grounded search: `GET /search?q=...&format=json`
- Admin model refresh without restart: `POST /admin/models/refresh`
- Optional API key protection
- Gemini-style error responses

OpenAI-compatible client endpoints such as `/v1/chat/completions`,
`/v1/responses`, and `/v1/images/generations` have been removed from the public
compatibility surface. Use the Gemini REST API under `/v1beta` instead.

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
while admin routes keep a small JSON error shape for operational commands.

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
curl http://127.0.0.1:8765/v1beta/models
```

Generate content:

```bash
curl http://127.0.0.1:8765/v1beta/models/gemini-3-flash-agent:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "Say hello in one short sentence."}]}]
  }'
```

Streaming:

```bash
curl http://127.0.0.1:8765/v1beta/models/gemini-3-flash-agent:streamGenerateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "Stream one short sentence."}]}]
  }'
```

Count tokens:

```bash
curl http://127.0.0.1:8765/v1beta/models/gemini-3-flash-agent:countTokens \
  -H "Content-Type: application/json" \
  -d '{"contents": [{"role": "user", "parts": [{"text": "Count these tokens."}]}]}'
```

Tool call:

```bash
curl http://127.0.0.1:8765/v1beta/models/gemini-3-flash-agent:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "What is the weather in Seoul?"}]}],
    "tools": [{
      "functionDeclarations": [{
        "name": "get_current_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }]
    }],
    "toolConfig": {
      "functionCallingConfig": {
        "mode": "ANY",
        "allowedFunctionNames": ["get_current_weather"]
      }
    }
  }'
```

Grounded search:

```bash
curl "http://127.0.0.1:8765/search?q=NVIDIA%20latest%20GPU&format=json"
```

Image generation:

```bash
curl http://127.0.0.1:8765/v1beta/models/gemini-image-latest:generateImages \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A small robot reading a book at a wooden desk",
    "config": {
      "imageSize": "1K",
      "numberOfImages": 1
    }
  }'
```

## Gemini API Compatibility

The proxy also exposes a Gemini REST-compatible surface for clients that can
use a custom Gemini base URL:

```text
http://127.0.0.1:8765/v1beta
```

Remote Tailscale example:

```text
http://your-host.ts.net:8765/v1beta
```

Some clients, including Hermes Desktop's Gemini provider, only enable their
native Gemini transport when the configured Base URL contains
`generativelanguage.googleapis.com`. For those clients, use the update-safe
gateway prefix below. The proxy strips that prefix internally and serves the
same `/v1beta` Gemini REST surface:

```text
http://127.0.0.1:8765/generativelanguage.googleapis.com/v1beta
```

Remote Tailscale Hermes example:

```text
http://your-host.ts.net:8765/generativelanguage.googleapis.com/v1beta
```

Gemini stable-version aliases are also accepted for Gemini-specific routes,
such as `/v1/models/{model}:generateContent`, `/v1/files:register`,
`/v1/cachedContents`, `/v1/batches`, and `/v1/live`. Preview `/v1alpha`
gateway paths are normalized to the same `/v1beta` handlers for Gemini REST
and upload routes. OpenAI-compatible routes under `/v1`, including
`/v1/chat/completions`, `/v1/responses`, and `/v1/images/generations`, are
intentionally removed.

Common SDK spelling variants are accepted for query parameters: `page_size`,
`page_token`, `update_mask`, `upload_type`, `display_name`, and
`return_partial_success` are normalized to the Gemini REST camelCase forms.
`generateContent?alt=sse` and
`generateContent?stream=true` are treated as streaming Gemini SSE responses.
Gemini streaming responses emit only JSON `data:` chunks; OpenAI-style
`data: [DONE]` terminators are intentionally not sent on Gemini REST streams
because the official Google GenAI SDK parses every Gemini SSE data segment as
JSON. Streaming fallback errors use the same Gemini `error` payload and
`google.rpc.ErrorInfo` details as non-streaming Gemini errors.
`interactions.create` accepts the same SDK-style `config` object used by
`generateContent`, including `systemInstruction`, `generationConfig` scalar
coercion, `safetySettings`, `functionDeclarations`, `toolConfig`, and
response-format aliases.

Python `google-genai` SDK clients can use a collection-scoped custom base URL:

```python
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    http_options=types.HttpOptions(
        base_url="http://127.0.0.1:8765/v1beta",
        api_version=None,
        base_url_resource_scope=types.ResourceScope.COLLECTION,
        headers={"x-goog-api-key": "your-proxy-key"},
    ),
)

response = client.models.generate_content(
    model="gemini-3-flash-agent",
    contents="Say hello.",
)
```

JavaScript `@google/genai` clients can use the same collection-scoped Vertex
Express style:

```js
import {GoogleGenAI} from "@google/genai";

const ai = new GoogleGenAI({
  vertexai: true,
  apiKey: "your-proxy-key",
  httpOptions: {
    baseUrl: "http://127.0.0.1:8765/v1beta",
    apiVersion: "",
    baseUrlResourceScope: "collection",
  },
});

const response = await ai.models.generateContent({
  model: "gemini-3-flash-agent",
  contents: "Say hello.",
});
```

For those SDK modes, the proxy accepts Vertex collection aliases such as
`/v1beta/publishers/google/models/{model}:generateContent` and full Vertex
resource paths such as
`/v1beta/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`
for every `/v1beta/models/{model}:...` model method implemented by the proxy,
plus model list/get aliases. Model names may be passed as a plain model ID,
`models/{model}`, `publishers/google/models/{model}`,
`projects/{project}/locations/{location}/publishers/google/models/{model}`, or
`google/{model}`.

Implemented Gemini-compatible routes:

- `GET /v1/models/{model}`
- `GET /v1/models/{model}/operations`
- `GET /v1/models/{model}/operations/{operation}`
- `POST /v1/models/{model}/operations/{operation}:wait`
- `POST /v1/models/{model}/operations/{operation}:cancel`
- `DELETE /v1/models/{model}/operations/{operation}`
- `POST /v1/models/{model}:generateContent`
- `POST /v1/models/{model}:streamGenerateContent`
- `POST /v1/dynamic/{dynamic_model}:generateContent`
- `POST /v1/dynamic/{dynamic_model}:streamGenerateContent`
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
- `POST /v1beta/dynamic/{dynamic_model}:generateContent`
- `POST /v1beta/dynamic/{dynamic_model}:streamGenerateContent`
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
- `POST /v1/agents`
- `GET /v1/agents`
- `GET /v1/agents/{agent}`
- `DELETE /v1/agents/{agent}`
- `POST /v1beta/agents`
- `GET /v1beta/agents`
- `GET /v1beta/agents/{agent}`
- `DELETE /v1beta/agents/{agent}`
- `POST /v1/interactions`
- `GET /v1/interactions`
- `GET /v1/interactions/{interaction}`
- `POST /v1/interactions/{interaction}/cancel`
- `POST /v1/interactions/{interaction}:cancel`
- `DELETE /v1/interactions/{interaction}`
- `POST /v1beta/interactions`
- `GET /v1beta/interactions`
- `GET /v1beta/interactions/{interaction}`
- `POST /v1beta/interactions/{interaction}/cancel`
- `POST /v1beta/interactions/{interaction}:cancel`
- `DELETE /v1beta/interactions/{interaction}`
- `WS /ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`
- `WS /v1beta/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`
- `WS /generativelanguage.googleapis.com/v1beta/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`
- `WS /v1alpha/live`
- `WS /v1beta/live`
- `WS /v1/live`
- `POST /v1/auth_tokens`
- `POST /v1beta/auth_tokens`
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
- `GET /v1/generatedFiles/{generated_file}/operations/{operation}`
- `POST /v1/generatedFiles/operations/{operation}:wait`
- `POST /v1/generatedFiles/operations/{operation}:cancel`
- `DELETE /v1/generatedFiles/operations/{operation}`
- `GET /v1beta/generatedFiles`
- `GET /v1beta/generatedFiles/{generated_file}`
- `GET /v1beta/generatedFiles/{generated_file}:download`
- `DELETE /v1beta/generatedFiles/{generated_file}`
- `GET /v1beta/generatedFiles/operations`
- `GET /v1beta/generatedFiles/operations/{operation}`
- `GET /v1beta/generatedFiles/{generated_file}/operations/{operation}`
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
- `GET /v1/corpora/{corpus}/operations/{operation}`
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
- `GET /v1beta/corpora/{corpus}/operations/{operation}`
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
- `POST /v1/tunedModels/{tuned_model}:generateText`
- `POST /v1/tunedModels/{tuned_model}:batchGenerateContent`
- `POST /v1/tunedModels/{tuned_model}:transferOwnership`
- `POST /v1/tunedModels/{tuned_model}:countTokens`
- `POST /v1/tunedModels/{tuned_model}:computeTokens`
- `POST /v1/tunedModels/{tuned_model}:embedContent`
- `POST /v1/tunedModels/{tuned_model}:batchEmbedContents`
- `POST /v1/tunedModels/{tuned_model}:asyncBatchEmbedContent`
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
- `POST /v1beta/tunedModels/{tuned_model}:generateText`
- `POST /v1beta/tunedModels/{tuned_model}:batchGenerateContent`
- `POST /v1beta/tunedModels/{tuned_model}:transferOwnership`
- `POST /v1beta/tunedModels/{tuned_model}:countTokens`
- `POST /v1beta/tunedModels/{tuned_model}:computeTokens`
- `POST /v1beta/tunedModels/{tuned_model}:embedContent`
- `POST /v1beta/tunedModels/{tuned_model}:batchEmbedContents`
- `POST /v1beta/tunedModels/{tuned_model}:asyncBatchEmbedContent`
- `GET /v1beta/tunedModels/{tuned_model}/permissions`
- `POST /v1beta/tunedModels/{tuned_model}/permissions`
- `GET /v1beta/tunedModels/{tuned_model}/permissions/{permission}`
- `PATCH /v1beta/tunedModels/{tuned_model}/permissions/{permission}`
- `POST /v1beta/tunedModels/{tuned_model}/permissions/{permission}:transferOwnership`
- `DELETE /v1beta/tunedModels/{tuned_model}/permissions/{permission}`
- `POST /v1/tuningJobs`
- `GET /v1/tuningJobs`
- `GET /v1/tuningJobs/{tuning_job}`
- `POST /v1/tuningJobs/{tuning_job}:cancel`
- `DELETE /v1/tuningJobs/{tuning_job}`
- `POST /v1/projects/{project}/locations/{location}/tuningJobs`
- `GET /v1/projects/{project}/locations/{location}/tuningJobs`
- `POST /v1/projects/{project}/locations/{location}/tuningJobs:validateReinforcementTuningReward`
- `GET /v1/projects/{project}/locations/{location}/tuningJobs/{tuning_job}`
- `POST /v1/projects/{project}/locations/{location}/tuningJobs/{tuning_job}:cancel`
- `DELETE /v1/projects/{project}/locations/{location}/tuningJobs/{tuning_job}`
- `POST /v1beta/tuningJobs`
- `GET /v1beta/tuningJobs`
- `GET /v1beta/tuningJobs/{tuning_job}`
- `POST /v1beta/tuningJobs/{tuning_job}:cancel`
- `DELETE /v1beta/tuningJobs/{tuning_job}`
- `POST /v1beta/projects/{project}/locations/{location}/tuningJobs`
- `GET /v1beta/projects/{project}/locations/{location}/tuningJobs`
- `POST /v1beta/projects/{project}/locations/{location}/tuningJobs:validateReinforcementTuningReward`
- `GET /v1beta/projects/{project}/locations/{location}/tuningJobs/{tuning_job}`
- `POST /v1beta/projects/{project}/locations/{location}/tuningJobs/{tuning_job}:cancel`
- `DELETE /v1beta/projects/{project}/locations/{location}/tuningJobs/{tuning_job}`

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
`file_search` tools are converted into retrieved context. Gemini tool option
aliases are normalized too, including `google_search.search_types`,
`google_search.time_range_filter`,
`google_search_retrieval.dynamic_retrieval_config.dynamic_threshold`,
`file_search.file_search_store_names`, `file_search.metadata_filter`, and
`file_search.top_k`, and `thinking_config.thinking_level`. `urlContext` is
implemented as a safe local fetch-and-inject shim: URLs found in the prompt are
fetched as bounded text context, private/loopback/link-local addresses are
blocked by default, and `urlContextMetadata` is attached to candidates.
`codeExecution`, `googleMaps`, and `mcpServers` are recognized but return
`UNIMPLEMENTED` because the current Antigravity backend does not expose those
hosted tools. `computerUse` on `generateContent`, `streamGenerateContent`,
tuned-model generate routes, and `interactions.create` is mapped locally to a
predefined Computer Use `functionCall` for the client or Hermes/cua-driver loop
to execute.
`toolConfig.functionCallingConfig.mode` and `allowedFunctionNames` are
normalized from common SDK spellings. Function
calling mode aliases such as `required`, `forced`, and `force` are treated as
Gemini `ANY`, and `tool_choice` / `toolChoice` aliases on Gemini requests are
folded into `toolConfig.functionCallingConfig` for custom function tools.
`toolConfig.includeServerSideToolInvocations` and the SDK snake_case alias are
accepted and coerced to a boolean.
`toolConfig.retrievalConfig` / `retrieval_config` is also normalized, including
`language_code` and `lat_lng` with numeric latitude/longitude coercion.
`tools` may be a single object or a list, and top-level
`functionDeclarations` / `function_declarations` and singular
`functionDeclaration` / `function_declaration` are wrapped into Gemini tool
objects before forwarding. Singular function declarations are also accepted
inside `tools`. Function declaration schema aliases such as
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
The Gemini `dynamic/{dynamic_model}:generateContent` and
`:streamGenerateContent` discovery routes are accepted as aliases for the same
model generation pipeline. Tuned models also expose the discovery-listed
`generateText`, `batchGenerateContent`, and `asyncBatchEmbedContent` methods,
mapped through the same local generation and completed-operation
compatibility layers as model calls. Tuned model `transferOwnership` updates
the local tuned-model owner and permission state.
They also accept the common SDK-style top-level `config` object and merge it
into Gemini REST fields such as `generationConfig`, `systemInstruction`,
`toolConfig`, `safetySettings`, `tools`, `cachedContent`, and `labels`.
Generation config aliases include `responseSchema` / `response_schema`,
`responseJsonSchema` / `response_json_schema`, `thinkingConfig` /
`thinking_config`, `responseModalities` / `response_modalities`, and
`enableEnhancedCivicAnswers` / `enable_enhanced_civic_answers`, plus
`audioTimestamp` / `audio_timestamp` and `translationConfig` /
`translation_config`. Speech config aliases such as `speech_config`,
`language_code`, `voice_config`, `prebuilt_voice_config`, and `voice_name` are
also normalized.
`speechConfig.voiceConfig` and `speechConfig.multiSpeakerVoiceConfig` are
validated as mutually exclusive, matching Gemini's REST schema.
Thinking config `thinking_level` accepts common spellings such as
`thinking level high` and normalizes them to Gemini enum values such as `HIGH`.
Response modality aliases such as `text`, `image`, and `audio` are normalized
to Gemini REST enum values, and media resolution aliases such as `low`,
`medium`, and `high` are normalized to `MEDIA_RESOLUTION_*` values.
`service_tier` / `serviceTier` and `store` are preserved as top-level
GenerateContent request fields. Service tier aliases such as `PRIORITY`,
`SERVICE_TIER_PRIORITY`, and `flex` are normalized to the Gemini REST enum
values `priority`, `standard`, `flex`, or `unspecified`.
Google provider wrappers such as `providerOptions.google` and
`provider_options.google` are merged through the same path, with direct request
and `config` fields taking precedence.
`processingOptions` / `processing_options` and aliases such as `start_offset`
and `end_offset` are accepted for current Gemini SDK compatibility; because
Antigravity's internal endpoint does not expose that field yet, the proxy
removes it before forwarding so requests do not fail with unknown-field errors.
Video part metadata normalizes SDK-style `video_metadata.start_offset`,
`video_metadata.end_offset`, and string `fps` values to Gemini REST
`videoMetadata.startOffset`, `videoMetadata.endOffset`, and numeric `fps`.
Content parts preserve SDK-style aliases such as `function_call`,
`function_response`, `executable_code`, `code_execution_result`, and
`thought_signature` as canonical Gemini REST fields. Latest Part fields such
as `tool_call`, `tool_response`, `tool_type`, `part_metadata`, and
`function_response.will_continue` are also normalized.
Function response `scheduling` accepts common spellings such as `when idle`
and normalizes them to Gemini enum values such as `WHEN_IDLE`.
Generated responses canonicalize common SDK/upstream snake_case metadata aliases
for `promptFeedback`, `citationMetadata`, `groundingMetadata`,
`groundingAttributions`,
`urlContextMetadata`, `logprobsResult`, and `modelStatus`, including nested
grounding chunk/support fields such as `retrieved_context`, `image_uri`,
`place_id`, `grounding_chunk_indices`, `confidence_scores`, `source_id`,
`grounding_passage`, `semantic_retriever_chunk`, and logprobs `token_id`.
Latest enum spellings for `UrlMetadata.urlRetrievalStatus`,
`ModelStatus.modelStage`, and `ComputerUse` options are normalized from common
SDK-style aliases to Gemini REST values.
String `stopSequences` and `responseModalities` values are normalized to the
Gemini REST list form. Numeric and boolean generation config values such as
`maxOutputTokens`, `temperature`, `topK`, and `responseLogprobs` are coerced
from common string forms when possible.
SDK transport-only options such as `httpOptions`, `requestOptions`,
`apiVersion`, and `baseUrl` are ignored rather than forwarded as API payload.
`response_format` / `responseFormat` wrappers and direct
`generationConfig.responseSchema` / SDK `config.response_schema` values are
mapped into Gemini-compatible JSON output controls.
`generationConfig.responseFormat` / `generation_config.response_format` is
handled through the same path, including nested `jsonSchema` / `json_schema`
objects and Gemini's `_responseJsonSchema` alias. Official
`customMetadata.string_list_value` entries are normalized to
`customMetadata.stringListValue`. Resource body aliases such as `base_model`,
`create_time`, `update_time`, and `expire_time` are normalized to their Gemini
REST casing. Tuning and batch aliases such as `text_input`, `learning_rate`,
`learning_rate_multiplier`, `batch_stats`, `request_count`, `responses_file`,
and `inlined_responses` are also normalized.
`responseFormat.text`, `.image`, and `.audio` configs are preserved, with
snake_case fields such as `mime_type`, `image_size`, `sample_rate`, and
`bit_rate` normalized to Gemini REST casing.
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
returned in Gemini REST camelCase form. Candidate `finishReason` accepts common
spellings such as `max tokens` and normalizes them to Gemini enum values such
as `MAX_TOKENS`. Safety rating values normalize common harm category and
probability spellings such as `harassment` / `medium` to Gemini enum values.
Gemini compatibility errors use the `google.rpc.Status`-style
`error.code` / `error.message` / `error.status` shape. `INVALID_ARGUMENT`
responses include `google.rpc.BadRequest` field-violation details, and
unsupported hosted features include `google.rpc.ErrorInfo` details so SDKs can
distinguish unsupported proxy features from malformed requests.
Gemini request validation failures, including invalid query parameters, return
`400 INVALID_ARGUMENT` rather than OpenAI-style validation errors.
Unmatched Gemini routes and uncaught Gemini `HTTPException`s also return this
Gemini error shape. Removed OpenAI-compatible routes return Gemini-style
`404 NOT_FOUND` errors.
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
`X-Goog-Upload-Offset`. Finalized uploads return
`X-Goog-Upload-Status: final`, and SDK-prefixed upload paths such as
`/v1beta/upload/v1beta/files` are accepted as aliases for
`/upload/v1beta/files`; this path is covered by a real `google-genai`
`files.upload()` compatibility test. Upload metadata accepts both official
`file` objects and SDK-style `config` wrappers such as
`{"config": {"mimeType": "text/plain"}}`.
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
proxy `{"file": ...}` form remains available as a compatibility alias, and
SDK-style `{"config":{"file": ...}}` metadata is accepted for single-file
registration and metadata-only creation. File
`state` and `source` values are normalized to Gemini-style enum names such as
`ACTIVE`, `FAILED`, `UPLOADED`, and `REGISTERED` across create, list, and get
responses. Metadata-only registered files receive a deterministic base64
`sha256Hash` when one is not supplied. SDK-style `video_metadata.video_duration`
is normalized to Gemini REST `videoMetadata.videoDuration`.

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
`fileData.fileUri` may also be a file resource object, and `fileData.file`
wrappers are accepted for SDK/adapter compatibility.
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

Then pass the returned `name` as `cachedContent` in `generateContent`. SDK-style
resource objects such as `{"cachedContent": {"name": "cachedContents/..."}}`
are also accepted by `generateContent` and `countTokens`. The proxy
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
  `config` / `embedContentConfig` / `embedding_config`, string or list
  `contents`, and the legacy top-level `outputDimensionality`, `taskType`, and
  `title` fields. They are
  normalized from either camelCase or snake_case, including string numeric
  `outputDimensionality`, common `taskType` spellings such as
  `retrieval document`, and boolean config fields such as `autoTruncate`,
  `documentOcr`, and `audioTrackExtraction`. They are stable and shaped like
  Gemini embeddings, but they are not semantic Google embedding model outputs
  because Antigravity does not expose a public embedding RPC. Responses include
  `usageMetadata.promptTokenCount` and `usageMetadata.promptTokenDetails` for
  the embedded input text. Single and batch embed requests accept SDK wrapper
  bodies such as `{"request": {...}}`,
  `{"embedContentRequest": {...}}`, and `providerOptions.google` /
  `provider_options.google` embedding config.
- Standard Gemini embedding model names such as `text-embedding-004`,
  `embedding-001`, and `gemini-embedding-001` are accepted on
  `embedContent` / `batchEmbedContents`; this covers `google-genai`
  `client.models.embed_content()`, which calls `:batchEmbedContents` in
  Developer API mode.
- `asyncBatchEmbedContent` stores the deterministic batch embedding result as
  an immediately completed local operation and batch resource. The create
  response exposes the reusable `batches/*` resource as top-level `name` for
  `google-genai` `client.batches.get/cancel/delete` compatibility, while the
  stored operation remains available under `metadata.operation`. It accepts
  raw `requests` plus SDK wrappers such as `embedContentBatch` /
  `embed_content_batch`, including shared `config` embedding options.
- `batchGenerateContent` runs requests synchronously through Antigravity and
  stores immediately completed operation and `batches/*` results with Gemini
  status values and `stats` counters. The create response uses top-level
  `name: "batches/..."` and preserves the operation name in
  `metadata.operation`, so both SDK batch lifecycle calls and direct
  operation polling are supported. Inline batch items may be plain request
  objects or SDK/REST wrapper items such as `{"request": {...}}` /
  `{"generateContentRequest": {...}}`. Python SDK inline source wrappers such
  as `batch.inputConfig.requests.requests[]` are also accepted, and Developer
  SDK file sources such as `batch.inputConfig.fileName: "files/..."` are read
  from the local Files API as JSON or JSONL batch input.
- `batches.create` accepts inline `requests` plus `model` and returns the
  completed `batches/*` operation view; the original operation is preserved
  under `metadata.operation` and the local batch resource is preserved under
  `metadata.batchResource`. It also accepts common SDK wrapper bodies
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
  and `priority`, including SDK-style paths such as `display_name`,
  `batch.display_name`, `generate_content_batch.display_name`, and
  `embed_content_batch.priority`. They return the same completed `batches/*`
  operation view.
- `batches.list` supports `filter` terms for common operation and batch fields
  such as `done`, `displayName`, `state`, `model`, and
  `metadata.batchResource.*`; `returnPartialSuccess` /
  `return_partial_success` returns an empty `unreachable` list for local stores.
- Scoped operation routes under models, generated files, file search stores,
  and tuned models validate that the operation belongs to the requested parent.
  `:cancel` on scoped routes stores a Gemini-style `CANCELLED` operation state
  for pending operations before returning the empty cancel response.
  Generated-file and file-search upload operation response shapes are covered
  by contract tests for get, scoped get, wait, cancel, and delete routes.
- `predict` and `predictLongRunning` are mapped to Gemini `generateContent`
  requests and return prediction/operation-shaped compatibility responses.
  Vertex-style `instances` requests preserve the same SDK `config`,
  `providerOptions.google`, and `processingOptions` compatibility path used by
  `generateContent`. `parameters` may carry Gemini generation config fields or
  top-level generate fields such as `safetySettings` / `safety_settings` and
  `toolConfig` / `tool_config`; these are split and normalized before
  forwarding. `predictLongRunning` stores a completed operation with
  normalized request metadata, timestamps, and `deployedModelId`.
- Legacy `generateText`, `generateMessage`, `generateAnswer`, `embedText`,
  `batchEmbedText`, `countTextTokens`, and `countMessageTokens` are accepted and
  mapped onto the newer local `generateContent`, embedding, and token-count
  compatibility paths. Legacy text/message generation also accepts SDK
  `config`, `providerOptions.google`, `response_format`, and
  `processingOptions` wrappers before forwarding to the generate path. Legacy
  message prompts accept both `prompt.messages[]` and single `prompt.message`
  shapes.

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
- `fileSearchStores.list` accepts `pageSize` / `pageToken` and snake_case
  `page_size` / `page_token`, defaulting to 10 items with Gemini's page size
  capped at 20.
- Deleting an indexed file search document also requires `force=true`, matching
  the Gemini REST `fileSearchStores.documents.delete` parameter.
- `importFile` imports files previously uploaded through the local Files API and
  preserves document display names, custom metadata, and document
  `chunkingConfig` when supplied. It accepts `fileMetadata` /
  `file_metadata` wrappers in addition to direct fields, and returns scoped
  operation names such as `fileSearchStores/{store}/operations/{operation}`.
  Operation responses include SDK-readable `parent` and `documentName` fields.
- `uploadToFileSearchStore` accepts raw, multipart, or JSON direct uploads and
  stores documents locally while preserving custom metadata and
  `chunkingConfig`. JSON uploads may provide document metadata through
  `file`, `fileMetadata`, or `file_metadata` wrappers. Upload operation names
  use `fileSearchStores/{store}/upload/operations/{operation}`.
- `fileSearchStores.documents.list` accepts `pageSize` / `pageToken` and
  snake_case `page_size` / `page_token`, defaulting to 10 items with the
  Gemini REST page size capped at 20.
- `google-genai` `file_search_stores.documents.get/list/delete` is supported.
  Document resources use the Gemini `DocumentState` enum values such as
  `STATE_ACTIVE`; older locally stored `ACTIVE` values are normalized on read.
- `tools.file_search` / `tools.fileSearch` performs local lexical retrieval
  against these stores and injects the best matching document snippets into the
  outgoing Gemini request context.

Corpora and semantic retriever:

- Legacy `corpora`, `documents`, `chunks`, and corpus `permissions` are
  implemented as a local compatibility layer under `data/gemini_corpora`.
- Corpus and document `:query` perform local lexical chunk matching. They return
  Gemini-shaped `relevantChunks`, but they are not semantic Google retriever
  scores because Antigravity does not expose that service.
- Corpus create and patch calls accept SDK-style `corpus` wrapper bodies.
- Corpus, document, and chunk patch calls honor `updateMask` / `update_mask`
  for supported mutable fields, including wrapper paths such as
  `corpus.displayName`, `document.customMetadata`, and `chunk.data`.
- Deleting a corpus that still contains documents requires `force=true`, matching
  the Gemini REST delete parameter behavior.
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
- Tuned model routes accept both short IDs such as `my_tuned` and full resource
  names embedded in the path, such as
  `/v1beta/tunedModels/tunedModels/my_tuned:generateContent`.
- `tunedModels.patch` honors `updateMask` / `update_mask` from either the query
  string or SDK wrapper body for mutable metadata fields such as `displayName`,
  `description`, `baseModel`, `tuningTask`, and `readerProjectNumbers`.
- `tunedModels.list` supports Gemini `filter` expressions over name,
  `displayName`, description, state, and base model metadata. `pageSize`
  values above 1000 are clamped to 1000 for SDK paginator compatibility.
- `tunedModels/{id}:generateContent`, `:streamGenerateContent`,
  `:generateText`, `:batchGenerateContent`, `:countTokens`,
  `:computeTokens`, `:embedContent`, `:batchEmbedContents`, and
  `:asyncBatchEmbedContent` forward to or reuse the configured `baseModel`
  compatibility path.
- `permissions` are stored locally for Gemini SDK compatibility; corpus and
  tuned-model permission create/patch accept `permission` wrappers plus
  snake_case aliases such as `email_address` and `grantee_type`, normalizing
  role/grantee enums to Gemini-style uppercase values. Permission patch honors
  `updateMask` / `update_mask` for `role`, `granteeType`, and `emailAddress`,
  and tuned-model permission routes also accept full permission resource names
  in the `{permission}` path segment. Permission list routes support
  `pageSize` / `pageToken` and SDK `page_size` / `page_token`, default to 10
  items, and clamp oversized pages to 1000.

Vertex tuning jobs:

- Vertex-style `tuningJobs` are stored locally under
  `data/gemini_tuning_jobs` by default; override with
  `ANTIGRAVITY_GEMINI_TUNING_JOBS_DIR`.
- The proxy supports collection and project-scoped paths such as
  `POST /v1beta/tuningJobs` and
  `POST /v1beta/projects/{project}/locations/{location}/tuningJobs`.
- `google-genai` Vertex/Enterprise `client.tunings.tune`, `get`, `list`,
  `cancel`, `validate_reward`, and direct REST delete are accepted. Create
  requests preserve `baseModel`, `preTunedModel`, tuning specs, labels,
  description, `tunedModelDisplayName`, output URI, encryption, evaluation,
  and service account metadata, then return an immediately completed local
  `TuningJob`. Reinforcement reward validation checks the request shape and
  returns deterministic local `overallReward` / `rewardInfoDetails` values for
  SDK compatibility.
- This is a compatibility shim, not a managed training service: it does not run
  real Gemini/Vertex tuning or produce a newly trained upstream model. Use
  `tunedModels` aliases when you need a locally named model that forwards to an
  existing Antigravity base model.

Generated files:

- Gemini image model calls through `generateContent`, `generateImages`, and
  `predict` are mapped to Antigravity image generation and return inline/base64
  image payloads plus local `generatedFiles/*` metadata. Image options are
  accepted from top-level fields, `config`, `generationConfig`, `parameters`,
  or nested `imageConfig`, including snake_case aliases for `aspectRatio`,
  `imageSize`, `numberOfImages`, and `sampleCount`; `prompt` / `text` may also
  be supplied through those wrappers. `generateImages` repeats the local
  generation call up to 8 images when `numberOfImages` or `sampleCount` is
  provided.
- Generated files are stored locally under `data/gemini_generated_files`;
  override with `ANTIGRAVITY_GEMINI_GENERATED_FILES_DIR`. Generated file
  resources expose Gemini File-like metadata including `downloadUri`,
  base64-encoded `sha256Hash`, `state: GENERATED`, and `source: GENERATED`.
  Generated file get, download, list, scoped operation, wait, cancel, and
  delete routes are tested against files produced by image generation calls.
  Generated-file operations are stored under scoped names such as
  `generatedFiles/{generated_file}/operations/{operation}`.

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
  details as the REST and SSE compatibility routes. WebSocket aliases are
  available at `/v1/live`, `/v1beta/live`, `/v1alpha/live`, `/v1beta/ws/...`,
  and the gateway-prefixed `/generativelanguage.googleapis.com/v1beta/ws/...`.
- Experimental Developer API auth tokens are implemented through
  `POST /v1beta/auth_tokens` and `POST /v1/auth_tokens`. The proxy stores local
  token metadata under `data/gemini_auth_tokens` by default; override with
  `ANTIGRAVITY_GEMINI_AUTH_TOKENS_DIR`. Created tokens are returned as
  `authTokens/...` resources and can authenticate Live WebSocket sessions via
  `access_token`, `key`, `x-goog-api-key`, `x-api-key`, `Bearer`, or `Token`
  credentials until they expire or exhaust their optional `uses` count.

Notes:

- Model names are exposed as Gemini resources like
  `models/gemini-3-flash-agent`; pass `gemini-3-flash-agent` in the path.
- Public-style aliases such as `gemini-flash-latest`, `gemini-pro-latest`,
  and `gemini-image-latest` resolve to matching Antigravity-backed models.
- Model resources include `supportedGenerationMethods`, token limits, and
  capability metadata for the local Gemini compatibility surface.
- `models.list` supports Gemini `pageSize` / `pageToken` pagination with the
  official default page size of 50 and maximum page size of 1000, plus simple
  `filter` terms over fields such as `name`, `displayName`,
  `supportedGenerationMethods`, `capabilities.*`, and `metadata.*`.
  `/v1/models` and `/v1beta/models` return the Gemini `{"models": ...}` shape.
- `google-genai` `client.models.update` / `delete` is supported for mutable
  `tunedModels/...` resources. Base Antigravity model resources remain
  read-only and return Gemini `UNIMPLEMENTED` for update/delete because there is
  no safe upstream operation for modifying or deleting the fixed model catalog.
- Gemini list-style routes accept both REST-style `pageSize` / `pageToken` and
  SDK-style `page_size` / `page_token` query aliases. Batch and operation list
  routes also accept `filter`, `return_partial_success`, and
  `returnPartialSuccess`.
- List routes clamp oversized `pageSize` values where the Gemini schema defines
  a hard maximum, including `models` at 1000, `generatedFiles` at 50, and
  `corpora` at 20. Invalid `pageToken` values return Gemini-style
  `INVALID_ARGUMENT` instead of silently replaying the first page.
- Agents preserve caller-supplied `id` values, return `object: "agent"`, and
  expose list results with the SDK-style `object/data/next_page_token` shape
  while keeping `agents/nextPageToken` aliases for older clients.
- Webhooks accept snake_case SDK fields such as `subscribed_events` and
  `new_signing_secret`. Public responses expose `subscribed_events`,
  `signing_secrets`, `new_signing_secret`, and snake_case timestamps; ping
  returns an empty response and `rotateSigningSecret` returns `{ "secret": ... }`.
  Webhooks and interactions list responses also expose SDK-style
  `object/data/next_page_token` aliases while retaining their original
  resource-specific list fields.
- Interactions streaming uses Gemini's `event_type` SSE discriminator with
  `step.start`, `step.delta`, `step.stop`, and `interaction.completed` events.
- Repeated slashes in Gemini SDK paths are normalized before routing, so
  collection-base clients that produce paths such as `/v1beta//interactions`
  are accepted without a redirect.
- File Search Store int64 counters, including `activeDocumentsCount`,
  `pendingDocumentsCount`, `failedDocumentsCount`, and `sizeBytes`, are returned
  as strings per the Gemini REST schema.
- `file_search_stores.upload_to_file_search_store()` from `google-genai` is
  supported with the resumable upload start/finalize flow, including collection
  base URL aliases such as `/v1beta/upload/v1beta/fileSearchStores/...`.
- `file_search_stores.documents.get/list/delete()` from `google-genai` is
  covered by SDK tests, including `delete(..., config={"force": True})` for
  indexed local documents.
- Local `corpora` document and chunk creation follows create semantics:
  duplicate caller-supplied document or chunk IDs return Gemini-style
  `ALREADY_EXISTS` instead of silently overwriting. Permission patch operations
  require `updateMask=role`; grantee and email fields remain immutable after
  creation.
- `countTokens` and `computeTokens` are approximate because Antigravity's internal endpoint does not
  expose a separate Gemini token-count RPC. Responses include `totalTokens`,
  `promptTokensDetails`, `cachedContentTokenCount`, and `cacheTokensDetails`
  fields for Gemini SDK compatibility. `computeTokens` returns deterministic
  `tokensInfo` with stable local token IDs and base64-encoded token bytes.
  `generateContentRequest` wrappers, string `contents`, and local
  `cachedContent` / `file_search` context are expanded before counting. Tool
  declarations and `toolConfig` are also included in the local prompt estimate.
- `generateContent`, `streamGenerateContent`, and token-count routes accept
  SDK/AIP wrappers such as `generateContentRequest` and `request`, then unwrap
  them before applying normal content, config, tool, cache, URL, and file-search
  normalization.
- Generate request `Content.role` values are normalized to Gemini-compatible
  roles: `assistant` becomes `model`, and OpenAI-style `system` / `developer`
  inputs are folded into `user`. `systemInstruction` is emitted with a
  Gemini-valid role instead of the OpenAI-only `system` role.
- Gemini native `computer_use` tools on `generateContent` and
  `streamGenerateContent` return a local Gemini `functionCall` action instead
  of `UNIMPLEMENTED`; other hosted-only built-ins such as `code_execution`,
  `google_maps`, and MCP servers remain explicit `UNIMPLEMENTED` responses.
- Upstream Gemini usage metadata aliases such as `prompt_token_count`,
  `candidates_token_count`, `total_token_count`, and `service_tier` are
  normalized to `promptTokenCount`, `candidatesTokenCount`,
  `totalTokenCount`, and `serviceTier`; nested modality entries normalize
  `token_count` to `tokenCount` and common lowercase modalities such as
  `document` to Gemini enum values such as `DOCUMENT`.
- Files are stored locally under `data/gemini_files` by default; override with
  `ANTIGRAVITY_GEMINI_FILES_DIR`. File resources include Gemini-style
  `downloadUri`, `source`, base64 `sha256Hash`, and video metadata fields when
  available. `POST /v1beta/files` supports official metadata-only File
  creation, the same Files API surface is also available under `/v1`, and
  `files:register` follows the Gemini `uris[] -> files[]` shape. `files.list`
  uses Gemini's default page size of 10 and maximum page size of 100. File
  responses omit non-schema fields such as `customMetadata`.
- Cached contents are stored locally under `data/gemini_cached_contents` by
  default; override with `ANTIGRAVITY_GEMINI_CACHED_CONTENTS_DIR`.
  `cachedContents` is available under both `/v1` and `/v1beta`;
  `cachedContents.list` supports Gemini `pageSize` / `pageToken` pagination and
  coerces page sizes above 1000 down to 1000. Vertex-style project-scoped cache
  paths such as
  `/v1beta/projects/{project}/locations/{location}/cachedContents/{cache}` are
  accepted for `google-genai` Vertex cache lifecycle compatibility. Create
  requests require `model`, matching the Gemini `CachedContent` schema.
  Cached content responses are filtered to Gemini `CachedContent` fields and
  expose `usageMetadata.totalTokenCount`.
- Corpora are stored locally under `data/gemini_corpora` by default; override
  with `ANTIGRAVITY_GEMINI_CORPORA_DIR`.
- Batch operations are stored locally under `data/gemini_operations` by default;
  override with `ANTIGRAVITY_GEMINI_OPERATIONS_DIR`. The top-level
  `operations.list` endpoint supports `filter` terms for common fields such as
  `done`, `name`, `metadata.*`, and `error.status`, plus
  `returnPartialSuccess` / `return_partial_success`.
- Batch resources are stored locally under `data/gemini_batches` by default;
  override with `ANTIGRAVITY_GEMINI_BATCHES_DIR`.
- Vertex-style `batchPredictionJobs` are available under both collection and
  project-scoped paths such as `POST /v1beta/batchPredictionJobs` and
  `POST /v1beta/projects/{project}/locations/{location}/batchPredictionJobs`.
  The proxy accepts GCS, BigQuery, and Vertex dataset `inputConfig` /
  `outputConfig` metadata, stores a completed local job, and returns the
  Vertex resource shape expected by `google-genai` `client.batches.create`,
  `get`, `list`, `cancel`, and `delete`. It does not execute a real managed
  Vertex batch job or write remote output files.
- Agents are stored locally under `data/gemini_agents` by default; override
  with `ANTIGRAVITY_GEMINI_AGENTS_DIR`. Agents support `displayName`,
  `description`, `model`, `systemInstruction`, `tools`, `toolConfig`,
  `baseEnvironment`, and SDK-style snake_case aliases.
- Interactions are stored locally under `data/gemini_interactions` by default;
  override with `ANTIGRAVITY_GEMINI_INTERACTIONS_DIR`.
- Interactions accept Gemini-style content items such as `{"type":"text"}`,
  `{"type":"image","image_url":...}`, `inline_data`, and `file_data`;
  snake_case Gemini SDK fields are normalized to REST casing before forwarding.
- Interaction responses include SDK-recognized `steps` with `model_output`
  content blocks. The proxy avoids non-standard step types in stored create
  responses so `google-genai` can parse the interaction without `UnknownStep`
  fallbacks. Streaming emits `interaction.step.completed` events for step-aware
  clients.
- Interactions with `tools: [{"type":"computer_use","environment":"browser"}]`
  return `status: "requires_action"` with a Computer Use `functionCall` instead
  of forwarding the hosted tool to the Antigravity upstream.
- Interaction create accepts SDK-style `config` / `interaction` wrappers,
  exposes `object: "interaction"`, `created` / `updated`,
  `created_at` / `updated_at`, `outputText` / `output_text`, legacy
  `createTime` / `updateTime` / `usageMetadata`, and both camelCase and
  snake_case token usage fields such as `totalTokens` / `total_tokens`,
  `total_input_tokens`, `total_output_tokens`, and
  `input_tokens_by_modality` / `output_tokens_by_modality`.
- Interaction create can reference a stored `agent`; the proxy merges the
  agent's model, system instruction, tools, tool config, and base environment
  into the generated request unless the interaction overrides them.
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
  Caller-supplied `tunedModelId` values are validated against Gemini's ID
  pattern, and duplicate IDs return `ALREADY_EXISTS`.
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

## Removed OpenAI Compatibility

The previous OpenAI-compatible surface has been removed from the public
runtime:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `DELETE /v1/responses/{response_id}`
- `POST /v1/responses/{response_id}/cancel`
- `GET /v1/responses/{response_id}/input_items`
- `POST /v1/responses/input_tokens`
- `POST /v1/responses/{response_id}/compact`
- `POST /v1/images/generations`

Use Gemini-native equivalents under `/v1beta`, such as `generateContent`,
`streamGenerateContent`, `countTokens`, `generateImages`, files, cached
contents, batches, and interactions.

## Hermes Gemini Provider Example

For a generic Gemini-compatible client with a custom Base URL field, use:

```text
http://127.0.0.1:8765/v1beta
```

or your remote proxy host:

```text
http://your-host.ts.net:8765/v1beta
```

For Hermes Desktop's Gemini provider, use the host-prefixed proxy URL below.
Hermes uses the `generativelanguage.googleapis.com` substring to select its
native Gemini transport; the proxy strips the prefix before routing:

```text
http://127.0.0.1:8765/generativelanguage.googleapis.com/v1beta
```

Remote Tailscale Hermes example:

```text
http://your-host.ts.net:8765/generativelanguage.googleapis.com/v1beta
```

If Hermes shows the default Gemini Base URL
`https://generativelanguage.googleapis.com/v1beta`, replace that whole value
with the prefixed proxy URL above. Do not leave the Google-hosted URL there
unless you want Hermes to call Google's Gemini API directly.

| Hermes field | Value |
| --- | --- |
| Provider type | Gemini |
| Base URL | `http://127.0.0.1:8765/generativelanguage.googleapis.com/v1beta` |
| Tailscale Base URL | `http://<tailscale-host>.ts.net:8765/generativelanguage.googleapis.com/v1beta` |
| API key | `ANTIGRAVITY_PROXY_API_KEY`, or any placeholder if proxy auth is disabled |
| Model | `gemini-3-flash-agent` |

Hermes/Gemini clients may send the key as `x-goog-api-key`,
`Authorization: Bearer <key>`, `X-API-Key: <key>`, or `?key=<key>`. The proxy
accepts all of these when `ANTIGRAVITY_PROXY_API_KEY` is configured.

To see Gemini-compatible model ids:

```bash
curl http://127.0.0.1:8765/v1beta/models \
  -H "x-goog-api-key: $ANTIGRAVITY_PROXY_API_KEY"
```

Most clients want the model id without the `models/` prefix, for example
`gemini-3-flash-agent`.

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

The repository includes `antigravity-proxy.service` as a starting point. Edit
its `WorkingDirectory`, `ANTIGRAVITY_PROXY_ENV_FILE`, and `ExecStart` paths to
match your install directory before copying it into systemd.

Example systemd user unit:

```ini
[Unit]
Description=Antigravity Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/workspace/Antigravity-Proxy
Environment=ANTIGRAVITY_PROXY_ENV_FILE=%h/workspace/Antigravity-Proxy/.env
ExecStart=%h/workspace/Antigravity-Proxy/.venv/bin/python %h/workspace/Antigravity-Proxy/antigravity_proxy.py
Restart=always
RestartSec=3

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
python scripts/update_gemini_discovery_fixture.py --check
```

The test suite avoids real upstream calls for proxy behavior. The discovery
check fetches Google's Gemini v1beta discovery document and fails when the
committed route fixture is stale, so compatibility drift is visible before a
release.
