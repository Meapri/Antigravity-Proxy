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
`Authorization: Bearer <key>` or `X-API-Key: <key>`.

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

Implemented Gemini-compatible routes:

- `GET /v1beta/models`
- `GET /v1beta/models/{model}`
- `POST /v1beta/models/{model}:generateContent`
- `POST /v1beta/models/{model}:streamGenerateContent`
- `POST /v1beta/models/{model}:countTokens`
- `POST /v1beta/models/{model}:embedContent`
- `POST /v1beta/models/{model}:batchEmbedContents`
- `POST /v1beta/models/{model}:batchGenerateContent`
- `POST /upload/v1beta/files`
- `POST /v1beta/files`
- `GET /v1beta/files`
- `GET /v1beta/files/{file}`
- `DELETE /v1beta/files/{file}`
- `POST /v1beta/cachedContents`
- `GET /v1beta/cachedContents`
- `GET /v1beta/cachedContents/{cached_content}`
- `PATCH /v1beta/cachedContents/{cached_content}`
- `DELETE /v1beta/cachedContents/{cached_content}`
- `GET /v1beta/operations`
- `GET /v1beta/operations/{operation}`
- `POST /v1beta/operations/{operation}:cancel`
- `DELETE /v1beta/operations/{operation}`
- `POST /v1beta/fileSearchStores`
- `GET /v1beta/fileSearchStores`
- `GET /v1beta/fileSearchStores/{store}`
- `DELETE /v1beta/fileSearchStores/{store}`
- `POST /v1beta/fileSearchStores/{store}:importFile`
- `POST /upload/v1beta/fileSearchStores/{store}:uploadToFileSearchStore`
- `GET /v1beta/fileSearchStores/{store}/documents`
- `GET /v1beta/fileSearchStores/{store}/documents/{document}`
- `DELETE /v1beta/fileSearchStores/{store}/documents/{document}`

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
`urlContext`, and `codeExecution`.

Files API example:

```bash
curl "http://127.0.0.1:8765/upload/v1beta/files?uploadType=media&displayName=note.txt" \
  -H "Content-Type: text/plain" \
  --data-binary @note.txt
```

The upload endpoint also supports the Gemini resumable upload headers
(`X-Goog-Upload-Protocol: resumable`, `X-Goog-Upload-Command: start`, then
`upload, finalize`) used by SDK-style file uploads.

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
expose a native public Files API.

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

Embeddings and batch operations:

- `embedContent` and `batchEmbedContents` return deterministic local embedding
  vectors for Gemini SDK compatibility. They are stable and shaped like Gemini
  embeddings, but they are not semantic Google embedding model outputs because
  Antigravity does not expose a public embedding RPC.
- `batchGenerateContent` runs requests synchronously through Antigravity and
  stores an immediately completed `operations/*` result.

File search stores:

- `fileSearchStores` and document management are implemented as a local
  compatibility layer under `data/gemini_file_search_stores`.
- `importFile` imports files previously uploaded through the local Files API.
- `uploadToFileSearchStore` accepts direct uploads and stores documents locally.
- The proxy preserves Gemini-compatible resource shapes and operations, but it
  does not yet execute a semantic vector search over these stores inside
  `tools.file_search`.

Notes:

- Model names are exposed as Gemini resources like
  `models/gemini-3-flash-agent`; pass `gemini-3-flash-agent` in the path.
- `countTokens` is approximate because Antigravity's internal endpoint does not
  expose a separate Gemini token-count RPC.
- Files are stored locally under `data/gemini_files` by default; override with
  `ANTIGRAVITY_GEMINI_FILES_DIR`.
- Cached contents are stored locally under `data/gemini_cached_contents` by
  default; override with `ANTIGRAVITY_GEMINI_CACHED_CONTENTS_DIR`.
- Batch operations are stored locally under `data/gemini_operations` by default;
  override with `ANTIGRAVITY_GEMINI_OPERATIONS_DIR`.
- File search stores are stored locally under `data/gemini_file_search_stores`;
  override with `ANTIGRAVITY_GEMINI_FILE_SEARCH_STORES_DIR`.
- Tuned models, live API, interactions, permissions, true async long-running
  jobs, semantic Google embeddings, and semantic `tools.file_search` retrieval
  are not fully implemented yet.

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
