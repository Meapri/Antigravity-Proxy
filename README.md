# Antigravity Proxy

OpenAI-compatible FastAPI proxy for Antigravity / Cloud Code models.

It exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/images/generations`
- `GET /search` as a SearXNG-compatible Google-grounded search endpoint
- `POST /admin/models/refresh` for API-key-protected model refresh

## Safety

This repository intentionally does **not** include OAuth credentials, access tokens,
refresh tokens, `.env` files, model caches, or local runtime state. Keep those files
outside git.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` and point it at your local Antigravity OAuth files:

```bash
cp .env.example .env
```

Required files:

- `ANTIGRAVITY_AUTH_FILE`: JSON file containing `access`, `refresh`, `expires`, and optionally `project_id`
- `ANTIGRAVITY_CLIENT_FILE`: JSON file containing OAuth `client_id` and `client_secret`

Optional protection:

- Set `ANTIGRAVITY_PROXY_API_KEY` to require `Authorization: Bearer ...` or `X-API-Key` for all routes except `/health`.

## Run

```bash
python antigravity_proxy.py
```

The server listens on `0.0.0.0:8765` by default.

## Test

```bash
pytest -q
```

## Notes

This is an unofficial compatibility proxy. It is designed for trusted local or
private-network deployments. If you expose it beyond localhost or a private VPN,
set `ANTIGRAVITY_PROXY_API_KEY` and put it behind TLS.
