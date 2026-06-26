"""
Antigravity OpenAI-compatible Proxy Server

Wraps AntigravityClient behind OpenAI /v1/chat/completions and /v1/models
endpoints so that any OpenAI-compatible client (including Hermes Agent) can
use Antigravity models.

Usage:
    python antigravity_proxy.py
    # or: uvicorn antigravity_proxy:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import sys
from email import policy
from email.parser import BytesParser

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from antigravity_proxy_core.antigravity import AntigravityClient
    from antigravity_proxy_core.config import Settings
except ModuleNotFoundError:  # pragma: no cover - compatibility for the original monorepo layout.
    from rizi_kakao_agent.antigravity import AntigravityClient
    from rizi_kakao_agent.config import Settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("antigravity_proxy")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Models sourced from Antigravity API v1internal:fetchAvailableModels.
# "id" = human-readable display name shown in /v1/models;
# "antigravity_model" = real upstream model ID passed to the API.
_MODELS: list[dict[str, Any]] = [
    # ── Gemini 3.5 Flash ──
    {
        "id": "Gemini 3.5 Flash (Medium)",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        "antigravity_model": "gemini-3.5-flash-low",
    },
    {
        "id": "Gemini 3.5 Flash (High)",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        "antigravity_model": "gemini-3-flash-agent",
    },
    {
        "id": "Gemini 3.5 Flash (Low)",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        "antigravity_model": "gemini-3.5-flash-extra-low",
    },
    # ── Gemini 3.1 Pro ──
    {
        "id": "Gemini 3.1 Pro (Low)",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        "antigravity_model": "gemini-3.1-pro-low",
    },
    {
        "id": "Gemini 3.1 Pro (High)",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        "antigravity_model": "gemini-pro-agent",
    },
    # ── Gemini 3 Flash / Lite ──
    {
        "id": "Gemini 3 Flash",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        "antigravity_model": "gemini-3-flash",
    },
    {
        "id": "Gemini 3.1 Flash Lite",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        # The REAL lite tier (was mislabelled as plain gemini-2.5-flash = regular
        # Flash). Confirmed via v1internal:fetchAvailableModels.
        "antigravity_model": "gemini-3.1-flash-lite",
    },
    # ── Image generation ──
    {
        "id": "Gemini 3.1 Flash Image",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        "antigravity_model": "gemini-3.1-flash-image",
    },
    # ── Claude ──
    {
        "id": "Claude Opus 4.6 (Thinking)",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        "antigravity_model": "claude-opus-4-6-thinking",
    },
    {
        "id": "Claude Sonnet 4.6 (Thinking)",
        "object": "model",
        "created": 1737500000,
        "owned_by": "antigravity",
        "antigravity_model": "claude-sonnet-4-6",
    },
]

def _display_id_for_model(model: dict[str, Any], duplicate_names: set[str]) -> str:
    display_id = str(model.get("id") or model.get("antigravity_model") or "").strip()
    upstream_id = str(model.get("antigravity_model") or display_id).strip()
    if display_id in duplicate_names and upstream_id and display_id != upstream_id:
        return f"{display_id} [{upstream_id}]"
    return display_id


def _is_internal_model(model: dict[str, Any]) -> bool:
    display_id = str(model.get("id") or "").strip().lower()
    upstream_id = str(model.get("antigravity_model") or display_id).strip().lower()
    return display_id.startswith(("tab_", "chat_")) or upstream_id.startswith(("tab_", "chat_"))


def _include_internal_models() -> bool:
    return os.getenv("ANTIGRAVITY_PROXY_INCLUDE_INTERNAL_MODELS", "").strip().lower() in {"1", "true", "yes", "on"}


def _model_capabilities(model: dict[str, Any]) -> dict[str, bool]:
    display_id = str(model.get("id") or "").lower()
    upstream_id = str(model.get("antigravity_model") or "").lower()
    text = f"{display_id} {upstream_id}"
    internal = _is_internal_model(model)
    image_generation = "image" in text
    return {
        "chat": not image_generation and not internal,
        "tools": not image_generation and not internal,
        "vision": not image_generation and not internal,
        "streaming": not image_generation and not internal,
        "grounding": not image_generation and not internal,
        "image_generation": image_generation,
        "internal": internal,
    }


def _model_sort_key(model: dict[str, Any]) -> tuple[int, str]:
    caps = _model_capabilities(model)
    upstream_id = str(model.get("antigravity_model") or "")
    display_id = str(model.get("id") or upstream_id)
    if caps["internal"]:
        group = 90
    elif caps["image_generation"]:
        group = 40
    elif "claude" in upstream_id.lower() or "claude" in display_id.lower():
        group = 20
    elif "gemini" in upstream_id.lower() or "gemini" in display_id.lower():
        group = 10
    else:
        group = 30
    return group, display_id.lower()


def _normalize_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate exact upstream models and disambiguate repeated display names."""
    by_upstream: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in models:
        upstream_id = str(raw.get("antigravity_model") or raw.get("id") or "").strip()
        if not upstream_id or upstream_id in by_upstream:
            continue
        model = dict(raw)
        model["antigravity_model"] = upstream_id
        model["id"] = str(model.get("id") or upstream_id).strip()
        by_upstream[upstream_id] = model
        order.append(upstream_id)

    name_counts: dict[str, int] = {}
    for model in by_upstream.values():
        name = str(model.get("id") or "").strip()
        name_counts[name] = name_counts.get(name, 0) + 1
    duplicate_names = {name for name, count in name_counts.items() if count > 1}

    normalized: list[dict[str, Any]] = []
    for upstream_id in order:
        model = dict(by_upstream[upstream_id])
        model["id"] = _display_id_for_model(model, duplicate_names)
        model["capabilities"] = _model_capabilities(model)
        model["metadata"] = {
            "antigravity_model": model["antigravity_model"],
            "source": "antigravity",
            "internal": model["capabilities"]["internal"],
        }
        normalized.append(model)
    if not _include_internal_models():
        normalized = [model for model in normalized if not _is_internal_model(model)]
    return sorted(normalized, key=_model_sort_key)


def _rebuild_model_map(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    model_map: dict[str, dict[str, Any]] = {}
    for m in models:
        model_map[m["id"]] = m
        model_map[m["id"].lower()] = m
        model_map[m["antigravity_model"]] = m
        model_map[m["antigravity_model"].lower()] = m
    for alias, target in _ALIASES.items():
        if target in model_map:
            model_map[alias] = model_map[target]
            model_map[alias.lower()] = model_map[target]
    return model_map

# Map client aliases (antigravity.py) to survive direct backend requests
_ALIASES = {
    "gemini-3.5-flash": "gemini-3.5-flash-low",
    "gemini-3.5-flash-high": "gemini-3-flash-agent",
    "gemini-3.5-flash-medium": "gemini-3.5-flash-low",
    "gemini-3.5-flash-low": "gemini-3.5-flash-extra-low",
    "claude-opus-4.6": "claude-opus-4-6-thinking",
    "claude-opus-4-6": "claude-opus-4-6-thinking",
    "claude-4.6-opus": "claude-opus-4-6-thinking",
    "claude-4-6-opus": "claude-opus-4-6-thinking",
    "claude-opus-4.6-thinking": "claude-opus-4-6-thinking",
    "gemini-3.1-pro-high": "gemini-pro-agent",
    "gemini-3.1-pro": "gemini-3.1-pro-low",
}
_MODELS = _normalize_models(_MODELS)
_MODEL_MAP = _rebuild_model_map(_MODELS)


# ---------------------------------------------------------------------------
# Gemini-compatible helpers
# ---------------------------------------------------------------------------
def _gemini_model_id(model: dict[str, Any]) -> str:
    return str(model.get("antigravity_model") or model.get("id") or "").strip()


def _gemini_model_name(model: dict[str, Any]) -> str:
    return "models/" + _gemini_model_id(model)


def _gemini_model_resource(model: dict[str, Any]) -> dict[str, Any]:
    caps = _model_capabilities(model)
    methods = [
        "generateContent",
        "streamGenerateContent",
        "countTokens",
        "embedContent",
        "batchEmbedContents",
        "batchGenerateContent",
    ]
    return {
        "name": _gemini_model_name(model),
        "version": _gemini_model_id(model),
        "displayName": str(model.get("id") or _gemini_model_id(model)),
        "description": "Antigravity-backed Gemini-compatible model",
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65536,
        "supportedGenerationMethods": methods if not caps["internal"] else [],
        "temperature": 1.0,
        "topP": 0.95,
        "topK": 64,
    }


def _resolve_gemini_model(model_name: str) -> dict[str, Any]:
    key = model_name.strip().strip("/")
    if key.startswith("models/"):
        key = key[len("models/"):]
    decoded = key.replace("%20", " ")
    model = _MODEL_MAP.get(decoded) or _MODEL_MAP.get(decoded.lower())
    if not model:
        raise HTTPException(status_code=404, detail=f"Gemini model '{model_name}' not found.")
    return model


_GEMINI_KEY_ALIASES = {
    "system_instruction": "systemInstruction",
    "generation_config": "generationConfig",
    "safety_settings": "safetySettings",
    "tool_config": "toolConfig",
    "cached_content": "cachedContent",
    "response_mime_type": "responseMimeType",
    "response_schema": "responseSchema",
    "max_output_tokens": "maxOutputTokens",
    "candidate_count": "candidateCount",
    "stop_sequences": "stopSequences",
    "response_logprobs": "responseLogprobs",
    "logprobs": "logprobs",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
    "thinking_config": "thinkingConfig",
    "function_declarations": "functionDeclarations",
    "function_calling_config": "functionCallingConfig",
    "allowed_function_names": "allowedFunctionNames",
    "code_execution": "codeExecution",
    "codeExecution": "codeExecution",
    "google_search": "google_search",
    "googleSearch": "google_search",
    "url_context": "url_context",
    "urlContext": "url_context",
    "file_search": "file_search",
    "fileSearch": "file_search",
    "file_data": "fileData",
    "fileData": "fileData",
    "inline_data": "inlineData",
    "inlineData": "inlineData",
    "mime_type": "mimeType",
    "mimeType": "mimeType",
    "file_uri": "fileUri",
    "fileUri": "fileUri",
    "video_metadata": "videoMetadata",
    "function_call": "functionCall",
    "functionCall": "functionCall",
    "function_response": "functionResponse",
    "functionResponse": "functionResponse",
}


def _gemini_normalize_request(value: Any) -> Any:
    if isinstance(value, list):
        return [_gemini_normalize_request(item) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, child in value.items():
        mapped = _GEMINI_KEY_ALIASES.get(str(key), key)
        out[mapped] = _gemini_normalize_request(child)
    return out


def _gemini_unwrap_response(data: dict[str, Any]) -> dict[str, Any]:
    response = data.get("response")
    if isinstance(response, dict):
        return response
    return data


def _gemini_error_response(message: Any, *, status_code: int, status: str | None = None) -> JSONResponse:
    if not isinstance(message, str):
        message = json.dumps(message, ensure_ascii=False)
    return JSONResponse(
        {"error": {"code": status_code, "message": message, "status": status or _openai_error_type(status_code).upper()}},
        status_code=status_code,
    )


def _gemini_count_tokens_request(body: dict[str, Any]) -> list[ChatMessage]:
    payload = body.get("generateContentRequest") if isinstance(body.get("generateContentRequest"), dict) else body
    contents = payload.get("contents") or []
    if isinstance(contents, dict):
        contents = [contents]
    messages: list[ChatMessage] = []
    for turn in contents if isinstance(contents, list) else []:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user")
        if role == "model":
            role = "assistant"
        parts = turn.get("parts") or []
        messages.append(ChatMessage(role=role, content=parts))
    system_instruction = payload.get("systemInstruction")
    if isinstance(system_instruction, dict):
        messages.insert(0, ChatMessage(role="system", content=system_instruction.get("parts") or []))
    return messages


def _gemini_content_text(content: Any) -> str:
    if not isinstance(content, dict):
        return _msg_text(content)
    parts = content.get("parts") or []
    texts: list[str] = []
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict):
                if part.get("text") is not None:
                    texts.append(str(part["text"]))
                elif isinstance(part.get("inlineData"), dict):
                    texts.append(f"[inlineData:{part['inlineData'].get('mimeType', 'application/octet-stream')}]")
                elif isinstance(part.get("fileData"), dict):
                    texts.append(f"[fileData:{part['fileData'].get('fileUri') or part['fileData'].get('uri') or ''}]")
            elif isinstance(part, str):
                texts.append(part)
    return "\n".join(texts)


def _gemini_embedding_values(text: str, *, dimensions: int = 768) -> list[float]:
    dimensions = max(1, min(int(dimensions or 768), 3072))
    values: list[float] = []
    seed = text.encode("utf-8", errors="ignore")
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for i in range(0, len(digest), 4):
            raw = int.from_bytes(digest[i:i + 4], "big", signed=False)
            values.append((raw / 2147483647.5) - 1.0)
            if len(values) >= dimensions:
                break
        counter += 1
    # Unit-normalize for a stable embedding-like vector.
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


def _gemini_embedding_from_request(body: dict[str, Any]) -> dict[str, Any]:
    content = body.get("content") or {}
    output_dim = body.get("outputDimensionality") or body.get("output_dimensionality") or 768
    text = _gemini_content_text(content)
    return {"embedding": {"values": _gemini_embedding_values(text, dimensions=int(output_dim))}}


def _gemini_batch_embedding_from_request(body: dict[str, Any]) -> dict[str, Any]:
    requests = body.get("requests")
    if not isinstance(requests, list):
        raise HTTPException(status_code=400, detail="batchEmbedContents requires a requests array.")
    embeddings: list[dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="batchEmbedContents request items must be objects.")
        embeddings.append(_gemini_embedding_from_request(_gemini_normalize_request(item))["embedding"])
    return {"embeddings": embeddings}


def _gemini_operations_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", "data/gemini_operations")).expanduser()


def _gemini_operations_index_path() -> Path:
    return _gemini_operations_root() / "index.json"


def _gemini_load_operations_index() -> dict[str, dict[str, Any]]:
    path = _gemini_operations_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini operations index; starting empty.")
        return {}


def _gemini_save_operations_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_operations_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_operations_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_operation_name(name: str) -> str:
    key = name.strip().strip("/")
    if key.startswith("v1beta/"):
        key = key[len("v1beta/"):]
    if key.startswith("operations/"):
        return key
    return "operations/" + key


def _gemini_store_operation(operation: dict[str, Any]) -> dict[str, Any]:
    index = _gemini_load_operations_index()
    index[operation["name"]] = operation
    _gemini_save_operations_index(index)
    return operation


def _gemini_get_operation(name: str) -> dict[str, Any] | None:
    return _gemini_load_operations_index().get(_gemini_operation_name(name))


def _gemini_files_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_FILES_DIR", "data/gemini_files")).expanduser()


def _gemini_files_index_path() -> Path:
    return _gemini_files_root() / "index.json"


def _gemini_upload_sessions_path() -> Path:
    return _gemini_files_root() / "upload_sessions.json"


def _gemini_load_files_index() -> dict[str, dict[str, Any]]:
    path = _gemini_files_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini files index; starting empty.")
    return {}


def _gemini_load_upload_sessions() -> dict[str, dict[str, Any]]:
    path = _gemini_upload_sessions_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _gemini_save_upload_sessions(sessions: dict[str, dict[str, Any]]) -> None:
    root = _gemini_files_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_upload_sessions_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_save_files_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_files_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_files_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_file_resource(meta: dict[str, Any]) -> dict[str, Any]:
    now = int(meta.get("createTime") or time.time())
    return {
        "name": meta["name"],
        "displayName": meta.get("displayName") or meta["name"].split("/", 1)[-1],
        "mimeType": meta.get("mimeType") or "application/octet-stream",
        "sizeBytes": str(int(meta.get("sizeBytes") or 0)),
        "createTime": meta.get("createTimeIso") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "updateTime": meta.get("updateTimeIso") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "expirationTime": meta.get("expirationTimeIso"),
        "sha256Hash": meta.get("sha256Hash") or "",
        "uri": meta.get("uri") or meta["name"],
        "state": meta.get("state") or "ACTIVE",
    }


def _gemini_store_file(data: bytes, *, mime_type: str | None = None, display_name: str | None = None) -> dict[str, Any]:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    root = _gemini_files_root()
    root.mkdir(parents=True, exist_ok=True)
    file_id = "file_" + uuid.uuid4().hex
    inferred = mimetypes.guess_type(display_name or "")[0] if display_name else None
    mime = (mime_type or inferred or "application/octet-stream").split(";", 1)[0].strip()
    blob_path = root / f"{file_id}.bin"
    blob_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    now = int(time.time())
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    meta = {
        "name": f"files/{file_id}",
        "displayName": display_name or file_id,
        "mimeType": mime,
        "sizeBytes": len(data),
        "createTime": now,
        "updateTime": now,
        "createTimeIso": iso,
        "updateTimeIso": iso,
        "sha256Hash": digest,
        "uri": f"files/{file_id}",
        "state": "ACTIVE",
        "path": str(blob_path),
    }
    index = _gemini_load_files_index()
    index[meta["name"]] = meta
    _gemini_save_files_index(index)
    return _gemini_file_resource(meta)


def _gemini_get_file_meta(file_name: str) -> dict[str, Any] | None:
    key = file_name.strip().strip("/")
    if key.startswith("v1beta/"):
        key = key[len("v1beta/"):]
    if key.startswith("files/"):
        name = key
    elif "/files/" in key:
        name = "files/" + key.rsplit("/files/", 1)[-1]
    else:
        name = "files/" + key
    return _gemini_load_files_index().get(name)


def _gemini_file_uri_to_inline(file_uri: str) -> dict[str, Any] | None:
    meta = _gemini_get_file_meta(file_uri)
    if not meta:
        return None
    path = Path(str(meta.get("path") or ""))
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inlineData": {"mimeType": meta.get("mimeType") or "application/octet-stream", "data": data}}


def _gemini_inline_local_files(value: Any) -> Any:
    if isinstance(value, list):
        return [_gemini_inline_local_files(item) for item in value]
    if not isinstance(value, dict):
        return value
    file_data = value.get("fileData")
    if isinstance(file_data, dict):
        uri = str(file_data.get("fileUri") or file_data.get("uri") or "")
        inline = _gemini_file_uri_to_inline(uri)
        if inline:
            return inline
    return {key: _gemini_inline_local_files(child) for key, child in value.items()}


def _parse_gemini_multipart_upload(content_type: str, body: bytes) -> tuple[dict[str, Any], bytes, str | None]:
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    metadata: dict[str, Any] = {}
    media = b""
    media_type: str | None = None
    for part in message.iter_parts():
        payload = part.get_payload(decode=True) or b""
        ptype = part.get_content_type()
        if ptype == "application/json" and not metadata:
            try:
                decoded = json.loads(payload.decode(part.get_content_charset() or "utf-8"))
                metadata = decoded if isinstance(decoded, dict) else {}
            except Exception:
                metadata = {}
        elif payload and not media:
            media = payload
            media_type = ptype
    return metadata, media, media_type


async def _gemini_upload_file_from_request(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    upload_type = request.query_params.get("uploadType", "").lower()
    display_name = request.query_params.get("displayName") or request.headers.get("x-goog-upload-file-name")
    mime_type = request.headers.get("x-goog-upload-header-content-type")
    metadata: dict[str, Any] = {}
    media = body
    if "multipart/" in content_type or upload_type == "multipart":
        metadata, media, media_type = _parse_gemini_multipart_upload(content_type, body)
        mime_type = mime_type or media_type
    elif upload_type not in {"", "media"} and request.headers.get("x-goog-upload-protocol", "").lower() != "raw":
        raise HTTPException(status_code=400, detail=f"Unsupported Gemini uploadType: {upload_type}")
    file_meta = metadata.get("file") if isinstance(metadata.get("file"), dict) else metadata
    if isinstance(file_meta, dict):
        display_name = file_meta.get("displayName") or file_meta.get("display_name") or display_name
        mime_type = file_meta.get("mimeType") or file_meta.get("mime_type") or mime_type
    return _gemini_store_file(media, mime_type=mime_type or content_type, display_name=display_name)


async def _gemini_start_resumable_upload(request: Request) -> JSONResponse:
    body = await request.body()
    metadata: dict[str, Any] = {}
    if body:
        try:
            decoded = json.loads(body.decode("utf-8"))
            metadata = decoded if isinstance(decoded, dict) else {}
        except Exception:
            metadata = {}
    file_meta = metadata.get("file") if isinstance(metadata.get("file"), dict) else metadata
    session_id = "upload_" + uuid.uuid4().hex
    sessions = _gemini_load_upload_sessions()
    sessions[session_id] = {
        "displayName": file_meta.get("displayName") or file_meta.get("display_name") or request.query_params.get("displayName"),
        "mimeType": (
            request.headers.get("x-goog-upload-header-content-type")
            or file_meta.get("mimeType")
            or file_meta.get("mime_type")
            or "application/octet-stream"
        ),
        "created": int(time.time()),
    }
    _gemini_save_upload_sessions(sessions)
    upload_url = str(request.base_url).rstrip("/") + f"/upload/v1beta/files/{session_id}"
    return JSONResponse(
        {},
        headers={
            "X-Goog-Upload-URL": upload_url,
            "X-Goog-Upload-Status": "active",
        },
    )


async def _gemini_finish_resumable_upload(session_id: str, request: Request) -> dict[str, Any]:
    sessions = _gemini_load_upload_sessions()
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Upload session '{session_id}' not found.")
    data = await request.body()
    file_resource = _gemini_store_file(
        data,
        mime_type=session.get("mimeType") or request.headers.get("content-type"),
        display_name=session.get("displayName"),
    )
    sessions.pop(session_id, None)
    _gemini_save_upload_sessions(sessions)
    return file_resource


def _gemini_cached_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_CACHED_CONTENTS_DIR", "data/gemini_cached_contents")).expanduser()


def _gemini_cached_index_path() -> Path:
    return _gemini_cached_root() / "index.json"


def _gemini_load_cached_index() -> dict[str, dict[str, Any]]:
    path = _gemini_cached_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini cachedContents index; starting empty.")
        return {}


def _gemini_save_cached_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_cached_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_cached_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_cached_name(name: str) -> str:
    key = name.strip().strip("/")
    if key.startswith("v1beta/"):
        key = key[len("v1beta/"):]
    if key.startswith("cachedContents/"):
        return key
    return "cachedContents/" + key


def _gemini_cached_resource(meta: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in meta.items() if k not in {"payload"} and v is not None}
    return out


def _gemini_create_cached_content(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_normalize_request(body)
    now = int(time.time())
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    cache_id = "cache_" + uuid.uuid4().hex
    ttl = body.get("ttl")
    expire_seconds = 3600
    if isinstance(ttl, str) and ttl.endswith("s"):
        try:
            expire_seconds = max(1, int(float(ttl[:-1])))
        except ValueError:
            expire_seconds = 3600
    expire_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + expire_seconds))
    meta = {
        "name": f"cachedContents/{cache_id}",
        "model": body.get("model"),
        "displayName": body.get("displayName") or body.get("display_name"),
        "createTime": iso,
        "updateTime": iso,
        "expireTime": body.get("expireTime") or expire_iso,
        "usageMetadata": {"totalTokenCount": _estimate_tokens(body)},
        "payload": body,
    }
    index = _gemini_load_cached_index()
    index[meta["name"]] = meta
    _gemini_save_cached_index(index)
    return _gemini_cached_resource(meta)


def _gemini_get_cached_meta(name: str) -> dict[str, Any] | None:
    return _gemini_load_cached_index().get(_gemini_cached_name(name))


def _gemini_apply_cached_content(body: dict[str, Any]) -> dict[str, Any]:
    cached_name = body.get("cachedContent")
    if not cached_name:
        return body
    meta = _gemini_get_cached_meta(str(cached_name))
    if not meta:
        raise HTTPException(status_code=404, detail=f"Cached content '{cached_name}' not found.")
    payload = meta.get("payload") if isinstance(meta.get("payload"), dict) else {}
    merged = dict(body)
    cached_contents = payload.get("contents") if isinstance(payload.get("contents"), list) else []
    current_contents = merged.get("contents") if isinstance(merged.get("contents"), list) else []
    if cached_contents:
        merged["contents"] = cached_contents + current_contents
    for key in ("systemInstruction", "tools", "toolConfig"):
        if key not in merged and key in payload:
            merged[key] = payload[key]
    merged.pop("cachedContent", None)
    return merged


def _gemini_fss_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_FILE_SEARCH_STORES_DIR", "data/gemini_file_search_stores")).expanduser()


def _gemini_fss_index_path() -> Path:
    return _gemini_fss_root() / "index.json"


def _gemini_load_fss_index() -> dict[str, dict[str, Any]]:
    path = _gemini_fss_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini fileSearchStores index; starting empty.")
        return {}


def _gemini_save_fss_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_fss_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_fss_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_fss_name(name: str) -> str:
    key = name.strip().strip("/")
    if key.startswith("v1beta/"):
        key = key[len("v1beta/"):]
    if key.startswith("fileSearchStores/"):
        return key
    return "fileSearchStores/" + key


def _gemini_document_name(store_name: str, document_id: str) -> str:
    doc = document_id.strip().strip("/")
    if "/documents/" in doc:
        doc = doc.rsplit("/documents/", 1)[-1]
    return f"{_gemini_fss_name(store_name)}/documents/{doc}"


def _gemini_fss_resource(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": meta["name"],
        "displayName": meta.get("displayName") or meta["name"].split("/", 1)[-1],
        "createTime": meta.get("createTime"),
        "updateTime": meta.get("updateTime"),
        "fileCount": len(meta.get("documents") or {}),
    }


def _gemini_document_resource(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k not in {"content"} and v is not None}


def _gemini_create_fss(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_normalize_request(body)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    store_id = "fs_" + uuid.uuid4().hex
    meta = {
        "name": f"fileSearchStores/{store_id}",
        "displayName": body.get("displayName") or body.get("display_name") or store_id,
        "createTime": now,
        "updateTime": now,
        "documents": {},
    }
    index = _gemini_load_fss_index()
    index[meta["name"]] = meta
    _gemini_save_fss_index(index)
    return _gemini_fss_resource(meta)


def _gemini_get_fss_meta(store_name: str) -> dict[str, Any] | None:
    return _gemini_load_fss_index().get(_gemini_fss_name(store_name))


def _gemini_store_document(store_name: str, *, display_name: str | None, mime_type: str | None, content: bytes) -> dict[str, Any]:
    index = _gemini_load_fss_index()
    store_key = _gemini_fss_name(store_name)
    store = index.get(store_key)
    if not store:
        raise HTTPException(status_code=404, detail=f"File search store '{store_name}' not found.")
    doc_id = "doc_" + uuid.uuid4().hex
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text_preview = content[:4096].decode("utf-8", errors="replace")
    doc = {
        "name": f"{store_key}/documents/{doc_id}",
        "displayName": display_name or doc_id,
        "mimeType": (mime_type or "application/octet-stream").split(";", 1)[0],
        "createTime": now,
        "updateTime": now,
        "state": "ACTIVE",
        "sizeBytes": str(len(content)),
        "sha256Hash": hashlib.sha256(content).hexdigest(),
        "content": base64.b64encode(content).decode("ascii"),
        "textPreview": text_preview,
    }
    store.setdefault("documents", {})[doc["name"]] = doc
    store["updateTime"] = now
    index[store_key] = store
    _gemini_save_fss_index(index)
    return _gemini_document_resource(doc)


def _gemini_import_file_to_fss(store_name: str, body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_normalize_request(body)
    file_name = str(body.get("fileName") or body.get("file") or body.get("fileUri") or "")
    meta = _gemini_get_file_meta(file_name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found.")
    path = Path(str(meta.get("path") or ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File blob for '{file_name}' not found.")
    return _gemini_store_document(
        store_name,
        display_name=meta.get("displayName"),
        mime_type=meta.get("mimeType"),
        content=path.read_bytes(),
    )


def _gemini_extract_search_query(body: dict[str, Any]) -> str:
    contents = body.get("contents") if isinstance(body.get("contents"), list) else []
    for turn in reversed(contents):
        if isinstance(turn, dict):
            text = _gemini_content_text(turn)
            if text.strip():
                return text.strip()
    return ""


def _gemini_file_search_store_names(tool: dict[str, Any]) -> list[str]:
    search_cfg = (
        tool.get("file_search")
        or tool.get("fileSearch")
        or tool.get("fileSearchRetrieval")
        or tool.get("file_search_retrieval")
        or {}
    )
    names: list[str] = []
    if isinstance(search_cfg, dict):
        for key in ("fileSearchStoreNames", "file_search_store_names", "fileSearchStores", "file_search_stores"):
            raw = search_cfg.get(key)
            if isinstance(raw, str):
                names.append(raw)
            elif isinstance(raw, list):
                names.extend(str(item) for item in raw if item)
        for key in ("fileSearchStoreName", "file_search_store_name", "fileSearchStore", "file_search_store"):
            raw = search_cfg.get(key)
            if raw:
                names.append(str(raw))
    return names


def _gemini_search_score(query: str, text: str) -> int:
    import re

    query_terms = {term.lower() for term in re.findall(r"[\w가-힣]+", query) if len(term) > 1}
    text_l = text.lower()
    return sum(1 for term in query_terms if term in text_l)


def _gemini_file_search_context(body: dict[str, Any]) -> str:
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    store_names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if any(key in tool for key in ("file_search", "fileSearch", "fileSearchRetrieval", "file_search_retrieval")):
            store_names.extend(_gemini_file_search_store_names(tool))
    if not store_names:
        return ""

    index = _gemini_load_fss_index()
    query = _gemini_extract_search_query(body)
    matches: list[tuple[int, str, dict[str, Any]]] = []
    for store_name in store_names:
        store = index.get(_gemini_fss_name(store_name))
        if not store:
            continue
        for doc in (store.get("documents") or {}).values():
            if not isinstance(doc, dict):
                continue
            text = str(doc.get("textPreview") or "")
            score = _gemini_search_score(query, text)
            if score or not query:
                matches.append((score, str(doc.get("name") or ""), doc))
    matches.sort(key=lambda item: (-item[0], item[1]))
    selected = matches[:5]
    if not selected:
        return ""
    blocks = []
    for rank, (_score, _name, doc) in enumerate(selected, start=1):
        title = doc.get("displayName") or doc.get("name")
        snippet = str(doc.get("textPreview") or "")[:3000]
        blocks.append(f"[{rank}] {title}\n{snippet}")
    return "Local Gemini file_search results:\n\n" + "\n\n".join(blocks)


def _gemini_apply_file_search(body: dict[str, Any]) -> dict[str, Any]:
    context = _gemini_file_search_context(body)
    if not context:
        return body
    merged = dict(body)
    contents = merged.get("contents") if isinstance(merged.get("contents"), list) else []
    merged["contents"] = [{"role": "user", "parts": [{"text": context}]}] + contents
    tools = merged.get("tools") if isinstance(merged.get("tools"), list) else []
    remaining_tools = [
        tool for tool in tools
        if not (
            isinstance(tool, dict)
            and any(key in tool for key in ("file_search", "fileSearch", "fileSearchRetrieval", "file_search_retrieval"))
        )
    ]
    if remaining_tools:
        merged["tools"] = remaining_tools
    else:
        merged.pop("tools", None)
    return merged


def _gemini_tuned_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_TUNED_MODELS_DIR", "data/gemini_tuned_models")).expanduser()


def _gemini_tuned_index_path() -> Path:
    return _gemini_tuned_root() / "index.json"


def _gemini_load_tuned_index() -> dict[str, dict[str, Any]]:
    path = _gemini_tuned_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini tunedModels index; starting empty.")
        return {}


def _gemini_save_tuned_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_tuned_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_tuned_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_tuned_name(name: str) -> str:
    key = name.strip().strip("/")
    if key.startswith("v1beta/"):
        key = key[len("v1beta/"):]
    if key.startswith("tunedModels/"):
        return key
    return "tunedModels/" + key


def _gemini_tuned_resource(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": meta["name"],
        "displayName": meta.get("displayName") or meta["name"].split("/", 1)[-1],
        "description": meta.get("description") or "",
        "baseModel": meta.get("baseModel") or "models/gemini-3-flash-agent",
        "state": meta.get("state") or "ACTIVE",
        "createTime": meta.get("createTime"),
        "updateTime": meta.get("updateTime"),
        "temperature": meta.get("temperature"),
        "topP": meta.get("topP"),
        "topK": meta.get("topK"),
    }


def _gemini_create_tuned_model(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_normalize_request(body)
    tuned_model = body.get("tunedModel") if isinstance(body.get("tunedModel"), dict) else body
    source_id = str(body.get("tunedModelId") or body.get("tuned_model_id") or "").strip()
    model_id = source_id or ("tuned_" + uuid.uuid4().hex)
    name = _gemini_tuned_name(model_id)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = {
        "name": name,
        "displayName": tuned_model.get("displayName") or tuned_model.get("display_name") or model_id,
        "description": tuned_model.get("description") or "",
        "baseModel": tuned_model.get("baseModel") or tuned_model.get("base_model") or body.get("baseModel") or "models/gemini-3-flash-agent",
        "state": "ACTIVE",
        "createTime": now,
        "updateTime": now,
        "temperature": tuned_model.get("temperature"),
        "topP": tuned_model.get("topP"),
        "topK": tuned_model.get("topK"),
        "permissions": {},
    }
    index = _gemini_load_tuned_index()
    index[name] = meta
    _gemini_save_tuned_index(index)
    return _gemini_tuned_resource(meta)


def _gemini_get_tuned_meta(name: str) -> dict[str, Any] | None:
    return _gemini_load_tuned_index().get(_gemini_tuned_name(name))


def _gemini_tuned_base_model(name: str) -> dict[str, Any]:
    meta = _gemini_get_tuned_meta(name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Tuned model '{name}' not found.")
    base = str(meta.get("baseModel") or "models/gemini-3-flash-agent")
    return _resolve_gemini_model(base)


def _gemini_permission_name(parent: str, permission_id: str) -> str:
    return f"{_gemini_tuned_name(parent)}/permissions/{permission_id.strip().strip('/')}"


def _gemini_permission_resource(parent: str, perm: dict[str, Any]) -> dict[str, Any]:
    parent_name = _gemini_tuned_name(parent)
    pid = str(perm.get("id") or perm.get("name", "").rsplit("/", 1)[-1] or ("perm_" + uuid.uuid4().hex))
    return {
        "name": f"{parent_name}/permissions/{pid}",
        "granteeType": perm.get("granteeType") or perm.get("grantee_type") or "USER",
        "emailAddress": perm.get("emailAddress") or perm.get("email_address"),
        "role": perm.get("role") or "READER",
    }


def _gemini_store_permission(parent: str, body: dict[str, Any]) -> dict[str, Any]:
    index = _gemini_load_tuned_index()
    parent_name = _gemini_tuned_name(parent)
    meta = index.get(parent_name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Tuned model '{parent}' not found.")
    permission_id = "perm_" + uuid.uuid4().hex
    perm = _gemini_permission_resource(parent_name, {"id": permission_id, **_gemini_normalize_request(body)})
    meta.setdefault("permissions", {})[perm["name"]] = perm
    meta["updateTime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    index[parent_name] = meta
    _gemini_save_tuned_index(index)
    return perm


# ---------------------------------------------------------------------------
# OpenAI-compatible Pydantic schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    # content may be a plain string, a list of content parts (multimodal), or
    # None (e.g. an assistant message that only carries tool_calls).
    content: Any | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str = "gemini-3-flash"
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=1048576)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    stream: bool = False
    user: str | None = None
    # Function/tool calling (OpenAI format). When present we forward them to
    # Antigravity as Gemini functionDeclarations and translate functionCall
    # responses back into OpenAI tool_calls.
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: Any | None = None
    reasoning: Any | None = None

    model_config = {"extra": "allow"}


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatUsage = Field(default_factory=ChatUsage)


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[dict[str, Any]]


class ResponseCreateRequest(BaseModel):
    model: str = "gemini-3-flash"
    input: Any
    instructions: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    store: bool = True
    background: bool = False
    metadata: dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=1048576)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    text: Any | None = None
    reasoning: Any | None = None
    parallel_tool_calls: bool | None = True
    truncation: str | None = "disabled"
    user: str | None = None

    model_config = {"extra": "allow"}


class ResponseInputTokensRequest(BaseModel):
    model: str | None = None
    input: Any
    instructions: str | None = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_settings: Settings | None = None
_client: AntigravityClient | None = None
_LAST_MODEL_REFRESH_TS: float = 0.0
_LAST_MODEL_REFRESH_OK: bool = False
_LAST_MODEL_REFRESH_ERROR: str = ""


def _responses_db_path() -> Path:
    return Path(os.getenv("ANTIGRAVITY_RESPONSES_DB", "data/responses.sqlite3")).expanduser()


def _responses_db() -> sqlite3.Connection:
    path = _responses_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            status TEXT NOT NULL,
            model TEXT NOT NULL,
            response_json TEXT NOT NULL,
            input_items_json TEXT NOT NULL,
            output_items_json TEXT NOT NULL,
            usage_json TEXT NOT NULL,
            previous_response_id TEXT,
            metadata_json TEXT NOT NULL,
            deleted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


def _responses_store_save(response: dict[str, Any], input_items: list[dict[str, Any]]) -> None:
    now = int(time.time())
    with _responses_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO responses
                (id, created_at, updated_at, status, model, response_json, input_items_json,
                 output_items_json, usage_json, previous_response_id, metadata_json, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                response["id"],
                int(response.get("created_at") or now),
                now,
                str(response.get("status") or "completed"),
                str(response.get("model") or ""),
                json.dumps(response, ensure_ascii=False),
                json.dumps(input_items, ensure_ascii=False),
                json.dumps(response.get("output") or [], ensure_ascii=False),
                json.dumps(response.get("usage") or {}, ensure_ascii=False),
                response.get("previous_response_id"),
                json.dumps(response.get("metadata") or {}, ensure_ascii=False),
            ),
        )


def _responses_store_get(response_id: str) -> dict[str, Any] | None:
    with _responses_db() as conn:
        row = conn.execute(
            "SELECT response_json FROM responses WHERE id = ? AND deleted = 0",
            (response_id,),
        ).fetchone()
    if not row:
        return None
    return json.loads(row["response_json"])


def _responses_store_get_input_items(response_id: str) -> list[dict[str, Any]] | None:
    with _responses_db() as conn:
        row = conn.execute(
            "SELECT input_items_json FROM responses WHERE id = ? AND deleted = 0",
            (response_id,),
        ).fetchone()
    if not row:
        return None
    return json.loads(row["input_items_json"])


def _responses_store_delete(response_id: str) -> bool:
    with _responses_db() as conn:
        cur = conn.execute(
            "UPDATE responses SET deleted = 1, updated_at = ? WHERE id = ? AND deleted = 0",
            (int(time.time()), response_id),
        )
        return cur.rowcount > 0


def _responses_store_update_status(response_id: str, status: str) -> dict[str, Any] | None:
    response = _responses_store_get(response_id)
    if not response:
        return None
    response["status"] = status
    if status == "cancelled":
        response["error"] = {"message": "Response cancelled.", "type": "cancelled", "code": "cancelled"}
    _responses_store_save(response, _responses_store_get_input_items(response_id) or [])
    return response


def _new_response_id() -> str:
    return "resp_" + uuid.uuid4().hex


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        log.info("Settings loaded (model=%s)", _settings.model)
    return _settings


def _get_client() -> AntigravityClient:
    global _client
    if _client is None:
        _client = AntigravityClient(settings=_get_settings())
        log.info("AntigravityClient initialized")
    return _client


async def _fetch_and_update_models() -> dict[str, Any]:
    log.info("Fetching available models from Antigravity upstream...")
    global _MODELS, _MODEL_MAP, _LAST_MODEL_REFRESH_TS, _LAST_MODEL_REFRESH_OK, _LAST_MODEL_REFRESH_ERROR
    try:
        client = _get_client()
        creds = await asyncio.to_thread(client._valid_credentials)

        url = "https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels"
        headers = {
            "Authorization": f"Bearer {creds.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "Antigravity/2.0.1 Chrome/138.0.7204.235 Electron/37.3.1 Safari/537.36",
            "X-Goog-Api-Client": "antigravity-cli/2.0.1",
        }

        def _do_fetch():
            import httpx as _httpx
            _retryable = {429, 500, 502, 503, 504}
            _delay = 1.0
            for _attempt in range(3):
                try:
                    resp = client._http.post(url, json={}, headers=headers, timeout=10.0)
                    resp.raise_for_status()
                    return resp.json()
                except _httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in _retryable or _attempt == 2:
                        raise
                    log.warning("fetchAvailableModels HTTP %d — retry %d/3 in %.1fs", exc.response.status_code, _attempt + 1, _delay)
                    time.sleep(_delay)
                    _delay = min(_delay * 2, 8.0)

        data = await asyncio.to_thread(_do_fetch)
        models_dict = data.get("models", {})
        if not models_dict:
            log.warning("No models returned from fetchAvailableModels.")
            _LAST_MODEL_REFRESH_TS = time.time()
            _LAST_MODEL_REFRESH_OK = False
            _LAST_MODEL_REFRESH_ERROR = "No models returned from fetchAvailableModels."
            return {
                "ok": False,
                "updated_count": 0,
                "model_count": len(_MODELS),
                "error": _LAST_MODEL_REFRESH_ERROR,
            }
            
        new_models = list(_MODELS)
        existing_antigravity_models = {m["antigravity_model"] for m in new_models}
        
        updated_count = 0
        for model_id, info in models_dict.items():
            if model_id not in existing_antigravity_models:
                new_models.append({
                    "id": info.get("displayName") or model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "antigravity",
                    "antigravity_model": model_id,
                })
                updated_count += 1
                
        if updated_count > 0:
            log.info("Discovered %d new models dynamically.", updated_count)
            _MODELS = _normalize_models(new_models)
            _MODEL_MAP = _rebuild_model_map(_MODELS)
            log.info("Model map rebuilt with total %d entries (including aliases).", len(_MODEL_MAP))
        _LAST_MODEL_REFRESH_TS = time.time()
        _LAST_MODEL_REFRESH_OK = True
        _LAST_MODEL_REFRESH_ERROR = ""
        return {
            "ok": True,
            "updated_count": updated_count,
            "model_count": len(_MODELS),
            "map_entries": len(_MODEL_MAP),
            "last_refresh_timestamp": _LAST_MODEL_REFRESH_TS,
        }
    except Exception as e:
        log.warning("Failed to fetch available models dynamically, using hardcoded fallback: %s", e)
        _LAST_MODEL_REFRESH_TS = time.time()
        _LAST_MODEL_REFRESH_OK = False
        _LAST_MODEL_REFRESH_ERROR = str(e)
        return {
            "ok": False,
            "updated_count": 0,
            "model_count": len(_MODELS),
            "error": _LAST_MODEL_REFRESH_ERROR,
            "last_refresh_timestamp": _LAST_MODEL_REFRESH_TS,
        }


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown — pre-warm credentials."""
    log.info("Starting Antigravity Proxy on 127.0.0.1:8765")
    if not sys.stdin.isatty():
        log.info(
            "Non-TTY (background/systemd) environment detected — "
            "interactive OAuth refresh is disabled. "
            "Run 'agy auth' from a terminal if credentials expire."
        )
    try:
        _get_client()  # triggers credential load
        await _fetch_and_update_models()
    except Exception as exc:
        log.warning("Pre-warm failed (will retry on first request): %s", exc)
    yield
    log.info("Antigravity Proxy shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Antigravity OpenAI Proxy",
    version="1.0.0",
    lifespan=lifespan,
)


def _proxy_api_key() -> str:
    return os.getenv("ANTIGRAVITY_PROXY_API_KEY", "").strip()


def _request_api_key_valid(request: Request) -> bool:
    expected = _proxy_api_key()
    if not expected:
        return False
    auth = request.headers.get("authorization", "").strip()
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    api_key = request.headers.get("x-api-key", "").strip()
    return secrets.compare_digest(bearer, expected) or secrets.compare_digest(api_key, expected)


def _openai_error_type(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "api_error"
    return "invalid_request_error"


def _openai_error_response(
    message: Any,
    *,
    status_code: int,
    error_type: str | None = None,
    code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    if not isinstance(message, str):
        message = json.dumps(message, ensure_ascii=False)
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type or _openai_error_type(status_code),
                "param": param,
                "code": code,
            }
        },
        status_code=status_code,
    )


def _responses_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                ptype = part.get("type")
                if ptype in {"input_text", "output_text", "text"} and part.get("text") is not None:
                    texts.append(str(part.get("text")))
        return "\n".join(t for t in texts if t)
    return str(content)


def _responses_content_to_chat(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _responses_message_text(content)
    out: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            out.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in {"input_text", "text"}:
            out.append({"type": "text", "text": str(part.get("text") or "")})
        elif ptype == "input_image":
            image_url = part.get("image_url") or part.get("url")
            if not image_url:
                raise HTTPException(status_code=400, detail="input_image requires image_url or url.")
            out.append({"type": "image_url", "image_url": {"url": image_url}})
        elif ptype in {"input_file", "file", "input_audio", "audio"}:
            raise HTTPException(status_code=400, detail=f"Unsupported Responses input content type: {ptype}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported Responses input content type: {ptype}")
    return out


def _normalize_responses_input(input_value: Any, *, instructions: str | None = None) -> tuple[list[ChatMessage], list[dict[str, Any]]]:
    messages: list[ChatMessage] = []
    input_items: list[dict[str, Any]] = []
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))
        input_items.append({
            "id": "item_" + uuid.uuid4().hex,
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": instructions}],
        })

    if isinstance(input_value, str):
        messages.append(ChatMessage(role="user", content=input_value))
        input_items.append({
            "id": "item_" + uuid.uuid4().hex,
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": input_value}],
        })
        return messages, input_items

    if not isinstance(input_value, list):
        raise HTTPException(status_code=400, detail="Responses input must be a string or list of input items.")

    for item in input_value:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Responses input list items must be objects.")
        item_type = item.get("type", "message")
        if item_type == "message":
            role = str(item.get("role") or "user")
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant"}:
                raise HTTPException(status_code=400, detail=f"Unsupported Responses message role: {role}")
            content = item.get("content", "")
            chat_content = _responses_content_to_chat(content)
            messages.append(ChatMessage(role=role, content=chat_content))
            input_items.append({
                "id": str(item.get("id") or ("item_" + uuid.uuid4().hex)),
                "type": "message",
                "role": role,
                "content": content if isinstance(content, list) else [{"type": "input_text", "text": str(content)}],
            })
        elif item_type == "function_call_output":
            call_id = str(item.get("call_id") or item.get("id") or "")
            output = item.get("output", "")
            messages.append(ChatMessage(role="tool", content=str(output), tool_call_id=call_id))
            input_items.append({
                "id": str(item.get("id") or ("item_" + uuid.uuid4().hex)),
                "type": "function_call_output",
                "call_id": call_id,
                "output": str(output),
            })
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported Responses input item type: {item_type}")
    return messages, input_items


def _responses_previous_messages(previous_response_id: str | None) -> list[ChatMessage]:
    if not previous_response_id:
        return []
    previous = _responses_store_get(previous_response_id)
    if not previous:
        raise HTTPException(status_code=404, detail=f"Previous response '{previous_response_id}' not found.")
    messages: list[ChatMessage] = []
    previous_items = _responses_store_get_input_items(previous_response_id) or []
    for item in previous_items:
        if item.get("type") == "message":
            role = str(item.get("role") or "user")
            if role == "developer":
                role = "system"
            if role in {"system", "user", "assistant"}:
                messages.append(ChatMessage(role=role, content=_responses_content_to_chat(item.get("content", ""))))
        elif item.get("type") == "function_call_output":
            messages.append(ChatMessage(
                role="tool",
                content=str(item.get("output") or ""),
                tool_call_id=str(item.get("call_id") or item.get("id") or ""),
            ))
    for item in previous.get("output") or []:
        if item.get("type") == "message":
            messages.append(ChatMessage(role="assistant", content=_responses_message_text(item.get("content"))))
        elif item.get("type") == "function_call":
            messages.append(ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[{
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"},
                }],
            ))
    return messages


def _responses_validate_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise HTTPException(status_code=400, detail=f"Unsupported Responses tool type: {tool}")
        if _is_grounding_tool(tool):
            normalized.append(tool)
            continue
        if tool.get("type") != "function":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported Responses tool type: {tool.get('type') if isinstance(tool, dict) else tool}",
            )
        if isinstance(tool.get("function"), dict):
            if not tool["function"].get("name"):
                raise HTTPException(status_code=400, detail="Responses function tool requires a name.")
            normalized.append(tool)
        else:
            if not tool.get("name"):
                raise HTTPException(status_code=400, detail="Responses function tool requires a name.")
            normalized.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            })
    return normalized


def _responses_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "function", "function": {"name": tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")}}
    return tool_choice


def _chat_response_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, JSONResponse):
        return json.loads(result.body)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    raise RuntimeError("Unsupported chat completion response type.")


def _responses_usage(chat_usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": int(chat_usage.get("prompt_tokens") or chat_usage.get("input_tokens") or 0),
        "output_tokens": int(chat_usage.get("completion_tokens") or chat_usage.get("output_tokens") or 0),
        "total_tokens": int(chat_usage.get("total_tokens") or 0),
    }


def _responses_output_from_chat(chat: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    choice = (chat.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []
    output_texts: list[str] = []
    content = message.get("content")
    if content:
        output_texts.append(str(content))
        output.append({
            "id": "msg_" + uuid.uuid4().hex,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": str(content), "annotations": []}],
        })
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        output.append({
            "id": str(tc.get("id") or ("fc_" + uuid.uuid4().hex)),
            "type": "function_call",
            "status": "completed",
            "call_id": str(tc.get("id") or ("call_" + uuid.uuid4().hex)),
            "name": str(fn.get("name") or ""),
            "arguments": str(fn.get("arguments") or "{}"),
        })
    return output, "\n".join(output_texts)


def _responses_build_object(
    *,
    response_id: str,
    req: ResponseCreateRequest,
    chat: dict[str, Any],
    input_items: list[dict[str, Any]],
    status: str = "completed",
) -> dict[str, Any]:
    output, output_text = _responses_output_from_chat(chat)
    created = int(time.time())
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "background": bool(req.background),
        "error": None,
        "incomplete_details": None,
        "instructions": req.instructions,
        "max_output_tokens": req.max_output_tokens,
        "model": req.model,
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": bool(req.parallel_tool_calls),
        "previous_response_id": req.previous_response_id,
        "reasoning": req.reasoning,
        "store": bool(req.store),
        "temperature": req.temperature,
        "text": req.text,
        "tool_choice": req.tool_choice,
        "tools": req.tools or [],
        "top_p": req.top_p,
        "truncation": req.truncation or "disabled",
        "usage": _responses_usage(chat.get("usage") or {}),
        "metadata": req.metadata or {},
        "input_items": input_items,
    }


async def _responses_generate(req: ResponseCreateRequest) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tools = _responses_validate_tools(req.tools)
    current_messages, input_items = _normalize_responses_input(req.input, instructions=req.instructions)
    messages = _responses_previous_messages(req.previous_response_id) + current_messages
    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_output_tokens,
        top_p=req.top_p,
        stream=False,
        user=req.user,
        tools=tools,
        tool_choice=_responses_tool_choice(req.tool_choice),
        text=req.text,
        reasoning=req.reasoning,
    )
    chat = _chat_response_to_dict(await chat_completions(chat_req))
    response = _responses_build_object(
        response_id=_new_response_id(),
        req=req,
        chat=chat,
        input_items=input_items,
    )
    return response, input_items


def _responses_sse(response: dict[str, Any]):
    def evt(name: str, payload: dict[str, Any]) -> str:
        data = {"type": name, **payload}
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def gen():
        yield evt("response.created", {"response": {**response, "status": "queued", "output": [], "output_text": ""}})
        yield evt("response.in_progress", {"response": {**response, "status": "in_progress", "output": [], "output_text": ""}})
        for index, item in enumerate(response.get("output") or []):
            yield evt("response.output_item.added", {"output_index": index, "item": item})
            if item.get("type") == "message":
                for content_index, part in enumerate(item.get("content") or []):
                    yield evt("response.content_part.added", {"output_index": index, "content_index": content_index, "part": part})
                    text = str(part.get("text") or "")
                    if text:
                        yield evt("response.output_text.delta", {"output_index": index, "content_index": content_index, "delta": text})
                        yield evt("response.output_text.done", {"output_index": index, "content_index": content_index, "text": text})
                    yield evt("response.content_part.done", {"output_index": index, "content_index": content_index, "part": part})
            elif item.get("type") == "function_call":
                args = str(item.get("arguments") or "")
                if args:
                    yield evt("response.function_call_arguments.delta", {"output_index": index, "item_id": item.get("id"), "delta": args})
                yield evt("response.function_call_arguments.done", {"output_index": index, "item_id": item.get("id"), "arguments": args})
            yield evt("response.output_item.done", {"output_index": index, "item": item})
        yield evt("response.completed", {"response": response})
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.middleware("http")
async def _optional_api_key_auth(request: Request, call_next):
    expected = _proxy_api_key()
    if not expected or request.url.path == "/health":
        return await call_next(request)
    if _request_api_key_valid(request):
        return await call_next(request)
    return _openai_error_response(
        "Invalid or missing API key.",
        status_code=401,
        error_type="authentication_error",
        code="invalid_api_key",
    )


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    return _openai_error_response(exc.detail, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    param = None
    if errors:
        loc = errors[0].get("loc") or []
        param = ".".join(str(part) for part in loc if part not in {"body", "query", "path"})
    return _openai_error_response(
        errors,
        status_code=422,
        error_type="invalid_request_error",
        code="validation_error",
        param=param,
    )


# ---------------------------------------------------------------------------
# Helpers: OpenAI ↔ Antigravity conversion
# ---------------------------------------------------------------------------
def _extract_system_prompt(messages: list[ChatMessage]) -> str:
    """Extract system message content from OpenAI messages."""
    for msg in messages:
        if msg.role == "system" and msg.content:
            return _msg_text(msg.content).strip()
    return ""


def _extract_user_prompt(messages: list[ChatMessage]) -> str:
    """Extract the last user message as the prompt."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return _msg_text(msg.content).strip()
    return ""


# ---------------------------------------------------------------------------
# Function calling: OpenAI tools  <->  Gemini functionDeclarations
# ---------------------------------------------------------------------------

# Gemini thinking models attach a `thoughtSignature` to functionCall parts that
# must be echoed back on the follow-up turn. The OpenAI round-trip can't carry
# it, so cache it keyed by the call's (name, args).
import threading

_SIGS_LOCK = threading.Lock()
_SIGS_PATH = os.path.join(os.path.dirname(__file__), "data", "thought_signatures.json")
_SIG_TTL = 86400.0  # 24 hours — entries older than this are garbage-collected


def _load_thought_sigs() -> dict[str, dict[str, Any]]:
    """Load thought signatures from disk, GC-ing entries older than _SIG_TTL.

    Stored format: {key: {"sig": str, "ts": float}}.
    Legacy format (plain string values) is migrated on load, stamped with
    the current time so they survive at least one more 24-hour window.
    """
    if not os.path.exists(_SIGS_PATH):
        return {}
    try:
        with open(_SIGS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        now = time.time()
        result: dict[str, dict[str, Any]] = {}
        for k, v in raw.items():
            if isinstance(v, str):
                # Migrate legacy plain-string value
                result[k] = {"sig": v, "ts": now}
            elif isinstance(v, dict):
                ts = float(v.get("ts") or 0)
                if now - ts < _SIG_TTL:
                    result[k] = v
                # else: expired — drop silently
        return result
    except Exception as e:
        log.warning("Failed to load thought signatures: %s", e)
        return {}


# In-memory store: {key: {"sig": str, "ts": float}}
_THOUGHT_SIGS: dict[str, dict[str, Any]] = _load_thought_sigs()


def _gc_thought_sigs_locked() -> None:
    """Remove expired entries from _THOUGHT_SIGS (must be called with _SIGS_LOCK held)."""
    now = time.time()
    expired = [k for k, v in _THOUGHT_SIGS.items() if now - float(v.get("ts") or 0) >= _SIG_TTL]
    for k in expired:
        del _THOUGHT_SIGS[k]
    if expired:
        log.debug("GC'd %d expired thought signature(s)", len(expired))


def _save_thought_sig(key: str, val: str) -> None:
    with _SIGS_LOCK:
        _THOUGHT_SIGS[key] = {"sig": val, "ts": time.time()}
        _gc_thought_sigs_locked()
        try:
            os.makedirs(os.path.dirname(_SIGS_PATH), exist_ok=True)
            # Atomic write: write to .tmp then rename so a concurrent reader
            # never sees a half-written JSON file.
            tmp = _SIGS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_THOUGHT_SIGS, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _SIGS_PATH)
        except Exception as e:
            log.warning("Failed to save thought signature: %s", e)

_SCHEMA_KEEP = {
    "type", "description", "properties", "required", "items",
    "enum", "nullable", "format", "minimum", "maximum",
}


def _sig_key(name: str, args: Any) -> str:
    try:
        return name + "|" + json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
        return name + "|" + str(args)


def _msg_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                out.append(str(part["text"]))
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return str(content)


# Allowed inline image MIME types for Gemini cloudcode vision input.
_IMAGE_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".bmp": "image/bmp",
}


def _image_url_to_inline_part(image_url: Any) -> dict[str, Any] | None:
    """Convert an OpenAI image_url part into a Gemini cloudcode inlineData part.

    Accepts either a data: URL ("data:<mime>;base64,<data>") or an http(s) URL
    (downloaded then base64-encoded). Returns
    {"inlineData": {"mimeType": <mime>, "data": <b64>}} matching the convention
    already used by antigravity_proxy_core/antigravity.py for describe_image/media.
    """
    import base64
    import binascii

    url: str | None = None
    if isinstance(image_url, str):
        url = image_url
    elif isinstance(image_url, dict):
        u = image_url.get("url")
        if isinstance(u, str):
            url = u
    if not url:
        return None

    # data: URL — parse inline base64 payload.
    if url.startswith("data:"):
        header, _, data_part = url[len("data:"):].partition(",")
        if not data_part:
            return None
        mime = "image/png"
        is_b64 = False
        if header:
            segs = header.split(";")
            if segs and segs[0]:
                mime = segs[0].strip()
            is_b64 = "base64" in (s.strip().lower() for s in segs[1:])
        try:
            if is_b64:
                raw = base64.b64decode(data_part, validate=False)
                b64 = base64.b64encode(raw).decode("ascii")
            else:
                # URL-encoded (rare) — re-encode raw bytes as base64.
                from urllib.parse import unquote_to_bytes

                raw = unquote_to_bytes(data_part)
                b64 = base64.b64encode(raw).decode("ascii")
        except (binascii.Error, ValueError):
            return None
        return {"inlineData": {"mimeType": mime, "data": b64}}

    # http(s) URL — download and base64-encode.
    if url.startswith("http://") or url.startswith("https://"):
        import httpx

        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
        except Exception:
            log.exception("Failed to fetch image URL for vision input: %s", url[:200])
            return None
        raw = resp.content
        mime = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            # Fall back to extension-based inference.
            from urllib.parse import urlparse
            import os as _os

            ext = _os.path.splitext(urlparse(url).path)[1].lower()
            mime = _IMAGE_MIME_BY_EXT.get(ext, "image/png")
        b64 = base64.b64encode(raw).decode("ascii")
        return {"inlineData": {"mimeType": mime, "data": b64}}

    return None


def _msg_parts(content: Any) -> list[dict[str, Any]]:
    """Convert OpenAI message content into Gemini cloudcode parts.

    Handles plain strings, multimodal lists with text parts
    ({"type":"text","text":...}) and image parts
    ({"type":"image_url","image_url":{"url":...}}). Image parts become
    inlineData parts; everything else is collected as text. Falls back to a
    single text part so the text-only path is fully preserved.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    parts.append({"text": part})
                continue
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image_url" or "image_url" in part:
                inline = _image_url_to_inline_part(part.get("image_url"))
                if inline:
                    parts.append(inline)
                continue
            if part.get("text"):
                parts.append({"text": str(part["text"])})
        return parts
    text = _msg_text(content)
    return [{"text": text}] if text else []


def _last_user_image_parts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Return the inlineData image parts from the last user message, if any.

    Used by the plain-text (no-tools) path to detect a multimodal/vision
    request so it can route through generate_raw() with image parts instead
    of the text-only complete() path. Returns [] for pure-text requests so
    that path is completely unaffected.
    """
    for msg in reversed(messages):
        if msg.role == "user" and isinstance(msg.content, list):
            return [p for p in _msg_parts(msg.content) if "inlineData" in p]
        if msg.role == "user":
            return []
    return []


def _sanitize_schema(schema: Any) -> Any:
    """Strip JSON-schema keys Gemini/Claude rejects, and enforce valid shapes."""
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [_sanitize_schema(s) for s in schema]
        return schema

    # Force object type if properties/required exist but type doesn't
    if ("properties" in schema or "required" in schema) and "type" not in schema:
        schema["type"] = "object"

    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k not in _SCHEMA_KEEP:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _sanitize_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _sanitize_schema(v)
        else:
            out[k] = v

    # Gemini requires non-empty properties for object type
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}

    # Ensure required properties exist in properties
    if "required" in out and isinstance(out["required"], list):
        props = out.get("properties", {})
        valid_required = [r for r in out["required"] if isinstance(r, str) and r in props]
        if valid_required:
            out["required"] = valid_required
        else:
            out.pop("required", None)

    # Force string items if array type missing items
    if out.get("type") == "array" and "items" not in out:
        out["items"] = {"type": "string"}

    return out


def _extra_value(req: Any, name: str, default: Any = None) -> Any:
    value = getattr(req, name, default)
    if value is not default:
        return value
    extra = getattr(req, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name, default)
    return default


def _structured_format(req: Any) -> dict[str, Any] | None:
    text_cfg = _extra_value(req, "text")
    if isinstance(text_cfg, dict):
        fmt = text_cfg.get("format")
        if isinstance(fmt, dict):
            return fmt
    response_format = _extra_value(req, "response_format")
    if isinstance(response_format, dict):
        if isinstance(response_format.get("json_schema"), dict):
            js = response_format["json_schema"]
            return {
                "type": "json_schema",
                "name": js.get("name"),
                "schema": js.get("schema"),
                "strict": js.get("strict"),
            }
        return response_format
    return None


def _apply_structured_output(gen: dict[str, Any], req: Any) -> None:
    fmt = _structured_format(req)
    if not fmt:
        return
    fmt_type = str(fmt.get("type") or "").strip().lower()
    if fmt_type in {"text", "plain_text"}:
        return
    if fmt_type in {"json_object", "json_schema"}:
        gen["responseMimeType"] = "application/json"
        schema = fmt.get("schema")
        if isinstance(schema, dict):
            gen["responseSchema"] = _sanitize_schema(dict(schema))
        return
    raise HTTPException(status_code=400, detail=f"Unsupported structured output format: {fmt.get('type')}")


def _thinking_config(req: Any) -> dict[str, Any] | None:
    reasoning = _extra_value(req, "reasoning")
    if reasoning in (None, False):
        return None
    if isinstance(reasoning, str):
        effort = reasoning.strip().lower()
        return {"thinkingLevel": effort} if effort else None
    if not isinstance(reasoning, dict):
        raise HTTPException(status_code=400, detail="reasoning must be an object or string.")

    cfg: dict[str, Any] = {}
    effort = reasoning.get("effort") or reasoning.get("thinking_level") or reasoning.get("thinkingLevel")
    if isinstance(effort, str) and effort.strip():
        cfg["thinkingLevel"] = effort.strip().lower()
    budget = reasoning.get("budget", reasoning.get("thinking_budget", reasoning.get("thinkingBudget")))
    if budget is not None:
        try:
            cfg["thinkingBudget"] = int(budget)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="reasoning budget must be an integer.")
    include_thoughts = reasoning.get("include_thoughts", reasoning.get("includeThoughts"))
    if include_thoughts is not None:
        cfg["includeThoughts"] = bool(include_thoughts)
    return cfg or None


def _generation_config(req: Any, *, max_tokens_attr: str = "max_tokens") -> dict[str, Any]:
    max_tokens = _extra_value(req, max_tokens_attr)
    gen: dict[str, Any] = {"maxOutputTokens": min(max_tokens or 4096, 65536)}
    temperature = _extra_value(req, "temperature")
    if temperature is not None:
        gen["temperature"] = temperature
    top_p = _extra_value(req, "top_p")
    if top_p is not None:
        gen["topP"] = top_p
    thinking = _thinking_config(req)
    if thinking:
        gen["thinkingConfig"] = thinking
    _apply_structured_output(gen, req)
    return gen


_RESPONSES_GROUNDING_TOOL_TYPES = {"web_search_preview", "web_search_preview_2025_03_11", "google_search"}


def _is_grounding_tool(tool: dict[str, Any]) -> bool:
    return str(tool.get("type") or "").strip().lower() in _RESPONSES_GROUNDING_TOOL_TYPES


def _uses_grounding_tool(tools: list[dict[str, Any]] | None) -> bool:
    return any(isinstance(tool, dict) and _is_grounding_tool(tool) for tool in tools or [])


def _validate_chat_tools(tools: list[dict[str, Any]] | None) -> None:
    for tool in tools or []:
        if not isinstance(tool, dict):
            raise HTTPException(status_code=400, detail=f"Unsupported tool type: {tool}")
        if tool.get("type") == "function" or _is_grounding_tool(tool):
            continue
        raise HTTPException(status_code=400, detail=f"Unsupported tool type: {tool.get('type')}")


def _gemini_tools(function_tools: list[dict[str, Any]] | None, *, grounding: bool = False) -> list[dict[str, Any]] | None:
    gemini_tools: list[dict[str, Any]] = []
    if grounding:
        gemini_tools.append({"google_search": {}})
    converted = _tools_to_gemini(function_tools)
    if converted:
        gemini_tools.extend(converted)
    return gemini_tools or None


def _tools_to_gemini(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    decls: list[dict[str, Any]] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        
        decl: dict[str, Any] = {"name": fn["name"]}
        if fn.get("description"):
            decl["description"] = fn["description"]
        
        params = fn.get("parameters")
        if isinstance(params, dict):
            decl["parameters"] = _sanitize_schema(params)
        else:
            decl["parameters"] = {"type": "object", "properties": {}}
            
        decls.append(decl)
    return [{"functionDeclarations": decls}] if decls else None


def _messages_to_gemini(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    system_texts: list[str] = []
    contents: list[dict[str, Any]] = []
    id_to_name: dict[str, str] = {}
    # tool_call ids that became a STRUCTURED functionCall (had a thoughtSignature).
    # Calls without a signature are rendered as text instead — and their tool
    # results must be too — so cloudcode doesn't 400 on a sig-less functionCall.
    structured_ids: set[str] = set()
    for m in messages:
        if m.role == "system":
            t = _msg_text(m.content)
            if t:
                system_texts.append(t)
        elif m.role == "user":
            user_parts = _msg_parts(m.content)
            if not user_parts:
                user_parts = [{"text": ""}]
            contents.append({"role": "user", "parts": user_parts})
        elif m.role == "assistant":
            parts: list[dict[str, Any]] = []
            t = _msg_text(m.content)
            if t:
                parts.append({"text": t})
            for tc in (m.tool_calls or []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
                except Exception:
                    args = {}
                tc_id = tc.get("id") if isinstance(tc, dict) else None
                if tc_id:
                    id_to_name[tc_id] = name
                with _SIGS_LOCK:
                    _entry = _THOUGHT_SIGS.get(_sig_key(name, args))
                    sig = _entry["sig"] if _entry else None
                if sig:
                    # Thinking models REQUIRE the thoughtSignature on functionCall
                    # parts; we have it, so emit a structured call.
                    parts.append({"functionCall": {"name": name, "args": args}, "thoughtSignature": sig})
                    if tc_id:
                        structured_ids.add(tc_id)
                # else: no cached signature (e.g. proxy restarted mid-conversation).
                # A sig-less functionCall → upstream 400 "missing thought_signature".
                # We DON'T emit a text stand-in for the call itself — the model
                # imitates a "[called ...]" pattern and stops emitting real
                # structured calls. The paired result is summarised as text below,
                # which is enough context without teaching a bad output shape.
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif m.role == "tool":
            name = m.name or id_to_name.get(m.tool_call_id or "", "") or "tool"
            raw = _msg_text(m.content)
            if m.tool_call_id and m.tool_call_id not in structured_ids:
                # Paired call was sig-less (dropped above) — summarise its result as
                # plain text so there's no orphan functionResponse and the model
                # still has the observation as context (capped to avoid bloat).
                contents.append({"role": "user", "parts": [{"text": f"(이전 {name} 도구 결과: {raw[:1500]})"}]})
            else:
                try:
                    resp = json.loads(raw) if raw.strip().startswith(("{", "[")) else {"result": raw}
                except Exception:
                    resp = {"result": raw}
                if not isinstance(resp, dict):
                    resp = {"result": resp}
                contents.append({"role": "user", "parts": [{"functionResponse": {"name": name, "response": resp}}]})

    # Merge consecutive turns with the same role to prevent Gemini API 400 errors
    merged_contents: list[dict[str, Any]] = []
    for turn in contents:
        if not turn.get("parts"):
            continue
        if merged_contents and merged_contents[-1]["role"] == turn["role"]:
            merged_contents[-1]["parts"].extend(turn["parts"])
        else:
            merged_contents.append(turn)

    return "\n\n".join(system_texts), merged_contents


def _parse_gemini_fc(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    inner = data.get("response") if isinstance(data.get("response"), dict) else data
    cands = inner.get("candidates") if isinstance(inner, dict) else None
    if not isinstance(cands, list) or not cands:
        return "", []
    parts = cands[0].get("content", {}).get("parts", []) or []
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("functionCall"), dict):
            fc = p["functionCall"]
            name = fc.get("name") or ""
            args = fc.get("args") or {}
            sig = p.get("thoughtSignature")
            if sig:
                _save_thought_sig(_sig_key(name, args), sig)
            tool_calls.append({
                "id": fc.get("id") or ("call_" + uuid.uuid4().hex[:10]),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            })
        elif p.get("text"):
            texts.append(str(p["text"]))
    return "\n".join(texts).strip(), tool_calls


def _iter_usage_candidates(value: Any):
    if isinstance(value, dict):
        usage_keys = {
            "usageMetadata",
            "usage_metadata",
            "usage",
            "tokenUsage",
            "token_usage",
            "promptTokenCount",
            "prompt_token_count",
            "prompt_tokens",
            "input_tokens",
            "candidatesTokenCount",
            "candidates_token_count",
            "completion_tokens",
            "output_tokens",
            "totalTokenCount",
            "total_token_count",
            "total_tokens",
        }
        if any(k in value for k in usage_keys):
            yield value
        for child in value.values():
            yield from _iter_usage_candidates(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_usage_candidates(child)


def _int_field(data: dict[str, Any], *names: str) -> int:
    for name in names:
        raw = data.get(name)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
        if isinstance(raw, str) and raw.isdigit():
            return int(raw)
    return 0


def _estimate_tokens(value: Any) -> int:
    text = _msg_text(value) if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)
    text = text.strip()
    if not text:
        return 0
    alnum_runs = 0
    in_run = False
    non_space_chars = 0
    cjk_chars = 0
    punctuation = 0
    for ch in text:
        if ch.isspace():
            in_run = False
            continue
        non_space_chars += 1
        if "\u3400" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7a3":
            cjk_chars += 1
            in_run = False
        elif ch.isalnum() or ch == "_":
            if not in_run:
                alnum_runs += 1
                in_run = True
        else:
            punctuation += 1
            in_run = False
    charish = max(1, (non_space_chars + 3) // 4)
    lexical = alnum_runs + max(1, (cjk_chars + 1) // 2) + punctuation
    return max(charish, lexical)


def _estimate_prompt_tokens(messages: list[ChatMessage]) -> int:
    total = 0
    for msg in messages:
        total += 4 + _estimate_tokens(msg.content)
        if msg.tool_calls:
            total += _estimate_tokens(msg.tool_calls)
        if msg.tool_call_id:
            total += _estimate_tokens(msg.tool_call_id)
        if msg.name:
            total += _estimate_tokens(msg.name)
    return total


def _usage_from_response(
    data: dict[str, Any] | None,
    *,
    messages: list[ChatMessage],
    completion: Any,
) -> dict[str, int]:
    if data:
        for candidate in _iter_usage_candidates(data):
            usage = (
                candidate.get("usageMetadata")
                or candidate.get("usage_metadata")
                or candidate.get("usage")
                or candidate.get("tokenUsage")
                or candidate.get("token_usage")
            )
            if isinstance(usage, dict):
                candidate = usage
            prompt = _int_field(
                candidate,
                "promptTokenCount",
                "prompt_token_count",
                "prompt_tokens",
                "input_tokens",
                "inputTokenCount",
                "input_token_count",
            )
            completion_tokens = _int_field(
                candidate,
                "candidatesTokenCount",
                "candidates_token_count",
                "completion_tokens",
                "output_tokens",
                "outputTokenCount",
                "output_token_count",
                "generated_tokens",
            )
            total = _int_field(candidate, "totalTokenCount", "total_token_count", "total_tokens", "tokens")
            if prompt or completion_tokens or total:
                if not total:
                    total = prompt + completion_tokens
                if not prompt and total >= completion_tokens:
                    prompt = total - completion_tokens
                if not completion_tokens and total >= prompt:
                    completion_tokens = total - prompt
                return {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total,
                }

    prompt = _estimate_prompt_tokens(messages)
    completion_tokens = _estimate_tokens(completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt + completion_tokens,
    }


def _tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def _tool_choice_is_none(tool_choice: Any) -> bool:
    return isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"


def _tool_choice_to_gemini(tool_choice: Any, tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        choice = tool_choice.strip().lower()
        if choice == "auto":
            return {"functionCallingConfig": {"mode": "AUTO"}}
        if choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if choice in {"required", "any"}:
            return {"functionCallingConfig": {"mode": "ANY"}}
        raise HTTPException(status_code=400, detail=f"Unsupported tool_choice: {tool_choice}")
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        name = fn.get("name") if isinstance(fn, dict) else tool_choice.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="tool_choice function name is required.")
        name = str(name)
        valid = _tool_names(tools)
        if valid and name not in valid:
            raise HTTPException(status_code=400, detail=f"tool_choice function '{name}' is not in tools.")
        return {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowed_function_names": [name],
            }
        }
    raise HTTPException(status_code=400, detail="Unsupported tool_choice shape.")


def _chunk_text(chunk: dict[str, Any]) -> str:
    """Extract text fragment from a streamGenerateContent response chunk."""
    inner = chunk.get("response", chunk)
    if not isinstance(inner, dict):
        inner = chunk
    cands = (inner.get("candidates") or []) if isinstance(inner, dict) else []
    if not cands:
        return ""
    parts = (cands[0].get("content") or {}).get("parts") or []
    return "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict))


def _fc_sse(response_id: str, created: int, model: str, msg: dict[str, Any], finish: str) -> StreamingResponse:
    def _split_arguments(args: str, size: int = 256) -> list[str]:
        if not args:
            return [""]
        return [args[i:i + size] for i in range(0, len(args), size)]

    def _gen():
        base = {"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model}
        first = dict(base)
        first["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

        if msg.get("content"):
            content_evt = dict(base)
            content_evt["choices"] = [
                {"index": 0, "delta": {"content": msg["content"]}, "finish_reason": None}
            ]
            yield f"data: {json.dumps(content_evt, ensure_ascii=False)}\n\n"

        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            start_evt = dict(base)
            start_evt["choices"] = [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": i,
                        "id": tc.get("id"),
                        "type": tc.get("type", "function"),
                        "function": {"name": fn.get("name", ""), "arguments": ""},
                    }]
                },
                "finish_reason": None,
            }]
            yield f"data: {json.dumps(start_evt, ensure_ascii=False)}\n\n"
            for piece in _split_arguments(str(fn.get("arguments") or "")):
                if not piece:
                    continue
                arg_evt = dict(base)
                arg_evt["choices"] = [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": i,
                            "function": {"arguments": piece},
                        }]
                    },
                    "finish_reason": None,
                }]
                yield f"data: {json.dumps(arg_evt, ensure_ascii=False)}\n\n"

        last = dict(base)
        last["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish}]
        yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream")


def _extract_memories_v2(messages: list[ChatMessage]) -> list[str]:
    """
    Convert conversation history into a list of memory strings.
    Only assistant messages are kept as memories; system and the last user
    message are excluded.
    """
    last_user_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    memories: list[str] = []
    for i, msg in enumerate(messages):
        if msg.role == "system":
            continue
        if msg.role == "user" and i == last_user_idx:
            continue
        content_text = _msg_text(msg.content).strip()
        if content_text:
            label = "assistant" if msg.role == "assistant" else msg.role
            memories.append(f"[{label}]: {content_text}")
    return memories


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """Return the list of supported models (OpenAI-compatible)."""
    return ModelListResponse(data=_MODELS)


@app.get("/v1beta/models")
async def gemini_list_models():
    """Gemini-compatible model listing."""
    return {
        "models": [
            _gemini_model_resource(model)
            for model in _MODELS
            if not _model_capabilities(model)["internal"]
        ]
    }


@app.get("/v1beta/models/{model_name:path}")
async def gemini_get_model(model_name: str):
    """Gemini-compatible model retrieval."""
    try:
        model = _resolve_gemini_model(model_name)
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="NOT_FOUND")
    return _gemini_model_resource(model)


@app.get("/v1beta/files")
async def gemini_list_files(pageSize: int = Query(default=100, ge=1, le=1000), pageToken: str | None = None):
    """Gemini-compatible local Files API listing."""
    index = _gemini_load_files_index()
    files = [_gemini_file_resource(meta) for meta in index.values()]
    files.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {
        "files": files[start:end],
        "nextPageToken": str(end) if end < len(files) else "",
    }


@app.get("/v1beta/files/{file_id:path}")
async def gemini_get_file(file_id: str):
    meta = _gemini_get_file_meta(file_id)
    if not meta:
        return _gemini_error_response(f"File '{file_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_file_resource(meta)


@app.delete("/v1beta/files/{file_id:path}")
async def gemini_delete_file(file_id: str):
    meta = _gemini_get_file_meta(file_id)
    if not meta:
        return _gemini_error_response(f"File '{file_id}' not found.", status_code=404, status="NOT_FOUND")
    index = _gemini_load_files_index()
    index.pop(meta["name"], None)
    _gemini_save_files_index(index)
    path = Path(str(meta.get("path") or ""))
    if path.exists():
        try:
            path.unlink()
        except OSError:
            log.warning("Failed to remove Gemini file blob: %s", path)
    return JSONResponse({})


@app.post("/v1beta/files")
@app.post("/upload/v1beta/files")
async def gemini_upload_file(request: Request):
    """Gemini-compatible simple media/multipart file upload."""
    try:
        if request.headers.get("x-goog-upload-protocol", "").lower() == "resumable":
            return await _gemini_start_resumable_upload(request)
        file_resource = await _gemini_upload_file_from_request(request)
        return {"file": file_resource}
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="INVALID_ARGUMENT")
    except Exception as exc:
        log.exception("Gemini file upload failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/upload/v1beta/files/{session_id}")
@app.put("/upload/v1beta/files/{session_id}")
async def gemini_resumable_upload(session_id: str, request: Request):
    try:
        command = request.headers.get("x-goog-upload-command", "").lower()
        if command and "finalize" not in command and "upload" not in command:
            raise HTTPException(status_code=400, detail=f"Unsupported upload command: {command}")
        return {"file": await _gemini_finish_resumable_upload(session_id, request)}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini resumable upload failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1beta/cachedContents")
async def gemini_create_cached_content(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return _gemini_create_cached_content(body)
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="INVALID_ARGUMENT")
    except Exception as exc:
        log.exception("Gemini cachedContents create failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.get("/v1beta/cachedContents")
async def gemini_list_cached_contents(pageSize: int = Query(default=100, ge=1, le=1000), pageToken: str | None = None):
    index = _gemini_load_cached_index()
    items = [_gemini_cached_resource(meta) for meta in index.values()]
    items.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"cachedContents": items[start:end], "nextPageToken": str(end) if end < len(items) else ""}


@app.get("/v1beta/cachedContents/{cache_id:path}")
async def gemini_get_cached_content(cache_id: str):
    meta = _gemini_get_cached_meta(cache_id)
    if not meta:
        return _gemini_error_response(f"Cached content '{cache_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_cached_resource(meta)


@app.patch("/v1beta/cachedContents/{cache_id:path}")
async def gemini_patch_cached_content(cache_id: str, request: Request):
    meta = _gemini_get_cached_meta(cache_id)
    if not meta:
        return _gemini_error_response(f"Cached content '{cache_id}' not found.", status_code=404, status="NOT_FOUND")
    body = await request.json()
    if isinstance(body, dict):
        if body.get("ttl"):
            meta["ttl"] = body["ttl"]
        if body.get("expireTime"):
            meta["expireTime"] = body["expireTime"]
        meta["updateTime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        index = _gemini_load_cached_index()
        index[meta["name"]] = meta
        _gemini_save_cached_index(index)
    return _gemini_cached_resource(meta)


@app.delete("/v1beta/cachedContents/{cache_id:path}")
async def gemini_delete_cached_content(cache_id: str):
    meta = _gemini_get_cached_meta(cache_id)
    if not meta:
        return _gemini_error_response(f"Cached content '{cache_id}' not found.", status_code=404, status="NOT_FOUND")
    index = _gemini_load_cached_index()
    index.pop(meta["name"], None)
    _gemini_save_cached_index(index)
    return JSONResponse({})


@app.post("/v1beta/fileSearchStores")
async def gemini_create_file_search_store(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return _gemini_create_fss(body)
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="INVALID_ARGUMENT")
    except Exception as exc:
        log.exception("Gemini fileSearchStores create failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.get("/v1beta/fileSearchStores")
async def gemini_list_file_search_stores(pageSize: int = Query(default=100, ge=1, le=1000), pageToken: str | None = None):
    stores = [_gemini_fss_resource(meta) for meta in _gemini_load_fss_index().values()]
    stores.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"fileSearchStores": stores[start:end], "nextPageToken": str(end) if end < len(stores) else ""}


@app.get("/v1beta/fileSearchStores/{store_id}")
async def gemini_get_file_search_store(store_id: str):
    meta = _gemini_get_fss_meta(store_id)
    if not meta:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_fss_resource(meta)


@app.delete("/v1beta/fileSearchStores/{store_id}")
async def gemini_delete_file_search_store(store_id: str):
    name = _gemini_fss_name(store_id)
    index = _gemini_load_fss_index()
    if name not in index:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    index.pop(name, None)
    _gemini_save_fss_index(index)
    return JSONResponse({})


@app.post("/v1beta/fileSearchStores/{store_id}:importFile")
async def gemini_import_file_to_file_search_store(store_id: str, request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        document = _gemini_import_file_to_fss(store_id, body)
        operation = {
            "name": "operations/importFile-" + uuid.uuid4().hex,
            "metadata": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.ImportFileMetadata",
                "fileSearchStore": _gemini_fss_name(store_id),
            },
            "done": True,
            "response": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.ImportFileResponse",
                "document": document,
            },
        }
        return _gemini_store_operation(operation)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini fileSearchStores importFile failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/upload/v1beta/fileSearchStores/{store_id}:uploadToFileSearchStore")
@app.post("/v1beta/fileSearchStores/{store_id}:uploadToFileSearchStore")
async def gemini_upload_to_file_search_store(store_id: str, request: Request):
    try:
        content_type = request.headers.get("content-type", "")
        body = await request.body()
        metadata: dict[str, Any] = {}
        media = body
        media_type = content_type
        if "multipart/" in content_type:
            metadata, media, parsed_type = _parse_gemini_multipart_upload(content_type, body)
            media_type = parsed_type or content_type
        file_meta = metadata.get("file") if isinstance(metadata.get("file"), dict) else metadata
        display_name = request.query_params.get("displayName")
        if isinstance(file_meta, dict):
            display_name = file_meta.get("displayName") or file_meta.get("display_name") or display_name
            media_type = file_meta.get("mimeType") or file_meta.get("mime_type") or media_type
        document = _gemini_store_document(store_id, display_name=display_name, mime_type=media_type, content=media)
        operation = {
            "name": "operations/uploadToFileSearchStore-" + uuid.uuid4().hex,
            "metadata": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.UploadToFileSearchStoreMetadata",
                "fileSearchStore": _gemini_fss_name(store_id),
            },
            "done": True,
            "response": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.UploadToFileSearchStoreResponse",
                "document": document,
            },
        }
        return _gemini_store_operation(operation)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini uploadToFileSearchStore failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.get("/v1beta/fileSearchStores/{store_id}/documents")
async def gemini_list_file_search_documents(store_id: str, pageSize: int = Query(default=100, ge=1, le=1000), pageToken: str | None = None):
    meta = _gemini_get_fss_meta(store_id)
    if not meta:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    docs = [_gemini_document_resource(doc) for doc in (meta.get("documents") or {}).values()]
    docs.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"documents": docs[start:end], "nextPageToken": str(end) if end < len(docs) else ""}


@app.get("/v1beta/fileSearchStores/{store_id}/documents/{document_id:path}")
async def gemini_get_file_search_document(store_id: str, document_id: str):
    meta = _gemini_get_fss_meta(store_id)
    if not meta:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    doc = (meta.get("documents") or {}).get(_gemini_document_name(store_id, document_id))
    if not doc:
        return _gemini_error_response(f"Document '{document_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_document_resource(doc)


@app.delete("/v1beta/fileSearchStores/{store_id}/documents/{document_id:path}")
async def gemini_delete_file_search_document(store_id: str, document_id: str):
    index = _gemini_load_fss_index()
    store_name = _gemini_fss_name(store_id)
    meta = index.get(store_name)
    if not meta:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    doc_name = _gemini_document_name(store_id, document_id)
    if doc_name not in (meta.get("documents") or {}):
        return _gemini_error_response(f"Document '{document_id}' not found.", status_code=404, status="NOT_FOUND")
    meta["documents"].pop(doc_name, None)
    meta["updateTime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    index[store_name] = meta
    _gemini_save_fss_index(index)
    return JSONResponse({})


@app.post("/v1beta/tunedModels")
async def gemini_create_tuned_model(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        tuned = _gemini_create_tuned_model(body)
        operation = {
            "name": "operations/createTunedModel-" + uuid.uuid4().hex,
            "metadata": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.CreateTunedModelMetadata",
                "tunedModel": tuned["name"],
            },
            "done": True,
            "response": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.TunedModel",
                **tuned,
            },
        }
        return _gemini_store_operation(operation)
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="INVALID_ARGUMENT")
    except Exception as exc:
        log.exception("Gemini tunedModels create failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.get("/v1beta/tunedModels")
async def gemini_list_tuned_models(pageSize: int = Query(default=100, ge=1, le=1000), pageToken: str | None = None):
    models = [_gemini_tuned_resource(meta) for meta in _gemini_load_tuned_index().values()]
    models.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"tunedModels": models[start:end], "nextPageToken": str(end) if end < len(models) else ""}


@app.get("/v1beta/tunedModels/{tuned_model_id}")
async def gemini_get_tuned_model(tuned_model_id: str):
    meta = _gemini_get_tuned_meta(tuned_model_id)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_tuned_resource(meta)


@app.patch("/v1beta/tunedModels/{tuned_model_id}")
async def gemini_patch_tuned_model(tuned_model_id: str, request: Request):
    index = _gemini_load_tuned_index()
    name = _gemini_tuned_name(tuned_model_id)
    meta = index.get(name)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    body = _gemini_normalize_request(await request.json())
    if isinstance(body, dict):
        for key in ("displayName", "description", "temperature", "topP", "topK"):
            if key in body:
                meta[key] = body[key]
        meta["updateTime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        index[name] = meta
        _gemini_save_tuned_index(index)
    return _gemini_tuned_resource(meta)


@app.delete("/v1beta/tunedModels/{tuned_model_id}")
async def gemini_delete_tuned_model(tuned_model_id: str):
    index = _gemini_load_tuned_index()
    name = _gemini_tuned_name(tuned_model_id)
    if name not in index:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    index.pop(name, None)
    _gemini_save_tuned_index(index)
    return JSONResponse({})


@app.post("/v1beta/tunedModels/{tuned_model_id}:generateContent")
async def gemini_tuned_generate_content(tuned_model_id: str, request: Request):
    try:
        model = _gemini_tuned_base_model(tuned_model_id)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        body = _gemini_inline_local_files(_gemini_apply_file_search(_gemini_apply_cached_content(body)))
        body.pop("model", None)
        data = await asyncio.to_thread(_get_client().generate_raw, request=body, model=str(model["antigravity_model"]))
        return JSONResponse(_gemini_unwrap_response(data))
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini tuned model generateContent failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1beta/tunedModels/{tuned_model_id}:countTokens")
async def gemini_tuned_count_tokens(tuned_model_id: str, request: Request):
    try:
        _gemini_tuned_base_model(tuned_model_id)
        body = _gemini_normalize_request(await request.json())
        if isinstance(body, dict):
            body = _gemini_apply_file_search(_gemini_apply_cached_content(body))
        return {"totalTokens": _estimate_prompt_tokens(_gemini_count_tokens_request(body if isinstance(body, dict) else {}))}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.get("/v1beta/tunedModels/{tuned_model_id}/permissions")
async def gemini_list_tuned_model_permissions(tuned_model_id: str):
    meta = _gemini_get_tuned_meta(tuned_model_id)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    return {"permissions": list((meta.get("permissions") or {}).values())}


@app.post("/v1beta/tunedModels/{tuned_model_id}/permissions")
async def gemini_create_tuned_model_permission(tuned_model_id: str, request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return _gemini_store_permission(tuned_model_id, body)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.get("/v1beta/tunedModels/{tuned_model_id}/permissions/{permission_id}")
async def gemini_get_tuned_model_permission(tuned_model_id: str, permission_id: str):
    meta = _gemini_get_tuned_meta(tuned_model_id)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    perm = (meta.get("permissions") or {}).get(_gemini_permission_name(tuned_model_id, permission_id))
    if not perm:
        return _gemini_error_response(f"Permission '{permission_id}' not found.", status_code=404, status="NOT_FOUND")
    return perm


@app.patch("/v1beta/tunedModels/{tuned_model_id}/permissions/{permission_id}")
async def gemini_patch_tuned_model_permission(tuned_model_id: str, permission_id: str, request: Request):
    index = _gemini_load_tuned_index()
    name = _gemini_tuned_name(tuned_model_id)
    meta = index.get(name)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    perm_name = _gemini_permission_name(tuned_model_id, permission_id)
    perm = (meta.get("permissions") or {}).get(perm_name)
    if not perm:
        return _gemini_error_response(f"Permission '{permission_id}' not found.", status_code=404, status="NOT_FOUND")
    body = _gemini_normalize_request(await request.json())
    if isinstance(body, dict):
        for key in ("role", "granteeType", "emailAddress"):
            if key in body:
                perm[key] = body[key]
    meta.setdefault("permissions", {})[perm_name] = perm
    index[name] = meta
    _gemini_save_tuned_index(index)
    return perm


@app.post("/v1beta/tunedModels/{tuned_model_id}/permissions/{permission_id}:transferOwnership")
async def gemini_transfer_tuned_model_permission(tuned_model_id: str, permission_id: str):
    index = _gemini_load_tuned_index()
    name = _gemini_tuned_name(tuned_model_id)
    meta = index.get(name)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    perm_name = _gemini_permission_name(tuned_model_id, permission_id)
    perm = (meta.get("permissions") or {}).get(perm_name)
    if not perm:
        return _gemini_error_response(f"Permission '{permission_id}' not found.", status_code=404, status="NOT_FOUND")
    perm["role"] = "OWNER"
    meta.setdefault("permissions", {})[perm_name] = perm
    index[name] = meta
    _gemini_save_tuned_index(index)
    return JSONResponse({})


@app.delete("/v1beta/tunedModels/{tuned_model_id}/permissions/{permission_id}")
async def gemini_delete_tuned_model_permission(tuned_model_id: str, permission_id: str):
    index = _gemini_load_tuned_index()
    name = _gemini_tuned_name(tuned_model_id)
    meta = index.get(name)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    perm_name = _gemini_permission_name(tuned_model_id, permission_id)
    if perm_name not in (meta.get("permissions") or {}):
        return _gemini_error_response(f"Permission '{permission_id}' not found.", status_code=404, status="NOT_FOUND")
    meta["permissions"].pop(perm_name, None)
    index[name] = meta
    _gemini_save_tuned_index(index)
    return JSONResponse({})


@app.post("/v1beta/models/{model_name:path}:countTokens")
async def gemini_count_tokens(model_name: str, request: Request):
    """Gemini-compatible approximate countTokens endpoint."""
    try:
        _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if isinstance(body, dict):
            body = _gemini_apply_cached_content(body)
            body = _gemini_apply_file_search(body)
        messages = _gemini_count_tokens_request(body if isinstance(body, dict) else {})
        return {"totalTokens": _estimate_prompt_tokens(messages)}
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code)
    except Exception as exc:
        log.exception("Gemini countTokens failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1beta/models/{model_name:path}:embedContent")
async def gemini_embed_content(model_name: str, request: Request):
    """Gemini-compatible embedContent endpoint with deterministic local vectors."""
    try:
        _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return _gemini_embedding_from_request(body)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini embedContent failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1beta/models/{model_name:path}:batchEmbedContents")
async def gemini_batch_embed_contents(model_name: str, request: Request):
    """Gemini-compatible batchEmbedContents endpoint with deterministic local vectors."""
    try:
        _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return _gemini_batch_embedding_from_request(body)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini batchEmbedContents failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1beta/models/{model_name:path}:generateContent")
async def gemini_generate_content(model_name: str, request: Request):
    """Gemini REST-compatible generateContent endpoint backed by Antigravity."""
    try:
        model = _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        body = _gemini_apply_cached_content(body)
        body = _gemini_apply_file_search(body)
        body = _gemini_inline_local_files(body)
        body.pop("model", None)
        data = await asyncio.to_thread(
            _get_client().generate_raw,
            request=body,
            model=str(model["antigravity_model"]),
        )
        return JSONResponse(_gemini_unwrap_response(data))
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        _upstream = ""
        _resp = getattr(exc, "response", None)
        if _resp is not None:
            try:
                _upstream = _resp.text[:2000]
            except Exception:
                pass
        log.error("Gemini generateContent failed: %s | UPSTREAM BODY: %s", exc, _upstream)
        log.exception("Gemini generateContent exception traceback:")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1beta/models/{model_name:path}:batchGenerateContent")
async def gemini_batch_generate_content(model_name: str, request: Request):
    """Gemini-compatible batchGenerateContent as an immediately completed operation."""
    try:
        model = _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        requests = body.get("requests")
        if not isinstance(requests, list):
            raise HTTPException(status_code=400, detail="batchGenerateContent requires a requests array.")
        responses: list[dict[str, Any]] = []
        for item in requests:
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="batchGenerateContent request items must be objects.")
            req_body = _gemini_normalize_request(dict(item))
            req_body = _gemini_apply_cached_content(req_body)
            req_body = _gemini_apply_file_search(req_body)
            req_body = _gemini_inline_local_files(req_body)
            req_body.pop("model", None)
            data = await asyncio.to_thread(
                _get_client().generate_raw,
                request=req_body,
                model=str(model["antigravity_model"]),
            )
            responses.append(_gemini_unwrap_response(data))
        operation = {
            "name": "operations/batchGenerateContent-" + uuid.uuid4().hex,
            "metadata": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.BatchGenerateContentMetadata",
                "model": _gemini_model_name(model),
                "requestCount": len(requests),
            },
            "done": True,
            "response": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.BatchGenerateContentResponse",
                "responses": responses,
            },
        }
        return _gemini_store_operation(operation)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini batchGenerateContent failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1beta/models/{model_name:path}:streamGenerateContent")
async def gemini_stream_generate_content(model_name: str, request: Request):
    """Gemini REST-compatible SSE streamGenerateContent endpoint."""
    try:
        model = _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        body = _gemini_apply_cached_content(body)
        body = _gemini_apply_file_search(body)
        body = _gemini_inline_local_files(body)
        body.pop("model", None)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")

    async def _gen():
        try:
            async for chunk in _get_client().generate_raw_stream_async(
                request=body,
                model=str(model["antigravity_model"]),
            ):
                payload = _gemini_unwrap_response(chunk)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            log.warning("Gemini streamGenerateContent failed; falling back to non-streaming: %s", exc)
            try:
                data = await asyncio.to_thread(
                    _get_client().generate_raw,
                    request=body,
                    model=str(model["antigravity_model"]),
                )
                yield f"data: {json.dumps(_gemini_unwrap_response(data), ensure_ascii=False)}\n\n"
            except Exception as inner:
                payload = {"error": {"code": 502, "message": f"Antigravity upstream error: {inner}", "status": "UNAVAILABLE"}}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/v1beta/operations")
async def gemini_list_operations(pageSize: int = Query(default=100, ge=1, le=1000), pageToken: str | None = None):
    index = _gemini_load_operations_index()
    operations = list(index.values())
    operations.sort(key=lambda item: item.get("name") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"operations": operations[start:end], "nextPageToken": str(end) if end < len(operations) else ""}


@app.get("/v1beta/operations/{operation_id:path}")
async def gemini_get_operation(operation_id: str):
    operation = _gemini_get_operation(operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.post("/v1beta/operations/{operation_id:path}:cancel")
async def gemini_cancel_operation(operation_id: str):
    operation = _gemini_get_operation(operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    if not operation.get("done"):
        operation["done"] = True
        operation["error"] = {"code": 1, "message": "Operation cancelled.", "status": "CANCELLED"}
        _gemini_store_operation(operation)
    return JSONResponse({})


@app.delete("/v1beta/operations/{operation_id:path}")
async def gemini_delete_operation(operation_id: str):
    name = _gemini_operation_name(operation_id)
    index = _gemini_load_operations_index()
    if name not in index:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    index.pop(name, None)
    _gemini_save_operations_index(index)
    return JSONResponse({})


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Antigravity itself is non-streaming, but many clients (e.g. the Hermes
    agent) always request `stream: true`. We compute the full completion and,
    when streaming is requested, emit it as a single OpenAI SSE chunk followed
    by [DONE]. This keeps both streaming and non-streaming clients working.
    """
    # Resolve model
    model_info = _MODEL_MAP.get(req.model)
    if model_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' not found. Available: {list(_MODEL_MAP.keys())}",
        )
    antigravity_model: str = model_info["antigravity_model"]
    client = _get_client()
    _validate_chat_tools(req.tools)

    # ── Function-calling path (the Hermes agent) ──
    # When the request carries tool definitions, forward them to Antigravity as
    # Gemini functionDeclarations and translate functionCall responses back into
    # OpenAI tool_calls — this is what makes the agent able to actually run tools.
    function_tools = [t for t in (req.tools or []) if isinstance(t, dict) and t.get("type") == "function"]
    grounding = _uses_grounding_tool(req.tools)
    raw_features = bool(req.tools) or _structured_format(req) is not None or _extra_value(req, "reasoning") is not None
    if raw_features:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        system_text, contents = _messages_to_gemini(req.messages)
        gemini_tools = None if _tool_choice_is_none(req.tool_choice) else _gemini_tools(function_tools, grounding=grounding)
        # Accept large client requests but cap to what Gemini accepts (3.x flash
        # supports up to 65536 output tokens).
        gen = _generation_config(req, max_tokens_attr="max_tokens")
        request_body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            ],
        }
        if system_text:
            request_body["systemInstruction"] = {"role": "system", "parts": [{"text": system_text}]}
        if gemini_tools:
            request_body["tools"] = gemini_tools
        tool_config = None
        if function_tools and not _tool_choice_is_none(req.tool_choice):
            tool_config = _tool_choice_to_gemini(req.tool_choice, function_tools)
        if tool_config:
            request_body["toolConfig"] = tool_config
        try:
            data = await asyncio.to_thread(client.generate_raw, request=request_body, model=antigravity_model)
        except Exception as exc:
            _upstream = ""
            _resp = getattr(exc, "response", None)
            if _resp is not None:
                try:
                    _upstream = _resp.text[:2000]
                except Exception:
                    pass
            log.error("generate_raw (function-calling) failed: %s | UPSTREAM BODY: %s", exc, _upstream)
            log.exception("generate_raw exception traceback:")
            raise HTTPException(status_code=502, detail=f"Antigravity upstream error: {exc}")
        text, tool_calls = _parse_gemini_fc(data)
        if not text and not tool_calls:
            raise HTTPException(status_code=502, detail="Antigravity returned empty response.")
        finish = "tool_calls" if tool_calls else "stop"
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        usage = _usage_from_response(data, messages=req.messages, completion=message)
        if req.stream:
            return _fc_sse(response_id, created, req.model, message, finish)
        return JSONResponse({
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        })

    # ── Plain text path (the lightweight chatbot, no tools) ──
    # Convert OpenAI messages → Antigravity params
    system = _extract_system_prompt(req.messages)
    prompt = _extract_user_prompt(req.messages)
    memories = _extract_memories_v2(req.messages)

    # Vision: if the last user message carries image parts, route through
    # generate_raw() with inlineData parts (complete() is text-only). Pure-text
    # requests have no image parts here, so they fall through untouched.
    image_parts = _last_user_image_parts(req.messages)

    if not prompt and not image_parts:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    # Build IDs once — shared by both streaming and non-streaming paths below.
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if image_parts:
        # Build a normal text request, then replace the user parts with the
        # full text+image part list so the upstream sees the image inline.
        raw_req = client._build_gemini_request(
            system=system,
            prompt=prompt or "이 이미지를 설명해줘.",
            memories=memories,
            grounding=False,
        )
        user_parts: list[dict[str, Any]] = []
        if prompt:
            user_parts.append({"text": prompt})
        user_parts.extend(image_parts)
        raw_req["contents"] = [{"role": "user", "parts": user_parts}]
        if req.max_tokens:
            raw_req["generationConfig"]["maxOutputTokens"] = req.max_tokens

        try:
            data = await asyncio.to_thread(
                client.generate_raw,
                request=raw_req,
                model=antigravity_model,
            )
            vision_text = client._extract_text(data)
        except Exception as exc:
            log.exception("Vision generate_raw() failed")
            raise HTTPException(
                status_code=502,
                detail=f"Antigravity vision upstream error: {exc}",
            )

        if not vision_text:
            raise HTTPException(status_code=502, detail="Antigravity returned empty vision response.")

        if req.stream:
            def _vision_sse():
                base = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                }
                first = dict(base)
                first["choices"] = [
                    {"index": 0, "delta": {"role": "assistant", "content": vision_text}, "finish_reason": None}
                ]
                yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
                last = dict(base)
                last["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_vision_sse(), media_type="text/event-stream")

        return ChatCompletionResponse(
            id=response_id,
            created=created,
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=vision_text),
                    finish_reason="stop",
                )
            ],
            usage=ChatUsage(**_usage_from_response(data, messages=req.messages, completion=vision_text)),
        )

    if req.stream:
        # Try real upstream SSE via v1internal:streamGenerateContent (AsyncClient).
        # If the endpoint is unsupported or returns an error, fall back to
        # complete() + fake single-chunk SSE so all clients stay working.
        raw_req = client._build_gemini_request(system=system, prompt=prompt, memories=memories)

        async def _sse_gen():
            base = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
            }
            first_frag = True
            got_any = False
            try:
                async for chunk in client.generate_raw_stream_async(
                    request=raw_req, model=antigravity_model
                ):
                    frag = _chunk_text(chunk)
                    if frag:
                        got_any = True
                        delta: dict[str, Any] = {"content": frag}
                        if first_frag:
                            delta["role"] = "assistant"
                            first_frag = False
                        evt = dict(base)
                        evt["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
                        yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                if got_any:
                    last = dict(base)
                    last["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                # Upstream returned no text chunks — fall through to complete()
                log.warning("streamGenerateContent returned no text chunks; falling back to complete()")
            except Exception as exc:
                log.warning(
                    "Upstream streaming failed (%s); falling back to complete()", exc
                )

            # Fallback: complete() → fake single-chunk SSE
            try:
                fb_text = await asyncio.to_thread(
                    client.complete,
                    system=system,
                    prompt=prompt,
                    memories=memories,
                    model=antigravity_model,
                )
            except Exception as exc2:
                log.error("Fallback complete() also failed: %s", exc2)
                return
            if fb_text:
                fb_first = dict(base)
                fb_first["choices"] = [
                    {"index": 0, "delta": {"role": "assistant", "content": fb_text}, "finish_reason": None}
                ]
                yield f"data: {json.dumps(fb_first, ensure_ascii=False)}\n\n"
            fb_last = dict(base)
            fb_last["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            yield f"data: {json.dumps(fb_last, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_gen(), media_type="text/event-stream")

    # ── Non-streaming path ──
    # complete() is synchronous — run in thread to avoid blocking the event loop
    try:
        text = await asyncio.to_thread(
            client.complete,
            system=system,
            prompt=prompt,
            memories=memories,
            model=antigravity_model,
        )
    except Exception as exc:
        log.exception("AntigravityClient.complete() failed")
        raise HTTPException(
            status_code=502,
            detail=f"Antigravity upstream error: {exc}",
        )

    if not text:
        raise HTTPException(status_code=502, detail="Antigravity returned empty response.")

    return ChatCompletionResponse(
        id=response_id,
        created=created,
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(**_usage_from_response(None, messages=req.messages, completion=text)),
    )


@app.post("/v1/responses")
async def create_response(req: ResponseCreateRequest):
    if req.background:
        response = {
            "id": _new_response_id(),
            "object": "response",
            "created_at": int(time.time()),
            "status": "queued",
            "background": True,
            "error": None,
            "incomplete_details": None,
            "instructions": req.instructions,
            "max_output_tokens": req.max_output_tokens,
            "model": req.model,
            "output": [],
            "output_text": "",
            "parallel_tool_calls": bool(req.parallel_tool_calls),
            "previous_response_id": req.previous_response_id,
            "reasoning": req.reasoning,
            "store": bool(req.store),
            "temperature": req.temperature,
            "text": req.text,
            "tool_choice": req.tool_choice,
            "tools": req.tools or [],
            "top_p": req.top_p,
            "truncation": req.truncation or "disabled",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "metadata": req.metadata or {},
            "input_items": [],
        }
        _, input_items = _normalize_responses_input(req.input, instructions=req.instructions)
        response["input_items"] = input_items
        if req.store:
            _responses_store_save(response, input_items)
        if req.stream:
            return _responses_sse(response)
        return JSONResponse(response)

    response, input_items = await _responses_generate(req)
    if req.store:
        _responses_store_save(response, input_items)
    if req.stream:
        return _responses_sse(response)
    return JSONResponse(response)


@app.post("/v1/responses/input_tokens")
async def count_response_input_tokens(req: ResponseInputTokensRequest):
    messages, _items = _normalize_responses_input(req.input, instructions=req.instructions)
    return {
        "object": "response.input_tokens",
        "input_tokens": _estimate_prompt_tokens(messages),
    }


@app.get("/v1/responses/{response_id}")
async def retrieve_response(response_id: str):
    response = _responses_store_get(response_id)
    if not response:
        raise HTTPException(status_code=404, detail=f"Response '{response_id}' not found.")
    return JSONResponse(response)


@app.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):
    if not _responses_store_delete(response_id):
        raise HTTPException(status_code=404, detail=f"Response '{response_id}' not found.")
    return {"id": response_id, "object": "response.deleted", "deleted": True}


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str):
    response = _responses_store_get(response_id)
    if not response:
        raise HTTPException(status_code=404, detail=f"Response '{response_id}' not found.")
    if response.get("status") in {"queued", "in_progress"}:
        response = _responses_store_update_status(response_id, "cancelled") or response
    return JSONResponse(response)


@app.get("/v1/responses/{response_id}/input_items")
async def list_response_input_items(response_id: str):
    items = _responses_store_get_input_items(response_id)
    if items is None:
        raise HTTPException(status_code=404, detail=f"Response '{response_id}' not found.")
    return {
        "object": "list",
        "data": items,
        "first_id": items[0]["id"] if items else None,
        "last_id": items[-1]["id"] if items else None,
        "has_more": False,
    }


@app.post("/v1/responses/{response_id}/compact")
async def compact_response(response_id: str):
    response = _responses_store_get(response_id)
    if not response:
        raise HTTPException(status_code=404, detail=f"Response '{response_id}' not found.")
    compact_input = (
        "Summarize the following prior response into a compact context message. "
        "Preserve user intent, tool results, and important facts.\n\n"
        + json.dumps({
            "input_items": _responses_store_get_input_items(response_id) or [],
            "output": response.get("output") or [],
            "output_text": response.get("output_text") or "",
        }, ensure_ascii=False)
    )
    req = ResponseCreateRequest(
        model=str(response.get("model") or "gemini-3-flash"),
        input=compact_input,
        store=True,
        metadata={"compact_of": response_id},
    )
    compacted, input_items = await _responses_generate(req)
    _responses_store_save(compacted, input_items)
    return JSONResponse(compacted)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/admin/models/refresh")
async def refresh_models(request: Request):
    """Refresh the dynamic Antigravity model catalog without restarting."""
    if not _proxy_api_key():
        return _openai_error_response(
            "Admin model refresh requires ANTIGRAVITY_PROXY_API_KEY to be configured.",
            status_code=403,
            error_type="permission_error",
            code="admin_api_key_not_configured",
        )
    if not _request_api_key_valid(request):
        return _openai_error_response(
            "Invalid or missing API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    result = await _fetch_and_update_models()
    status_code = 200 if result.get("ok") else 502
    return JSONResponse(result, status_code=status_code)


# ---------------------------------------------------------------------------
# SearXNG-compatible web search, backed by Gemini Google-Search grounding.
#
# Hermes' built-in ``searxng`` web-search backend issues
#   GET {SEARXNG_URL}/search?q=...&format=json&pageno=1
# and expects a SearXNG JSON body: {"results": [{title, url, content, score}]}.
# We answer that exact contract here using Antigravity's ``google_search``
# grounding tool, so the Hermes agent gets real Google-grounded web search with
# ZERO changes to Hermes itself — just set web.search_backend=searxng and
# SEARXNG_URL to this proxy. Survives `hermes update` (only our code + ~/.hermes
# config/.env are involved, never the Hermes git tree).
# ---------------------------------------------------------------------------
# Grounding model = the REAL 3.1 Flash Lite (`gemini-3.1-flash-lite`, Antigravity's
# alias for Google MODEL_GOOGLE_GEMINI_2_5_FLASH_LITE; `gemini-2.5-flash-lite` is the
# same model). Confirmed via v1internal:fetchAvailableModels — our proxy's _MODELS
# map mislabelled "Gemini 3.1 Flash Lite" as plain `gemini-2.5-flash` (regular Flash).
# The lite tier is a NON-thinking model: ~2-7s/query with rich grounding chunks
# (10-21), no thinkingBudget needed. Set ANTIGRAVITY_GROUNDING_THINKING>=0 to add a
# thinkingConfig (only useful for thinking models like gemini-3-flash-agent). All
# env-overridable.
_GROUNDING_MODEL = os.getenv("ANTIGRAVITY_GROUNDING_MODEL", "gemini-3.1-flash-lite")
_GROUNDING_THINKING = int(os.getenv("ANTIGRAVITY_GROUNDING_THINKING", "-1"))  # <0 = no thinkingConfig
_GROUNDING_MAXTOK = int(os.getenv("ANTIGRAVITY_GROUNDING_MAXTOK", "0"))  # 0 = no output cap


def _grounding_search(query: str, limit: int) -> dict[str, Any]:
    """Run a Google-Search-grounded query and map it to a SearXNG JSON body."""
    client = _get_client()
    # The lite default needs no caps: measured 2-7s with rich chunks even with no
    # output limit. Both knobs are opt-in for thinking/heavier models — set
    # ANTIGRAVITY_GROUNDING_MAXTOK>0 to cap output, ANTIGRAVITY_GROUNDING_THINKING>=0
    # to attach a thinkingConfig — to keep them under Hermes' 15s timeout.
    _gencfg: dict[str, Any] = {"temperature": 0.0}
    if _GROUNDING_MAXTOK > 0:
        _gencfg["maxOutputTokens"] = _GROUNDING_MAXTOK
    if _GROUNDING_THINKING >= 0:
        _gencfg["thinkingConfig"] = {"thinkingBudget": _GROUNDING_THINKING}
    req = {
        "contents": [{"role": "user", "parts": [{"text": query}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": _gencfg,
    }
    data = client.generate_raw(request=req, model=_GROUNDING_MODEL)
    # Antigravity wraps the Gemini payload under "response".
    resp = data.get("response", data)
    cand = (resp.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts") or []
    answer = "".join(p.get("text", "") for p in parts).strip()
    gm = cand.get("groundingMetadata") or cand.get("grounding_metadata") or {}
    chunks = gm.get("groundingChunks") or gm.get("grounding_chunks") or []
    supports = gm.get("groundingSupports") or gm.get("grounding_supports") or []

    # Build a snippet per source from the answer segments that cite it.
    snippet_by_chunk: dict[int, list[str]] = {}
    for s in supports:
        seg = (s.get("segment") or {}).get("text", "")
        idxs = s.get("groundingChunkIndices") or s.get("grounding_chunk_indices") or []
        for idx in idxs:
            if seg:
                snippet_by_chunk.setdefault(int(idx), []).append(seg)

    results: list[dict[str, Any]] = []
    n = len(chunks)
    for i, ch in enumerate(chunks):
        w = ch.get("web") or {}
        url = (w.get("uri") or "").strip()
        if not url:
            continue
        title = (w.get("title") or url).strip()
        snippet = " ".join(snippet_by_chunk.get(i, [])).strip() or answer[:400]
        results.append({
            "title": title,
            "url": url,  # vertexaisearch grounding-redirect → resolves to the real source
            "content": snippet[:600],
            "score": float(n - i),  # preserve Gemini's source ordering
            "engine": "google-grounding",
        })

    results = results[:limit]
    return {
        "query": query,
        "number_of_results": len(results),
        "results": results,
        "answers": [answer] if answer else [],
        "suggestions": gm.get("webSearchQueries") or gm.get("web_search_queries") or [],
        "infoboxes": [],
    }


_SEARCH_CACHE: dict[str, dict[str, Any]] = {}
_SEARCH_CACHE_TTL = 300.0  # 5 minutes
_SEARCH_CACHE_LOCK = threading.Lock()

# In-flight dedup: maps query string → asyncio.Future[payload].
# Concurrent requests for the same query share one upstream call.
_SEARCH_INFLIGHT: dict[str, Any] = {}  # Any = asyncio.Future[dict]


def _get_cached_search(query: str) -> dict[str, Any] | None:
    with _SEARCH_CACHE_LOCK:
        entry = _SEARCH_CACHE.get(query)
        if entry:
            if time.time() - entry["timestamp"] < _SEARCH_CACHE_TTL:
                return entry["payload"]
            else:
                del _SEARCH_CACHE[query]
    return None


def _set_cached_search(query: str, payload: dict[str, Any]):
    with _SEARCH_CACHE_LOCK:
        _SEARCH_CACHE[query] = {
            "timestamp": time.time(),
            "payload": payload,
        }


async def _get_or_fetch_search(query: str, limit: int) -> dict[str, Any]:
    """Deduplicated search fetch: concurrent requests for the same query share
    one upstream call and all receive the same cached result."""
    cached = _get_cached_search(query)
    if cached is not None:
        return cached

    # If another coroutine is already fetching this query, wait for it.
    # asyncio is single-threaded within the event loop, so the check-and-insert
    # below has no race condition (no await between the check and the insert).
    if query in _SEARCH_INFLIGHT:
        try:
            return await asyncio.shield(_SEARCH_INFLIGHT[query])
        except Exception:
            # Original fetch failed; try cache (may have been populated) or reraise.
            cached = _get_cached_search(query)
            if cached is not None:
                return cached
            raise

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    _SEARCH_INFLIGHT[query] = fut
    try:
        payload = await asyncio.to_thread(_grounding_search, query, limit)
        _set_cached_search(query, payload)
        fut.set_result(payload)
        return payload
    except asyncio.CancelledError:
        if not fut.done():
            fut.cancel()
        raise
    except Exception as exc:
        if not fut.done():
            fut.set_exception(exc)
        raise
    finally:
        _SEARCH_INFLIGHT.pop(query, None)


@app.get("/search")
async def searxng_search(
    q: str = Query("", description="search query"),
    format: str = Query("json"),
    pageno: int = Query(1),
    limit: int = Query(10, ge=1, le=20),
):
    """SearXNG-compatible JSON search endpoint (Gemini Google-Search grounded)."""
    query = (q or "").strip()
    empty = {"query": query, "number_of_results": 0, "results": [], "answers": []}
    if not query or pageno > 1:
        # Grounding has no pagination; page >1 simply has no further results.
        return JSONResponse(empty)
    cached = _get_cached_search(query)
    if cached is not None:
        log.info("SearXNG search cache hit for: %r", query)
        return JSONResponse(cached)
    try:
        # Hard 13s ceiling — return a valid empty body BEFORE Hermes' own 15s
        # SearXNG client timeout fires (which would surface as a tool error).
        # _get_or_fetch_search deduplicates concurrent identical queries so
        # only one upstream grounding call is made per unique query string.
        payload = await asyncio.wait_for(
            _get_or_fetch_search(query, limit), timeout=13.0
        )
        return JSONResponse(payload)
    except asyncio.TimeoutError:
        log.warning("grounding search timed out (>13s) for %r", query)
        return JSONResponse(empty)
    except Exception as exc:  # noqa: BLE001
        log.warning("grounding search failed for %r: %s", query, exc)
        # Return an empty-but-valid SearXNG body so Hermes degrades gracefully.
        return JSONResponse(empty)


# ---------------------------------------------------------------------------
# Image generation (OpenAI-compatible /v1/images/generations)
# ---------------------------------------------------------------------------
class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "gemini-3.1-flash-image"
    n: int = Field(default=1, ge=1, le=1)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    user: str | None = None

    model_config = {"extra": "allow"}


@app.post("/v1/images/generations")
async def create_image(req: ImageGenerationRequest):
    """Generate an image via Antigravity (Gemini native image output).

    Returns OpenAI-compatible b64_json response that the chatbot adapter
    saves to a temp file and sends to KakaoTalk.
    """
    import base64
    import tempfile
    from pathlib import Path

    client = _get_client()

    # Map size → aspect_ratio + image_size for Antigravity
    size_map = {
        "1024x1024": ("square", "1K"),
        "1792x1024": ("landscape", "1K"),
        "1024x1792": ("portrait", "1K"),
    }
    aspect_ratio, image_size = size_map.get(req.size, ("square", "1K"))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = await asyncio.to_thread(
                client.generate_image,
                prompt=req.prompt,
                output_dir=Path(tmpdir),
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            )
            image_bytes = output_path.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
    except Exception as exc:
        log.exception("Image generation failed")
        raise HTTPException(
            status_code=502,
            detail=f"Image generation failed: {exc}",
        )

    return JSONResponse({
        "created": int(time.time()),
        "data": [
            {
                "b64_json": image_b64,
                "revised_prompt": req.prompt,
            }
        ],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "antigravity_proxy:app",
        host="0.0.0.0",
        port=8765,
        log_level="info",
        # No reload in production-like proxy
    )
