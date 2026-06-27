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
import hmac
import hashlib
import json
import logging
import mimetypes
import os
import re
import secrets
import sqlite3
import time
import tempfile
import uuid
from urllib.parse import parse_qsl, urlencode, urlparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import sys
from email import policy
from email.parser import BytesParser

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

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
    "gemini-flash-latest": "gemini-3-flash-agent",
    "gemini-pro-latest": "gemini-pro-agent",
    "gemini-3-flash-latest": "gemini-3-flash-agent",
    "gemini-3.5-flash-latest": "gemini-3-flash-agent",
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
    "gemini-3-pro": "gemini-pro-agent",
    "gemini-3-pro-latest": "gemini-pro-agent",
    "gemini-3-flash-image": "gemini-3.1-flash-image",
    "gemini-image-latest": "gemini-3.1-flash-image",
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


def _gemini_resource_model_id(model_name: str) -> str:
    key = model_name.strip().strip("/")
    if key.startswith("models/"):
        key = key[len("models/"):]
    return key.replace("%20", " ")


def _is_gemini_video_model_id(model_id: str) -> bool:
    return _gemini_resource_model_id(model_id).lower().startswith("veo-")


def _gemini_video_model_resource(model_id: str) -> dict[str, Any]:
    normalized_id = _gemini_resource_model_id(model_id)
    return {
        "name": "models/" + normalized_id,
        "version": normalized_id,
        "displayName": normalized_id,
        "description": "Gemini-compatible video generation model placeholder",
        "inputTokenLimit": 32768,
        "outputTokenLimit": 0,
        "supportedGenerationMethods": ["predictLongRunning", "generateVideos"],
        "capabilities": {
            "chat": False,
            "tools": False,
            "vision": False,
            "streaming": False,
            "imageGeneration": False,
            "videoGeneration": True,
        },
    }


def _gemini_model_resource(model: dict[str, Any]) -> dict[str, Any]:
    caps = _model_capabilities(model)
    if caps["image_generation"]:
        methods = ["generateContent", "generateImages", "predict", "predictLongRunning"]
        input_limit = 32768
        output_limit = 8192
    elif caps["internal"]:
        methods = []
        input_limit = 0
        output_limit = 0
    else:
        methods = [
            "generateContent",
            "streamGenerateContent",
            "countTokens",
            "computeTokens",
            "embedContent",
            "batchEmbedContents",
            "asyncBatchEmbedContent",
            "batchGenerateContent",
            "predict",
            "predictLongRunning",
            "generateText",
            "generateMessage",
            "generateAnswer",
            "embedText",
            "batchEmbedText",
            "countTextTokens",
            "countMessageTokens",
        ]
        input_limit = 1048576
        output_limit = 65536
    return {
        "name": _gemini_model_name(model),
        "version": _gemini_model_id(model),
        "displayName": str(model.get("id") or _gemini_model_id(model)),
        "description": "Antigravity-backed Gemini-compatible model",
        "inputTokenLimit": input_limit,
        "outputTokenLimit": output_limit,
        "supportedGenerationMethods": methods,
        "temperature": 1.0,
        "topP": 0.95,
        "topK": 64,
        "capabilities": {
            "chat": caps["chat"],
            "tools": caps["tools"],
            "vision": caps["vision"],
            "streaming": caps["streaming"],
            "imageGeneration": caps["image_generation"],
        },
    }


def _resolve_gemini_model(model_name: str) -> dict[str, Any]:
    decoded = _gemini_resource_model_id(model_name)
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
    "processing_options": "processingOptions",
    "processingOptions": "processingOptions",
    "response_mime_type": "responseMimeType",
    "response_schema": "responseSchema",
    "response_json_schema": "responseJsonSchema",
    "_response_json_schema": "responseJsonSchema",
    "_responseJsonSchema": "responseJsonSchema",
    "response_format": "responseFormat",
    "responseFormat": "responseFormat",
    "enable_enhanced_civic_answers": "enableEnhancedCivicAnswers",
    "generate_content_request": "generateContentRequest",
    "generateContentRequest": "generateContentRequest",
    "http_options": "httpOptions",
    "httpOptions": "httpOptions",
    "request_options": "requestOptions",
    "requestOptions": "requestOptions",
    "provider_options": "providerOptions",
    "providerOptions": "providerOptions",
    "provider_metadata": "providerMetadata",
    "providerMetadata": "providerMetadata",
    "api_version": "apiVersion",
    "apiVersion": "apiVersion",
    "base_url": "baseUrl",
    "baseUrl": "baseUrl",
    "json_schema": "jsonSchema",
    "jsonSchema": "jsonSchema",
    "max_output_tokens": "maxOutputTokens",
    "candidate_count": "candidateCount",
    "stop_sequences": "stopSequences",
    "top_p": "topP",
    "top_k": "topK",
    "response_logprobs": "responseLogprobs",
    "logprobs": "logprobs",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
    "thinking_config": "thinkingConfig",
    "thinking_budget": "thinkingBudget",
    "include_thoughts": "includeThoughts",
    "response_modalities": "responseModalities",
    "media_resolution": "mediaResolution",
    "audio_timestamp": "audioTimestamp",
    "audioTimestamp": "audioTimestamp",
    "start_offset": "startOffset",
    "startOffset": "startOffset",
    "end_offset": "endOffset",
    "endOffset": "endOffset",
    "image_config": "imageConfig",
    "aspect_ratio": "aspectRatio",
    "image_size": "imageSize",
    "number_of_images": "numberOfImages",
    "sample_count": "sampleCount",
    "speech_config": "speechConfig",
    "routing_config": "routingConfig",
    "embedding_config": "embedContentConfig",
    "embeddingConfig": "embedContentConfig",
    "embed_content_config": "embedContentConfig",
    "embedContentConfig": "embedContentConfig",
    "output_dimensionality": "outputDimensionality",
    "task_type": "taskType",
    "auto_truncate": "autoTruncate",
    "document_ocr": "documentOcr",
    "audio_track_extraction": "audioTrackExtraction",
    "display_name": "displayName",
    "tuned_model": "tunedModel",
    "tuned_model_id": "tunedModelId",
    "tuned_model_source": "tunedModelSource",
    "tuning_task": "tuningTask",
    "reader_project_numbers": "readerProjectNumbers",
    "training_data": "trainingData",
    "validation_data": "validationData",
    "document_id": "documentId",
    "documentId": "documentId",
    "chunk_id": "chunkId",
    "chunkId": "chunkId",
    "service_tier": "serviceTier",
    "serviceTier": "serviceTier",
    "generate_content_batch": "generateContentBatch",
    "embed_content_batch": "embedContentBatch",
    "input_config": "inputConfig",
    "output_config": "outputConfig",
    "instances_format": "instancesFormat",
    "predictions_format": "predictionsFormat",
    "epoch_count": "epochCount",
    "batch_size": "batchSize",
    "target_uri": "targetUri",
    "update_mask": "updateMask",
    "event_types": "eventTypes",
    "subscribed_events": "subscribedEvents",
    "chunking_config": "chunkingConfig",
    "embedding_model": "embeddingModel",
    "white_space_config": "whiteSpaceConfig",
    "max_tokens_per_chunk": "maxTokensPerChunk",
    "max_overlap_tokens": "maxOverlapTokens",
    "custom_metadata": "customMetadata",
    "permission": "permission",
    "grantee_type": "granteeType",
    "email_address": "emailAddress",
    "webhook_secret": "webhookSecret",
    "signing_secrets": "signingSecrets",
    "new_signing_secret": "newSigningSecret",
    "function_declarations": "functionDeclarations",
    "function_declaration": "functionDeclaration",
    "functionDeclaration": "functionDeclaration",
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
    "size_bytes": "sizeBytes",
    "string_value": "stringValue",
    "stringValue": "stringValue",
    "numeric_value": "numericValue",
    "numericValue": "numericValue",
    "download_uri": "downloadUri",
    "expiration_time": "expirationTime",
    "sha256_hash": "sha256Hash",
    "file_name": "fileName",
    "file_uri": "fileUri",
    "fileUri": "fileUri",
    "image_url": "imageUrl",
    "image_bytes": "imageBytes",
    "video_metadata": "videoMetadata",
    "function_call": "functionCall",
    "functionCall": "functionCall",
    "function_response": "functionResponse",
    "functionResponse": "functionResponse",
    "thought_signature": "thoughtSignature",
    "thoughtSignature": "thoughtSignature",
    "executable_code": "executableCode",
    "executableCode": "executableCode",
    "code_execution_result": "codeExecutionResult",
    "codeExecutionResult": "codeExecutionResult",
    "harm_category": "category",
    "harmCategory": "category",
    "harm_block_threshold": "threshold",
    "harmBlockThreshold": "threshold",
}

_GEMINI_GENERATE_CONFIG_TOP_LEVEL_KEYS = {
    "systemInstruction",
    "safetySettings",
    "tools",
    "toolConfig",
    "cachedContent",
    "processingOptions",
    "labels",
    "serviceTier",
    "store",
}

_GEMINI_GENERATION_CONFIG_KEYS = {
    "temperature",
    "topP",
    "topK",
    "candidateCount",
    "maxOutputTokens",
    "stopSequences",
    "responseMimeType",
    "responseSchema",
    "responseJsonSchema",
    "responseLogprobs",
    "logprobs",
    "presencePenalty",
    "frequencyPenalty",
    "seed",
    "thinkingConfig",
    "responseModalities",
    "mediaResolution",
    "audioTimestamp",
    "imageConfig",
    "speechConfig",
    "routingConfig",
    "enableEnhancedCivicAnswers",
}

_GEMINI_SDK_TRANSPORT_KEYS = {
    "httpOptions",
    "requestOptions",
    "providerOptions",
    "providerMetadata",
    "google",
    "apiVersion",
    "baseUrl",
    "headers",
    "timeout",
}


def _gemini_string_list(value: Any) -> Any:
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    return value


def _gemini_int_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return value


def _gemini_float_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def _gemini_bool_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    return value


def _gemini_service_tier_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "unspecified": "unspecified",
        "service_tier_unspecified": "unspecified",
        "standard": "standard",
        "service_tier_standard": "standard",
        "flex": "flex",
        "service_tier_flex": "flex",
        "priority": "priority",
        "service_tier_priority": "priority",
    }
    return aliases.get(normalized, value)


def _gemini_media_resolution_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "unspecified": "MEDIA_RESOLUTION_UNSPECIFIED",
        "media_resolution_unspecified": "MEDIA_RESOLUTION_UNSPECIFIED",
        "low": "MEDIA_RESOLUTION_LOW",
        "media_resolution_low": "MEDIA_RESOLUTION_LOW",
        "medium": "MEDIA_RESOLUTION_MEDIUM",
        "media_resolution_medium": "MEDIA_RESOLUTION_MEDIUM",
        "high": "MEDIA_RESOLUTION_HIGH",
        "media_resolution_high": "MEDIA_RESOLUTION_HIGH",
    }
    return aliases.get(normalized, value)


def _gemini_response_modality_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "unspecified": "MODALITY_UNSPECIFIED",
        "modality_unspecified": "MODALITY_UNSPECIFIED",
        "text": "TEXT",
        "image": "IMAGE",
        "audio": "AUDIO",
    }
    return aliases.get(normalized, value)


def _gemini_response_modalities_value(value: Any) -> Any:
    modalities = _gemini_string_list(value)
    if isinstance(modalities, list):
        return [_gemini_response_modality_value(item) for item in modalities]
    return modalities


def _gemini_normalize_generation_config(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = _gemini_normalize_request(value)
    response_format = out.pop("responseFormat", None)
    if response_format is not None:
        _gemini_apply_response_format_to_generation_config(out, response_format)
    if "stopSequences" in out:
        out["stopSequences"] = _gemini_string_list(out["stopSequences"])
    if "responseModalities" in out:
        out["responseModalities"] = _gemini_response_modalities_value(out["responseModalities"])
    for key in ("topK", "candidateCount", "maxOutputTokens", "logprobs", "seed"):
        if key in out:
            out[key] = _gemini_int_value(out[key])
    for key in ("temperature", "topP", "presencePenalty", "frequencyPenalty"):
        if key in out:
            out[key] = _gemini_float_value(out[key])
    for key in ("responseLogprobs", "enableEnhancedCivicAnswers", "audioTimestamp"):
        if key in out:
            out[key] = _gemini_bool_value(out[key])
    if "mediaResolution" in out:
        out["mediaResolution"] = _gemini_media_resolution_value(out["mediaResolution"])
    if isinstance(out.get("responseSchema"), dict):
        out["responseSchema"] = _sanitize_schema(dict(out["responseSchema"]))
    if isinstance(out.get("responseJsonSchema"), dict):
        out["responseJsonSchema"] = _sanitize_schema(dict(out["responseJsonSchema"]))
    if isinstance(out.get("thinkingConfig"), dict):
        thinking = _gemini_normalize_request(out["thinkingConfig"])
        if "thinkingBudget" in thinking:
            thinking["thinkingBudget"] = _gemini_int_value(thinking["thinkingBudget"])
        if "includeThoughts" in thinking:
            thinking["includeThoughts"] = _gemini_bool_value(thinking["includeThoughts"])
        out["thinkingConfig"] = thinking
    return out


def _gemini_apply_response_format_to_generation_config(gen: dict[str, Any], fmt: Any) -> None:
    if isinstance(fmt, dict):
        nested_json_schema = fmt.get("jsonSchema") if isinstance(fmt.get("jsonSchema"), dict) else None
        fmt_type = str(fmt.get("type") or "").strip().lower()
        mime = (
            fmt.get("mimeType")
            or fmt.get("responseMimeType")
            or (nested_json_schema or {}).get("mimeType")
            or (nested_json_schema or {}).get("responseMimeType")
        )
        if mime:
            gen["responseMimeType"] = mime
        elif fmt_type in {"json_object", "json_schema", "application/json"} or nested_json_schema:
            gen["responseMimeType"] = "application/json"

        schema = (
            fmt.get("schema")
            or fmt.get("responseSchema")
            or (nested_json_schema or {}).get("schema")
            or (nested_json_schema or {}).get("responseSchema")
        )
        if isinstance(schema, dict):
            gen["responseSchema"] = _sanitize_schema(dict(schema))
        json_schema = (
            fmt.get("responseJsonSchema")
            or fmt.get("_responseJsonSchema")
            or (nested_json_schema or {}).get("responseJsonSchema")
            or (nested_json_schema or {}).get("_responseJsonSchema")
        )
        if isinstance(json_schema, dict):
            gen["responseJsonSchema"] = _sanitize_schema(dict(json_schema))
    elif isinstance(fmt, str) and fmt.strip():
        fmt_type = fmt.strip().lower()
        if "/" in fmt_type:
            gen["responseMimeType"] = fmt.strip()
        elif fmt_type in {"json", "json_object", "json_schema"}:
            gen["responseMimeType"] = "application/json"


def _gemini_normalize_embedding_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out = _gemini_normalize_request(value)
    if "outputDimensionality" in out:
        out["outputDimensionality"] = _gemini_int_value(out["outputDimensionality"])
    for key in ("autoTruncate", "documentOcr", "audioTrackExtraction"):
        if key in out:
            out[key] = _gemini_bool_value(out[key])
    task_type = out.get("taskType")
    if isinstance(task_type, str):
        normalized = task_type.strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "UNSPECIFIED": "TASK_TYPE_UNSPECIFIED",
            "TASK_TYPE_UNSPECIFIED": "TASK_TYPE_UNSPECIFIED",
            "RETRIEVAL_QUERY": "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT": "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY": "SEMANTIC_SIMILARITY",
            "CLASSIFICATION": "CLASSIFICATION",
            "CLUSTERING": "CLUSTERING",
            "QUESTION_ANSWERING": "QUESTION_ANSWERING",
            "FACT_VERIFICATION": "FACT_VERIFICATION",
            "CODE_RETRIEVAL_QUERY": "CODE_RETRIEVAL_QUERY",
        }
        out["taskType"] = aliases.get(normalized, task_type)
    return out


def _gemini_normalize_function_calling_config(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = dict(value)
    mode = out.get("mode")
    if isinstance(mode, str):
        normalized = mode.strip().upper().replace("-", "_")
        if normalized.startswith("MODE_"):
            normalized = normalized[len("MODE_"):]
        if normalized in {"REQUIRED", "FORCED", "FORCE"}:
            normalized = "ANY"
        if normalized in {"AUTO", "ANY", "NONE", "VALIDATED", "UNSPECIFIED"}:
            out["mode"] = "MODE_UNSPECIFIED" if normalized == "UNSPECIFIED" else normalized
    names = out.get("allowedFunctionNames")
    if names is None and "allowed_function_names" in out:
        names = out.pop("allowed_function_names")
    if isinstance(names, str):
        out["allowedFunctionNames"] = [names]
    elif isinstance(names, tuple):
        out["allowedFunctionNames"] = list(names)
    elif isinstance(names, list):
        out["allowedFunctionNames"] = names
    return out


def _gemini_normalize_request(value: Any) -> Any:
    if isinstance(value, list):
        return [_gemini_normalize_request(item) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, child in value.items():
        mapped = _GEMINI_KEY_ALIASES.get(str(key), key)
        out[mapped] = _gemini_normalize_request(child)
    if isinstance(out.get("functionCallingConfig"), dict):
        out["functionCallingConfig"] = _gemini_normalize_function_calling_config(out["functionCallingConfig"])
    if isinstance(out.get("toolConfig"), dict) and isinstance(out["toolConfig"].get("functionCallingConfig"), dict):
        out["toolConfig"]["functionCallingConfig"] = _gemini_normalize_function_calling_config(
            out["toolConfig"]["functionCallingConfig"]
        )
    return out


def _gemini_provider_google_config(body: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("providerOptions", "providerMetadata"):
        options = body.get(key)
        if not isinstance(options, dict):
            continue
        google = options.get("google")
        if isinstance(google, dict):
            merged.update(_gemini_normalize_request(google))
    google = body.get("google")
    if isinstance(google, dict):
        merged.update(_gemini_normalize_request(google))
    return merged


def _gemini_apply_generate_config(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        return body
    provider_config = _gemini_provider_google_config(body)
    config = body.get("config")
    out = {key: value for key, value in body.items() if key not in _GEMINI_SDK_TRANSPORT_KEYS and key != "config"}
    if not provider_config and not isinstance(config, dict):
        return out
    merged_config = dict(provider_config)
    if isinstance(config, dict):
        merged_config.update(_gemini_normalize_request(config))
    gen = _gemini_normalize_generation_config(out.get("generationConfig") or {}) if isinstance(out.get("generationConfig"), dict) else {}
    for key, value in merged_config.items():
        if value is None:
            continue
        if key in _GEMINI_SDK_TRANSPORT_KEYS:
            continue
        if key in _GEMINI_GENERATE_CONFIG_TOP_LEVEL_KEYS:
            out.setdefault(key, value)
        elif key == "responseFormat":
            out.setdefault("responseFormat", value)
        elif key in _GEMINI_GENERATION_CONFIG_KEYS:
            gen.setdefault(key, value)
        else:
            out.setdefault(key, value)
    if gen:
        out["generationConfig"] = _gemini_normalize_generation_config(gen)
    return out


def _gemini_apply_response_format(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        return body
    out = dict(body)
    gen = _gemini_normalize_generation_config(out.get("generationConfig") or {}) if isinstance(out.get("generationConfig"), dict) else {}
    fmt = out.pop("responseFormat", None)

    if out.get("responseMimeType") is not None:
        gen["responseMimeType"] = out.pop("responseMimeType")
    if out.get("responseSchema") is not None:
        schema = out.pop("responseSchema")
        gen["responseSchema"] = _sanitize_schema(dict(schema)) if isinstance(schema, dict) else schema
    if out.get("responseJsonSchema") is not None:
        schema = out.pop("responseJsonSchema")
        gen["responseJsonSchema"] = _sanitize_schema(dict(schema)) if isinstance(schema, dict) else schema

    _gemini_apply_response_format_to_generation_config(gen, fmt)

    if gen:
        out["generationConfig"] = _gemini_normalize_generation_config(gen)
    return out


def _gemini_unwrap_response(data: dict[str, Any]) -> dict[str, Any]:
    response = data.get("response")
    if isinstance(response, dict):
        return response
    return data


def _gemini_error_payload(
    message: Any,
    *,
    status_code: int,
    status: str | None = None,
    field: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    if not isinstance(message, str):
        message = json.dumps(message, ensure_ascii=False)
    error_status = status or _gemini_status_for_http(status_code)
    error: dict[str, Any] = {"code": status_code, "message": message, "status": error_status}
    details: list[dict[str, Any]] = []
    if error_status == "INVALID_ARGUMENT" or field:
        violation: dict[str, Any] = {"description": message}
        if field:
            violation["field"] = field
        details.append({
            "@type": "type.googleapis.com/google.rpc.BadRequest",
            "fieldViolations": [violation],
        })
    if reason or error_status in {
        "UNIMPLEMENTED",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "RESOURCE_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "UNAVAILABLE",
        "INTERNAL",
    }:
        details.append({
            "@type": "type.googleapis.com/google.rpc.ErrorInfo",
            "reason": reason or error_status,
            "domain": "generativelanguage.googleapis.com",
        })
    if details:
        error["details"] = details
    return {"error": error}


def _gemini_error_response(
    message: Any,
    *,
    status_code: int,
    status: str | None = None,
    field: str | None = None,
    reason: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        _gemini_error_payload(message, status_code=status_code, status=status, field=field, reason=reason),
        status_code=status_code,
    )


def _gemini_status_for_http(status_code: int) -> str:
    mapping = {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        405: "UNIMPLEMENTED",
        409: "ABORTED",
        412: "FAILED_PRECONDITION",
        422: "INVALID_ARGUMENT",
        429: "RESOURCE_EXHAUSTED",
        499: "CANCELLED",
        500: "INTERNAL",
        501: "UNIMPLEMENTED",
        502: "UNAVAILABLE",
        503: "UNAVAILABLE",
        504: "DEADLINE_EXCEEDED",
    }
    return mapping.get(status_code, "INTERNAL" if status_code >= 500 else "INVALID_ARGUMENT")


def _gemini_inline_data_part(value: dict[str, Any]) -> dict[str, Any] | None:
    data = value.get("data") or value.get("base64") or value.get("bytesBase64Encoded")
    mime_type = value.get("mimeType") or value.get("mime_type")
    if data is None or not mime_type:
        return None
    text_data = str(data)
    resolved_mime = str(mime_type)
    if text_data.startswith("data:"):
        header, _, text_data = text_data.partition(",")
        resolved_mime = header.split(";", 1)[0].removeprefix("data:") or resolved_mime
    return {"inlineData": {"mimeType": resolved_mime, "data": text_data}}


def _gemini_content_part(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        out = _gemini_normalize_request(value)
        if any(key in out for key in (
            "text",
            "inlineData",
            "fileData",
            "functionCall",
            "functionResponse",
            "executableCode",
            "codeExecutionResult",
        )):
            return out
        inline_part = _gemini_inline_data_part(out)
        if inline_part:
            if out.get("thoughtSignature") is not None:
                inline_part["thoughtSignature"] = out["thoughtSignature"]
            if out.get("thought") is not None:
                inline_part["thought"] = out["thought"]
            return inline_part
        file_uri = out.get("uri") or out.get("fileUri") or out.get("name")
        if file_uri and (out.get("mimeType") or str(file_uri).startswith("files/")):
            file_data: dict[str, Any] = {"fileUri": str(file_uri)}
            mime_type = out.get("mimeType")
            if mime_type:
                file_data["mimeType"] = str(mime_type)
            part: dict[str, Any] = {"fileData": file_data}
            if out.get("thoughtSignature") is not None:
                part["thoughtSignature"] = out["thoughtSignature"]
            if out.get("thought") is not None:
                part["thought"] = out["thought"]
            return part
        if out.get("type") in {"text", "input_text", "output_text"} and out.get("text") is not None:
            return {"text": str(out["text"])}
        if isinstance(out.get("content"), str):
            return {"text": out["content"]}
    return {"text": "" if value is None else str(value)}


def _gemini_content_from_value(value: Any, *, default_role: str = "user") -> dict[str, Any]:
    if isinstance(value, dict):
        if "parts" in value:
            out = dict(value)
            parts = out.get("parts")
            if isinstance(parts, str):
                out["parts"] = [{"text": parts}]
            elif isinstance(parts, list):
                out["parts"] = [_gemini_content_part(part) for part in parts]
            elif parts is None:
                out["parts"] = []
            else:
                out["parts"] = [_gemini_content_part(parts)]
            out["role"] = str(out.get("role") or default_role)
            return out
        if "role" in value and "content" in value:
            content = _gemini_content_from_value(value.get("content"), default_role=str(value.get("role") or default_role))
            content["role"] = str(value.get("role") or content.get("role") or default_role)
            return content
        if any(key in value for key in ("text", "inlineData", "fileData", "functionCall", "functionResponse")):
            return {"role": default_role, "parts": [_gemini_content_part(value)]}
    if isinstance(value, list):
        return {"role": default_role, "parts": [_gemini_content_part(part) for part in value]}
    return {"role": default_role, "parts": [{"text": "" if value is None else str(value)}]}


def _gemini_normalize_contents(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_gemini_content_from_value(item) for item in value]
    return [_gemini_content_from_value(value)]


def _gemini_normalize_system_instruction(value: Any) -> Any:
    if value is None:
        return None
    return _gemini_content_from_value(value, default_role="system")


_GEMINI_HARM_CATEGORY_ALIASES = {
    "HARASSMENT": "HARM_CATEGORY_HARASSMENT",
    "HATE_SPEECH": "HARM_CATEGORY_HATE_SPEECH",
    "HATE": "HARM_CATEGORY_HATE_SPEECH",
    "SEXUALLY_EXPLICIT": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "SEXUAL": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "DANGEROUS_CONTENT": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "DANGEROUS": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "CIVIC_INTEGRITY": "HARM_CATEGORY_CIVIC_INTEGRITY",
    "CIVIC": "HARM_CATEGORY_CIVIC_INTEGRITY",
}

_GEMINI_HARM_THRESHOLD_ALIASES = {
    "OFF": "OFF",
    "NONE": "BLOCK_NONE",
    "BLOCK_NONE": "BLOCK_NONE",
    "ONLY_HIGH": "BLOCK_ONLY_HIGH",
    "BLOCK_ONLY_HIGH": "BLOCK_ONLY_HIGH",
    "MEDIUM_AND_ABOVE": "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE": "BLOCK_MEDIUM_AND_ABOVE",
    "LOW_AND_ABOVE": "BLOCK_LOW_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE": "BLOCK_LOW_AND_ABOVE",
}


def _gemini_enum_alias(value: Any, aliases: dict[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    key = value.strip().upper().replace("-", "_").replace(" ", "_")
    return aliases.get(key, value)


def _gemini_normalize_safety_settings(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict) and not any(key in value for key in ("category", "threshold", "method")):
        value = [{"category": key, "threshold": threshold} for key, threshold in value.items()]
    items = value if isinstance(value, list) else [value]
    settings: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out = dict(item)
            if "category" in out:
                out["category"] = _gemini_enum_alias(out["category"], _GEMINI_HARM_CATEGORY_ALIASES)
            if "threshold" in out:
                out["threshold"] = _gemini_enum_alias(out["threshold"], _GEMINI_HARM_THRESHOLD_ALIASES)
            method = out.get("method") or out.pop("harmBlockMethod", None) or out.pop("harm_block_method", None)
            if method is not None:
                out["method"] = _gemini_enum_alias(method, {
                    "SEVERITY": "SEVERITY",
                    "PROBABILITY": "PROBABILITY",
                })
            settings.append(out)
    return settings


def _gemini_normalize_tool_config(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = dict(value)
    if "functionCallingConfig" not in out:
        fc_keys = {"mode", "allowedFunctionNames"}
        if any(key in out for key in fc_keys):
            out = {"functionCallingConfig": {key: out.pop(key) for key in list(out.keys()) if key in fc_keys}, **out}
    if isinstance(out.get("functionCallingConfig"), dict):
        out["functionCallingConfig"] = _gemini_normalize_function_calling_config(out["functionCallingConfig"])
    return out


def _gemini_normalize_function_declaration(decl: dict[str, Any]) -> dict[str, Any]:
    out = dict(decl)
    params = out.pop("parametersJsonSchema", None) or out.pop("parameters_json_schema", None)
    if params is not None and "parameters" not in out:
        out["parameters"] = params
    response = out.pop("responseJsonSchema", None) or out.pop("response_json_schema", None)
    if response is not None and "response" not in out:
        out["response"] = response
    for key in ("parameters", "response"):
        if isinstance(out.get(key), dict):
            out[key] = _sanitize_schema(out[key])
    return out


def _gemini_normalize_tools_value(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    tools: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item = _gemini_normalize_request(item)
        if "functionDeclaration" in item and "functionDeclarations" not in item:
            item = dict(item)
            item["functionDeclarations"] = item.pop("functionDeclaration")
        if "functionDeclarations" in item:
            decls = item.get("functionDeclarations")
            if isinstance(decls, dict):
                item = dict(item)
                item["functionDeclarations"] = [decls]
            if isinstance(item.get("functionDeclarations"), list):
                item = dict(item)
                item["functionDeclarations"] = [
                    _gemini_normalize_function_declaration(decl) if isinstance(decl, dict) else decl
                    for decl in item["functionDeclarations"]
                ]
            tools.append(item)
            continue
        if "googleSearchRetrieval" in item and "googleSearch" not in item and "google_search" not in item:
            item = dict(item)
            item["googleSearch"] = item.pop("googleSearchRetrieval")
        if "googleSearch" in item:
            item = dict(item)
            item["google_search"] = item.pop("googleSearch")
        if any(key in item for key in ("google_search", "codeExecution", "urlContext", "file_search", "fileSearchRetrieval")):
            tools.append(item)
            continue
        if item.get("name") or item.get("function") or item.get("declaration"):
            fn = item.get("function") if isinstance(item.get("function"), dict) else item.get("declaration")
            if not isinstance(fn, dict):
                fn = item
            if isinstance(fn, dict) and fn.get("name"):
                decl: dict[str, Any] = {"name": fn["name"]}
                if fn.get("description"):
                    decl["description"] = fn["description"]
                for key in ("parameters", "parametersJsonSchema", "response", "responseJsonSchema"):
                    if key in fn:
                        decl[key] = fn[key]
                decl = _gemini_normalize_function_declaration(decl)
                tools.append({"functionDeclarations": [decl]})
                continue
        tools.append(item)
    return tools


def _gemini_normalize_generate_body(body: dict[str, Any]) -> dict[str, Any]:
    out = dict(body)
    tool_choice = out.pop("toolChoice", None)
    if tool_choice is None:
        tool_choice = out.pop("tool_choice", None)
    function_declarations = out.pop("functionDeclarations", None)
    if function_declarations is None:
        function_declarations = out.pop("functionDeclaration", None)
    if function_declarations is not None:
        existing = _gemini_normalize_tools_value(out.get("tools"))
        if isinstance(function_declarations, dict):
            function_declarations = [function_declarations]
        if isinstance(function_declarations, list):
            existing.append({"functionDeclarations": [
                _gemini_normalize_function_declaration(decl) if isinstance(decl, dict) else decl
                for decl in function_declarations
            ]})
        if existing:
            out["tools"] = existing
    elif "tools" in out:
        tools = _gemini_normalize_tools_value(out.get("tools"))
        if tools:
            out["tools"] = tools
        else:
            out.pop("tools", None)
    if "contents" in out:
        out["contents"] = _gemini_normalize_contents(out.get("contents"))
    if "systemInstruction" in out:
        out["systemInstruction"] = _gemini_normalize_system_instruction(out.get("systemInstruction"))
    if "safetySettings" in out:
        safety_settings = _gemini_normalize_safety_settings(out.get("safetySettings"))
        if safety_settings:
            out["safetySettings"] = safety_settings
        else:
            out.pop("safetySettings", None)
    if "toolConfig" in out:
        out["toolConfig"] = _gemini_normalize_tool_config(out.get("toolConfig"))
    if tool_choice is not None:
        choice_config = _tool_choice_to_gemini(tool_choice, out.get("tools") if isinstance(out.get("tools"), list) else None)
        if choice_config:
            existing_tool_config = out.get("toolConfig") if isinstance(out.get("toolConfig"), dict) else {}
            merged_tool_config = {**existing_tool_config, **choice_config}
            out["toolConfig"] = _gemini_normalize_tool_config(merged_tool_config)
    if "store" in out:
        out["store"] = _gemini_bool_value(out.get("store"))
    if "serviceTier" in out:
        out["serviceTier"] = _gemini_service_tier_value(out.get("serviceTier"))
    out.pop("processingOptions", None)
    return out


def _gemini_generate_request_payload(body: dict[str, Any]) -> dict[str, Any]:
    for key in ("generateContentRequest", "request"):
        value = body.get(key)
        if isinstance(value, dict):
            return value
    return body


def _gemini_count_tokens_request(body: dict[str, Any]) -> list[ChatMessage]:
    payload = _gemini_generate_request_payload(body)
    if isinstance(payload, dict):
        payload = _gemini_apply_generate_config(payload)
        payload = _gemini_normalize_generate_body(payload)
        payload = _gemini_apply_cached_content(payload)
        payload = _gemini_apply_file_search(payload)
    contents = payload.get("contents") or []
    if isinstance(contents, str):
        contents = [{"role": "user", "parts": [{"text": contents}]}]
    if isinstance(contents, dict):
        contents = [contents]
    messages: list[ChatMessage] = []
    for turn in contents if isinstance(contents, list) else []:
        if isinstance(turn, str):
            messages.append(ChatMessage(role="user", content=[{"text": turn}]))
            continue
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user")
        if role == "model":
            role = "assistant"
        parts = turn.get("parts") or []
        if isinstance(parts, str):
            parts = [{"text": parts}]
        messages.append(ChatMessage(role=role, content=parts))
    system_instruction = payload.get("systemInstruction")
    if isinstance(system_instruction, dict):
        messages.insert(0, ChatMessage(role="system", content=system_instruction.get("parts") or []))
    tool_context = {
        key: payload[key]
        for key in ("tools", "toolConfig")
        if payload.get(key) is not None
    }
    if tool_context:
        messages.insert(0, ChatMessage(
            role="system",
            content=[{"text": json.dumps(tool_context, ensure_ascii=False, sort_keys=True)}],
        ))
    return messages


def _gemini_cached_content_tokens(body: dict[str, Any]) -> int:
    payload = _gemini_generate_request_payload(body)
    if not isinstance(payload, dict):
        return 0
    cached_name = _gemini_cached_content_reference(payload.get("cachedContent"))
    if not cached_name:
        return 0
    meta = _gemini_load_cached_index().get(_gemini_cached_name(cached_name))
    if not meta:
        return 0
    usage = meta.get("usageMetadata") if isinstance(meta.get("usageMetadata"), dict) else {}
    cached_total = usage.get("totalTokenCount") or usage.get("totalTokens")
    if cached_total is not None:
        try:
            return max(0, int(cached_total))
        except (TypeError, ValueError):
            pass
    payload = meta.get("payload") if isinstance(meta.get("payload"), dict) else {}
    return _estimate_prompt_tokens(_gemini_count_tokens_request(payload))


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


def _gemini_embedding_config(body: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    provider_config = _gemini_provider_google_config(body)
    if provider_config:
        config.update(_gemini_normalize_embedding_config(provider_config))
    for key in ("config", "embedContentConfig"):
        value = body.get(key)
        if isinstance(value, dict):
            config.update(_gemini_normalize_embedding_config(value))
    for key in ("outputDimensionality", "taskType", "title", "autoTruncate", "documentOcr", "audioTrackExtraction"):
        if body.get(key) is not None:
            config[key] = body[key]
    return _gemini_normalize_embedding_config(config)


def _gemini_embedding_content_items(body: dict[str, Any]) -> list[Any]:
    if body.get("content") is not None:
        return [body["content"]]
    contents = body.get("contents")
    if contents is None:
        return [{}]
    if isinstance(contents, list):
        return contents
    return [contents]


def _gemini_embed_request_payload(body: dict[str, Any]) -> dict[str, Any]:
    for key in ("embedContentRequest", "request"):
        wrapped = body.get(key)
        if isinstance(wrapped, dict):
            out = dict(wrapped)
            for outer_key in (
                "model",
                "config",
                "embedContentConfig",
                "outputDimensionality",
                "taskType",
                "title",
                "autoTruncate",
                "documentOcr",
                "audioTrackExtraction",
                "providerOptions",
                "providerMetadata",
                "google",
            ):
                if outer_key in body and outer_key not in out:
                    out[outer_key] = body[outer_key]
            return out
    return body


def _gemini_embedding_content(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if isinstance(value.get("content"), dict):
            return _gemini_embedding_content(value["content"])
        if "parts" in value:
            return value
        if value.get("text") is not None:
            return {"parts": [{"text": str(value["text"])}]}
    if isinstance(value, list):
        return {"parts": [_gemini_embedding_part(item) for item in value]}
    return {"parts": [{"text": "" if value is None else str(value)}]}


def _gemini_embedding_part(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if "text" in value or "inlineData" in value or "fileData" in value:
            return value
        if value.get("type") in {"text", "input_text"} and value.get("text") is not None:
            return {"text": str(value["text"])}
    return {"text": "" if value is None else str(value)}


def _gemini_response_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if isinstance(part, dict) and part.get("text") is not None:
                texts.append(str(part["text"]))
    return "".join(texts)


def _gemini_normalize_usage_metadata(value: Any, *, request_body: dict[str, Any] | None, response: dict[str, Any]) -> dict[str, Any]:
    usage = dict(value) if isinstance(value, dict) else {}
    aliases = {
        "prompt_tokens": "promptTokenCount",
        "input_tokens": "promptTokenCount",
        "candidates_tokens": "candidatesTokenCount",
        "candidate_tokens": "candidatesTokenCount",
        "output_tokens": "candidatesTokenCount",
        "total_tokens": "totalTokenCount",
        "thoughts_tokens": "thoughtsTokenCount",
        "thoughts_token_count": "thoughtsTokenCount",
        "tool_use_prompt_tokens": "toolUsePromptTokenCount",
        "tool_use_prompt_token_count": "toolUsePromptTokenCount",
        "prompt_tokens_details": "promptTokensDetails",
        "candidates_tokens_details": "candidatesTokensDetails",
        "tool_use_prompt_tokens_details": "toolUsePromptTokensDetails",
        "thoughts_tokens_details": "thoughtsTokensDetails",
        "cache_tokens_details": "cacheTokensDetails",
        "cached_content_token_count": "cachedContentTokenCount",
        "traffic_type": "trafficType",
    }
    for old, new in aliases.items():
        if usage.get(new) is None and usage.get(old) is not None:
            usage[new] = usage[old]
        usage.pop(old, None)
    if usage.get("promptTokenCount") is None:
        usage["promptTokenCount"] = _estimate_tokens(request_body or {})
    if usage.get("candidatesTokenCount") is None:
        usage["candidatesTokenCount"] = _estimate_tokens(_gemini_response_text(response))
    if usage.get("totalTokenCount") is None:
        try:
            usage["totalTokenCount"] = sum(
                int(usage.get(key) or 0)
                for key in ("promptTokenCount", "candidatesTokenCount", "toolUsePromptTokenCount", "thoughtsTokenCount")
            )
        except (TypeError, ValueError):
            usage["totalTokenCount"] = _estimate_tokens(request_body or {}) + _estimate_tokens(_gemini_response_text(response))
    return usage


def _gemini_interaction_usage(value: Any) -> dict[str, Any]:
    usage = dict(value) if isinstance(value, dict) else {}
    aliases = {
        "promptTokenCount": "inputTokens",
        "candidatesTokenCount": "outputTokens",
        "totalTokenCount": "totalTokens",
        "cachedContentTokenCount": "cachedTokens",
        "thoughtsTokenCount": "reasoningTokens",
    }
    for old, new in aliases.items():
        if usage.get(new) is None and usage.get(old) is not None:
            usage[new] = usage[old]
    snake_aliases = (
        ("inputTokens", "input_tokens"),
        ("inputTokens", "total_input_tokens"),
        ("outputTokens", "output_tokens"),
        ("outputTokens", "total_output_tokens"),
        ("totalTokens", "total_tokens"),
        ("cachedTokens", "cached_tokens"),
        ("reasoningTokens", "reasoning_tokens"),
    )
    for old, new in snake_aliases:
        if usage.get(new) is None and usage.get(old) is not None:
            usage[new] = usage[old]
    legacy_snake_aliases = {
        "total_input_tokens": "input_tokens",
        "total_output_tokens": "output_tokens",
    }
    for old, new in legacy_snake_aliases.items():
        if usage.get(new) is None and usage.get(old) is not None:
            usage[new] = usage[old]

    def _modality_entries(details: Any, fallback_total: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if isinstance(details, dict):
            for modality, tokens in details.items():
                try:
                    entries.append({"modality": str(modality).upper(), "tokens": int(tokens or 0)})
                except (TypeError, ValueError):
                    continue
        elif isinstance(details, list):
            for item in details:
                if not isinstance(item, dict):
                    continue
                try:
                    entries.append({
                        "modality": str(item.get("modality") or "TEXT").upper(),
                        "tokens": int(item.get("tokens") or item.get("tokenCount") or item.get("token_count") or 0),
                    })
                except (TypeError, ValueError):
                    continue
        if entries:
            return entries
        try:
            total = int(fallback_total or 0)
        except (TypeError, ValueError):
            total = 0
        return [{"modality": "TEXT", "tokens": total}] if total > 0 else []

    if usage.get("input_tokens_by_modality") is None:
        entries = _modality_entries(
            usage.get("promptTokensDetails"),
            usage.get("total_input_tokens") or usage.get("inputTokens"),
        )
        if entries:
            usage["input_tokens_by_modality"] = entries
    if usage.get("output_tokens_by_modality") is None:
        entries = _modality_entries(
            usage.get("candidatesTokensDetails"),
            usage.get("total_output_tokens") or usage.get("outputTokens"),
        )
        if entries:
            usage["output_tokens_by_modality"] = entries
    return usage


def _gemini_interaction_resource(value: dict[str, Any]) -> dict[str, Any]:
    out = dict(value)
    out.setdefault("object", "interaction")
    if out.get("created_at") is None:
        out["created_at"] = out.get("created") or out.get("createTime")
    if out.get("updated_at") is None:
        out["updated_at"] = out.get("updated") or out.get("updateTime")
    if out.get("previous_interaction_id") is None and out.get("previousInteractionId") is not None:
        out["previous_interaction_id"] = out.get("previousInteractionId")
    if out.get("output_text") is None and out.get("outputText") is not None:
        out["output_text"] = out.get("outputText")
    out["usage"] = _gemini_interaction_usage(out.get("usage") or {})
    if isinstance(out.get("usageMetadata"), dict):
        out["usageMetadata"] = _gemini_normalize_usage_metadata(out["usageMetadata"], request_body=out.get("request"), response=out.get("output") or {})
    else:
        out["usageMetadata"] = _gemini_normalize_usage_metadata(out["usage"], request_body=out.get("request"), response=out.get("output") or {})
    return out


def _gemini_normalize_candidate(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(candidate)
    aliases = {
        "finish_reason": "finishReason",
        "finish_message": "finishMessage",
        "safety_ratings": "safetyRatings",
        "citation_metadata": "citationMetadata",
        "grounding_metadata": "groundingMetadata",
        "avg_logprobs": "avgLogprobs",
        "logprobs_result": "logprobsResult",
        "token_count": "tokenCount",
    }
    for old, new in aliases.items():
        if out.get(new) is None and out.get(old) is not None:
            out[new] = out[old]
        out.pop(old, None)
    out.setdefault("index", index)
    out.setdefault("finishReason", "STOP")
    content = out.get("content")
    if isinstance(content, dict):
        content = dict(content)
        content.setdefault("role", "model")
        parts = content.get("parts")
        if parts is None:
            content["parts"] = []
        elif not isinstance(parts, list):
            content["parts"] = [_gemini_content_part(parts)]
        else:
            content["parts"] = [_gemini_content_part(part) for part in parts]
        out["content"] = content
    for key in ("safetyRatings", "citationMetadata", "groundingMetadata", "logprobsResult"):
        if key in out:
            out[key] = _gemini_normalize_response_object(out[key])
    return out


def _gemini_normalize_response_object(value: Any) -> Any:
    if isinstance(value, list):
        return [_gemini_normalize_response_object(item) for item in value]
    if not isinstance(value, dict):
        return value
    aliases = {
        "safety_ratings": "safetyRatings",
        "block_reason": "blockReason",
        "block_reason_message": "blockReasonMessage",
        "citation_metadata": "citationMetadata",
        "grounding_metadata": "groundingMetadata",
        "url_context_metadata": "urlContextMetadata",
        "search_entry_point": "searchEntryPoint",
        "rendered_content": "renderedContent",
        "grounding_chunks": "groundingChunks",
        "grounding_supports": "groundingSupports",
        "web_search_queries": "webSearchQueries",
        "retrieval_metadata": "retrievalMetadata",
        "google_search_dynamic_retrieval_score": "googleSearchDynamicRetrievalScore",
        "token_count": "tokenCount",
    }
    out: dict[str, Any] = {}
    for key, child in value.items():
        mapped = aliases.get(str(key), key)
        out[mapped] = _gemini_normalize_response_object(child)
    return out


def _gemini_finalize_generate_response(response: dict[str, Any], *, model_name: str, request_body: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(response, dict):
        return response
    out = dict(response)
    for old, new in {
        "model_version": "modelVersion",
        "response_id": "responseId",
        "prompt_feedback": "promptFeedback",
        "automatic_function_calling_history": "automaticFunctionCallingHistory",
    }.items():
        if out.get(new) is None and out.get(old) is not None:
            out[new] = out[old]
        out.pop(old, None)
    out.setdefault("modelVersion", _gemini_resource_model_id(model_name))
    out.setdefault("responseId", "resp_" + uuid.uuid4().hex)
    if isinstance(out.get("promptFeedback"), dict):
        out["promptFeedback"] = _gemini_normalize_response_object(out["promptFeedback"])
    if isinstance(out.get("automaticFunctionCallingHistory"), list):
        out["automaticFunctionCallingHistory"] = _gemini_normalize_contents(out["automaticFunctionCallingHistory"])
    candidates = out.get("candidates")
    if isinstance(candidates, list):
        finalized_candidates: list[Any] = []
        for idx, candidate in enumerate(candidates):
            if isinstance(candidate, dict):
                candidate = _gemini_normalize_candidate(candidate, idx)
            finalized_candidates.append(candidate)
        out["candidates"] = finalized_candidates
    out["usageMetadata"] = _gemini_normalize_usage_metadata(
        out.get("usageMetadata") or out.get("usage_metadata"),
        request_body=request_body,
        response=out,
    )
    out.pop("usage_metadata", None)
    return out


def _gemini_image_prompt_from_body(body: dict[str, Any]) -> str:
    contents = body.get("contents")
    if isinstance(contents, list):
        texts = [_gemini_content_text(content) for content in contents if isinstance(content, dict)]
        prompt = "\n".join(text for text in texts if text.strip()).strip()
        if prompt:
            return prompt
    instances = body.get("instances")
    if isinstance(instances, list) and instances:
        first = instances[0]
        if isinstance(first, dict):
            for key in ("prompt", "text", "content"):
                if first.get(key) is not None:
                    return _msg_text(first[key]).strip()
        return _msg_text(first).strip()
    for key in ("prompt", "text"):
        if body.get(key) is not None:
            return _msg_text(body[key]).strip()
    for wrapper_key in ("config", "parameters", "generationConfig"):
        wrapper = body.get(wrapper_key)
        if isinstance(wrapper, dict):
            normalized = _gemini_normalize_request(wrapper)
            for key in ("prompt", "text"):
                if normalized.get(key) is not None:
                    return _msg_text(normalized[key]).strip()
    return ""


def _gemini_image_options_from_body(body: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for key in ("parameters", "generationConfig", "config"):
        value = body.get(key)
        if not isinstance(value, dict):
            continue
        normalized = _gemini_normalize_request(value)
        options.update(normalized)
        image_config = normalized.get("imageConfig")
        if isinstance(image_config, dict):
            options.update(_gemini_normalize_request(image_config))
    for key in ("aspectRatio", "imageSize", "numberOfImages", "sampleCount"):
        if body.get(key) is not None:
            options[key] = body[key]
    return options


def _gemini_image_count(body: dict[str, Any]) -> int:
    options = _gemini_image_options_from_body(body)
    raw_count = options.get("numberOfImages", options.get("sampleCount", 1))
    count = _gemini_int_value(raw_count)
    if not isinstance(count, int) or isinstance(count, bool):
        return 1
    return max(1, min(count, 8))


async def _gemini_generate_image_payload(model_name: str, body: dict[str, Any]) -> dict[str, Any]:
    model = _resolve_gemini_model(model_name)
    if not _model_capabilities(model)["image_generation"]:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not an image generation model.")
    prompt = _gemini_image_prompt_from_body(body)
    if not prompt:
        raise HTTPException(status_code=400, detail="Image generation requires a prompt or text content.")
    options = _gemini_image_options_from_body(body)
    aspect_ratio = (
        body.get("aspectRatio")
        or options.get("aspectRatio")
        or "square"
    )
    image_size = (
        body.get("imageSize")
        or options.get("imageSize")
        or "1K"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = await asyncio.to_thread(
            _get_client().generate_image,
            prompt=prompt,
            output_dir=Path(tmpdir),
            aspect_ratio=str(aspect_ratio),
            image_size=str(image_size),
        )
        image_bytes = output_path.read_bytes()
        mime_type = mimetypes.guess_type(str(output_path))[0] or "image/png"
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    generated_file = _gemini_store_generated_file(
        image_bytes,
        mime_type=mime_type,
        display_name=Path(str(output_path)).name,
    )
    operation = {
        "name": "operations/generateImage-" + uuid.uuid4().hex,
        "metadata": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.GenerateImageMetadata",
            "model": _gemini_model_name(model),
            "generatedFile": generated_file["name"],
        },
        "done": True,
        "response": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.GeneratedFile",
            **generated_file,
        },
    }
    _gemini_store_operation(operation)
    return {
        "prompt": prompt,
        "mimeType": mime_type,
        "base64": image_b64,
        "generatedFile": generated_file,
        "operation": operation,
    }


def _gemini_content_item_to_part(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        return {"text": item}
    if not isinstance(item, dict):
        if item is None:
            return None
        return {"text": str(item)}
    normalized = _gemini_normalize_request(item)
    if not isinstance(normalized, dict):
        return {"text": str(normalized)}
    item_type = str(normalized.get("type") or "").strip()
    if item_type in {"text", "input_text", "output_text"} and normalized.get("text") is not None:
        return {"text": str(normalized["text"])}
    inline_part = _gemini_inline_data_part(normalized)
    if inline_part:
        return inline_part
    if item_type in {"image", "input_image"}:
        mime_type = str(normalized.get("mimeType") or "image/jpeg")
        uri = (
            normalized.get("fileUri")
            or normalized.get("uri")
            or normalized.get("url")
            or normalized.get("imageUrl")
        )
        if isinstance(normalized.get("imageUrl"), dict):
            uri = normalized["imageUrl"].get("url") or normalized["imageUrl"].get("uri")
        if uri:
            return {"fileData": {"mimeType": mime_type, "fileUri": str(uri)}}
        data = (
            normalized.get("data")
            or normalized.get("imageBytes")
            or normalized.get("base64")
            or normalized.get("b64_json")
        )
        if data is not None:
            text_data = str(data)
            if text_data.startswith("data:"):
                header, _, text_data = text_data.partition(",")
                mime_type = header.split(";", 1)[0].removeprefix("data:") or mime_type
            return {"inlineData": {"mimeType": mime_type, "data": text_data}}
    if item_type in {"file", "input_file"}:
        uri = normalized.get("fileUri") or normalized.get("uri") or normalized.get("url")
        if uri:
            return {
                "fileData": {
                    "mimeType": str(normalized.get("mimeType") or "application/octet-stream"),
                    "fileUri": str(uri),
                }
            }
    for key in ("text", "inlineData", "fileData", "functionCall", "functionResponse", "executableCode", "codeExecutionResult"):
        if key in normalized:
            if key == "text":
                return {"text": str(normalized[key])}
            return {key: normalized[key]}
    return None


def _gemini_message_to_content(message: Any) -> dict[str, Any] | None:
    if isinstance(message, str):
        return {"role": "user", "parts": [{"text": message}]}
    if not isinstance(message, dict):
        return None
    message = _gemini_normalize_request(message)
    if isinstance(message.get("parts"), list):
        role = str(message.get("role") or "user")
        parts = [_gemini_content_item_to_part(part) for part in message["parts"]]
        return {"role": "model" if role == "assistant" else role, "parts": [part for part in parts if part]}
    if message.get("type") in {"text", "input_text", "output_text", "image", "input_image", "file", "input_file"}:
        part = _gemini_content_item_to_part(message)
        if part is None:
            return None
        role = str(message.get("role") or "user")
        return {"role": "model" if role == "assistant" else role, "parts": [part]}
    role = str(message.get("role") or "user")
    content = message.get("content", message.get("text", message.get("input")))
    parts: list[Any]
    if isinstance(content, list):
        parts = []
        for item in content:
            part = _gemini_content_item_to_part(item)
            if part is not None:
                parts.append(part)
    elif content is None:
        parts = []
    else:
        part = _gemini_content_item_to_part(content)
        parts = [part] if part is not None else []
    if not parts:
        return None
    return {"role": "model" if role == "assistant" else role, "parts": parts}


def _gemini_interaction_contents(body: dict[str, Any]) -> list[dict[str, Any]]:
    source = body.get("contents")
    if source is None:
        source = body.get("messages")
    if source is None:
        source = body.get("input")
    if source is None:
        raise HTTPException(status_code=400, detail="interactions.create requires contents, messages, or input.")
    if isinstance(source, dict):
        source = [source]
    if isinstance(source, str):
        source = [source]
    if not isinstance(source, list):
        raise HTTPException(status_code=400, detail="Interaction input must be a string, object, or array.")
    contents: list[dict[str, Any]] = []
    for item in source:
        content = _gemini_message_to_content(item)
        if content is not None:
            contents.append(content)
    if not contents:
        raise HTTPException(status_code=400, detail="Interaction input did not contain any supported content.")
    return contents


def _gemini_interaction_body(body: Any) -> dict[str, Any]:
    normalized = _gemini_normalize_request(body)
    if not isinstance(normalized, dict):
        return {}
    config = normalized.get("config") if isinstance(normalized.get("config"), dict) else {}
    interaction = normalized.get("interaction") if isinstance(normalized.get("interaction"), dict) else {}
    if interaction:
        merged = dict(interaction)
        if config and "config" not in merged:
            merged["config"] = config
        for key, value in normalized.items():
            if key not in {"config", "interaction"} and key not in merged:
                merged[key] = value
        return merged
    return normalized


def _gemini_live_turns_from_message(message: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    client_content = message.get("clientContent") or message.get("client_content")
    if not isinstance(client_content, dict):
        return [], False
    turns = client_content.get("turns") or []
    if isinstance(turns, dict) or isinstance(turns, str):
        turns = [turns]
    contents: list[dict[str, Any]] = []
    if isinstance(turns, list):
        for turn in turns:
            content = _gemini_message_to_content(turn)
            if content is not None:
                contents.append(content)
    return contents, bool(client_content.get("turnComplete") or client_content.get("turn_complete"))


async def _gemini_live_generate(
    *,
    model_name: str,
    history: list[dict[str, Any]],
    setup: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model = _resolve_gemini_model(model_name)
    setup = _gemini_apply_response_format(setup)
    body: dict[str, Any] = {"contents": history}
    for key in ("systemInstruction", "generationConfig", "safetySettings", "tools", "toolConfig"):
        if key in setup:
            body[key] = setup[key]
    body = _gemini_apply_file_search(_gemini_inline_local_files(body))
    data = await asyncio.to_thread(
        _get_client().generate_raw,
        request=body,
        model=str(model["antigravity_model"]),
    )
    response = _gemini_finalize_generate_response(
        _gemini_unwrap_response(data),
        model_name=model_name,
        request_body=body,
    )
    candidates = response.get("candidates") or []
    model_turn = None
    if candidates and isinstance(candidates[0], dict) and isinstance(candidates[0].get("content"), dict):
        model_turn = candidates[0]["content"]
    return response, model_turn


def _gemini_predict_to_generate_body(body: dict[str, Any]) -> dict[str, Any]:
    if isinstance(body.get("contents"), list):
        request_body = dict(body)
        request_body.pop("instances", None)
        request_body.pop("parameters", None)
        return request_body
    instances = body.get("instances")
    if not isinstance(instances, list):
        raise HTTPException(status_code=400, detail="predict requires instances or contents.")
    contents: list[dict[str, Any]] = []
    for instance in instances:
        if isinstance(instance, dict) and isinstance(instance.get("content"), dict):
            contents.append(instance["content"])
        elif isinstance(instance, dict) and isinstance(instance.get("parts"), list):
            contents.append({"role": instance.get("role") or "user", "parts": instance["parts"]})
        elif isinstance(instance, dict) and instance.get("text") is not None:
            contents.append({"role": "user", "parts": [{"text": str(instance["text"])}]})
        else:
            contents.append({"role": "user", "parts": [{"text": json.dumps(instance, ensure_ascii=False)}]})
    request_body = {"contents": contents}
    parameters = body.get("parameters")
    if isinstance(parameters, dict):
        params = _gemini_apply_generate_config(_gemini_normalize_request(parameters))
        params = _gemini_apply_response_format(params)
        nested_gen = params.pop("generationConfig", None)
        if isinstance(nested_gen, dict):
            request_body["generationConfig"] = _gemini_normalize_generation_config(nested_gen)
        for key in _GEMINI_GENERATE_CONFIG_TOP_LEVEL_KEYS:
            if key in params and key not in request_body:
                request_body[key] = params.pop(key)
        remaining_gen = {
            key: value
            for key, value in params.items()
            if key in _GEMINI_GENERATION_CONFIG_KEYS and value is not None
        }
        if remaining_gen:
            existing_gen = request_body.get("generationConfig") if isinstance(request_body.get("generationConfig"), dict) else {}
            request_body["generationConfig"] = _gemini_normalize_generation_config({**existing_gen, **remaining_gen})
    for key in _GEMINI_GENERATE_CONFIG_TOP_LEVEL_KEYS | {"generationConfig"}:
        if key in body and key not in request_body:
            request_body[key] = body[key]
    return _gemini_normalize_generate_body(request_body)


def _gemini_legacy_prompt_text(body: dict[str, Any]) -> str:
    prompt = body.get("prompt")
    if isinstance(prompt, dict):
        if prompt.get("text") is not None:
            return str(prompt["text"])
        messages = prompt.get("messages")
        if isinstance(messages, list):
            return "\n".join(_msg_text(item.get("content") if isinstance(item, dict) else item) for item in messages)
    if isinstance(prompt, str):
        return prompt
    if body.get("text") is not None:
        return str(body["text"])
    message = body.get("message")
    if isinstance(message, dict):
        if message.get("content") is not None:
            return _msg_text(message["content"])
        if message.get("text") is not None:
            return str(message["text"])
    return _msg_text(body.get("contents") or body.get("input") or "")


def _gemini_legacy_body_to_generate(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_apply_generate_config(body)
    if isinstance(body.get("contents"), list):
        out = dict(body)
        out.pop("prompt", None)
        out.pop("message", None)
        out.pop("text", None)
        return out
    text = _gemini_legacy_prompt_text(body)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Legacy Gemini request requires prompt, text, message, or contents.")
    out: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": text}]}]}
    for key in ("systemInstruction", "safetySettings", "generationConfig", "tools", "toolConfig", "cachedContent", "labels", "serviceTier", "store", "processingOptions", "responseFormat"):
        if key in body:
            out[key] = body[key]
    for old_key, new_key in (
        ("temperature", "temperature"),
        ("candidateCount", "candidateCount"),
        ("topP", "topP"),
        ("topK", "topK"),
        ("maxOutputTokens", "maxOutputTokens"),
    ):
        if old_key in body:
            out.setdefault("generationConfig", {})[new_key] = body[old_key]
    return out


async def _gemini_legacy_generate(model_name: str, body: dict[str, Any]) -> dict[str, Any]:
    model = _resolve_gemini_model(model_name)
    request_body = _gemini_legacy_body_to_generate(body)
    request_body = _gemini_apply_response_format(request_body)
    request_body = _gemini_normalize_generate_body(request_body)
    request_body = _gemini_apply_cached_content(request_body)
    request_body = _gemini_apply_file_search(request_body)
    _gemini_reject_unsupported_builtin_tools(request_body)
    request_body = _gemini_inline_local_files(request_body)
    data = await asyncio.to_thread(
        _get_client().generate_raw,
        request=request_body,
        model=str(model["antigravity_model"]),
    )
    return _gemini_unwrap_response(data)


async def _gemini_tuned_legacy_generate(tuned_model_id: str, body: dict[str, Any]) -> dict[str, Any]:
    model = _gemini_tuned_base_model(tuned_model_id)
    request_body = _gemini_legacy_body_to_generate(body)
    request_body = _gemini_apply_response_format(request_body)
    request_body = _gemini_normalize_generate_body(request_body)
    request_body = _gemini_apply_cached_content(request_body)
    request_body = _gemini_apply_file_search(request_body)
    _gemini_reject_unsupported_builtin_tools(request_body)
    request_body = _gemini_inline_local_files(request_body)
    data = await asyncio.to_thread(
        _get_client().generate_raw,
        request=request_body,
        model=str(model["antigravity_model"]),
    )
    return _gemini_finalize_generate_response(
        _gemini_unwrap_response(data),
        model_name=_gemini_tuned_name(tuned_model_id),
        request_body=request_body,
    )


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
    body = _gemini_normalize_request(_gemini_embed_request_payload(body))
    config = _gemini_embedding_config(body)
    output_dim = config.get("outputDimensionality") or 768
    embeddings: list[dict[str, Any]] = []
    for item in _gemini_embedding_content_items(body):
        content = _gemini_embedding_content(item)
        seed_parts = [_gemini_content_text(content)]
        if config.get("taskType"):
            seed_parts.append(f"taskType:{config['taskType']}")
        if config.get("title"):
            seed_parts.append(f"title:{config['title']}")
        text = "\n".join(str(part) for part in seed_parts if part is not None)
        embeddings.append({"values": _gemini_embedding_values(text, dimensions=int(output_dim))})
    if len(embeddings) == 1:
        return {"embedding": embeddings[0], "embeddings": embeddings}
    return {"embeddings": embeddings, "embedding": embeddings[0]}


def _gemini_batch_request_item(item: dict[str, Any], *wrapper_keys: str) -> dict[str, Any]:
    for key in wrapper_keys:
        wrapped = item.get(key)
        if isinstance(wrapped, dict):
            out = dict(wrapped)
            for outer_key in (
                "model",
                "config",
                "generationConfig",
                "embedContentConfig",
                "outputDimensionality",
                "taskType",
                "title",
                "autoTruncate",
                "documentOcr",
                "audioTrackExtraction",
                "providerOptions",
                "providerMetadata",
                "google",
            ):
                if outer_key in item and outer_key not in out:
                    out[outer_key] = item[outer_key]
            return out
    return dict(item)


def _gemini_batch_embedding_from_request(body: dict[str, Any]) -> dict[str, Any]:
    requests = body.get("requests")
    if not isinstance(requests, list):
        raise HTTPException(status_code=400, detail="batchEmbedContents requires a requests array.")
    embeddings: list[dict[str, Any]] = []
    shared_config = _gemini_embedding_config(body)
    for item in requests:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="batchEmbedContents request items must be objects.")
        normalized_item = _gemini_normalize_request(_gemini_batch_request_item(item, "request", "embedContentRequest"))
        if shared_config and isinstance(normalized_item, dict):
            item_config = _gemini_embedding_config(normalized_item)
            merged_config = dict(shared_config)
            merged_config.update(item_config)
            normalized_item["config"] = merged_config
        embedded = _gemini_embedding_from_request(normalized_item)
        embeddings.extend(embedded.get("embeddings") or [embedded["embedding"]])
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
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("operations/"):
        return key
    return "operations/" + key


def _gemini_store_operation(operation: dict[str, Any]) -> dict[str, Any]:
    index = _gemini_load_operations_index()
    index[operation["name"]] = operation
    _gemini_save_operations_index(index)
    return operation


def _gemini_get_operation(name: str) -> dict[str, Any] | None:
    index = _gemini_load_operations_index()
    resolved = _gemini_resolve_operation_key(index, name)
    if resolved:
        return index.get(resolved)
    return None


def _gemini_cancel_operation(operation: dict[str, Any]) -> None:
    if operation.get("done"):
        return
    operation["done"] = True
    operation["error"] = {"code": 1, "message": "Operation cancelled.", "status": "CANCELLED"}
    metadata = operation.get("metadata")
    if isinstance(metadata, dict):
        metadata["endTime"] = _gemini_now_iso()
    _gemini_store_operation(operation)


def _gemini_resolve_operation_key(index: dict[str, dict[str, Any]], name: str) -> str | None:
    normalized = _gemini_operation_name(name)
    if normalized in index:
        return normalized
    key = name.strip().strip("/")
    if "/operations/" in key:
        suffix = key.rsplit("/operations/", 1)[-1]
        for op_name in index:
            if op_name == f"operations/{suffix}" or op_name.endswith("/" + suffix):
                return op_name
    return None


def _gemini_operation_scope_value(operation: dict[str, Any], key: str) -> str:
    metadata = operation.get("metadata") if isinstance(operation.get("metadata"), dict) else {}
    response = operation.get("response") if isinstance(operation.get("response"), dict) else {}
    value = metadata.get(key) or response.get(key)
    if value is None and key == "generatedFile":
        value = metadata.get("generated_file") or metadata.get("generatedFileName") or response.get("name")
    if value is None and key == "corpus":
        value = metadata.get("corpusName") or metadata.get("corpora") or response.get("corpus")
    return str(value or "")


def _gemini_operation_scope_matches(operation: dict[str, Any], scope_key: str, scope_name: str | None = None) -> bool:
    if scope_key == "generatedFile" and scope_name is None:
        return bool(_gemini_operation_scope_value(operation, "generatedFile"))
    target = scope_name or ""
    value = _gemini_operation_scope_value(operation, scope_key)
    return value == target or value.endswith("/" + target.strip("/"))


def _gemini_get_scoped_operation(scope_key: str, scope_name: str | None, operation_id: str) -> dict[str, Any] | None:
    operation = _gemini_get_operation(operation_id)
    if not operation:
        operation = _gemini_get_operation(f"{scope_name or ''}/operations/{operation_id}")
    if operation and _gemini_operation_scope_matches(operation, scope_key, scope_name):
        return operation
    return None


def _gemini_operations_for_scope(scope_key: str, scope_name: str | None = None) -> list[dict[str, Any]]:
    operations = list(_gemini_load_operations_index().values())
    if scope_key == "all":
        return operations
    if scope_key == "generatedFile":
        scoped = [op for op in operations if _gemini_operation_scope_value(op, "generatedFile")]
    else:
        target = scope_name or ""
        scoped = [
            op for op in operations
            if _gemini_operation_scope_value(op, scope_key) == target
            or _gemini_operation_scope_value(op, scope_key).endswith("/" + target.strip("/"))
        ]
    scoped.sort(key=lambda item: item.get("name") or "")
    return scoped


def _gemini_operation_list_response(operations: list[dict[str, Any]], page_size: int, page_token: str | None) -> dict[str, Any]:
    operations.sort(key=lambda item: item.get("name") or "")
    start = int(page_token or 0) if page_token and page_token.isdigit() else 0
    end = start + page_size
    return {"operations": operations[start:end], "nextPageToken": str(end) if end < len(operations) else ""}


def _gemini_permission_list_response(permissions: Any, request: Request) -> dict[str, Any]:
    page_size, page_token = _gemini_list_query_params(
        request,
        default_page_size=10,
        max_page_size=1000,
        clamp_page_size=True,
    )
    source = permissions.values() if isinstance(permissions, dict) else (permissions or [])
    items = [item for item in source if isinstance(item, dict)]
    items.sort(key=lambda item: item.get("name") or "")
    start = int(page_token or 0) if page_token and page_token.isdigit() else 0
    end = start + page_size
    return {"permissions": items[start:end], "nextPageToken": str(end) if end < len(items) else ""}


def _gemini_list_query_params(
    request: Request,
    *,
    default_page_size: int,
    max_page_size: int,
    clamp_page_size: bool = False,
) -> tuple[int, str | None]:
    query = request.query_params
    raw_page_size = query.get("pageSize") or query.get("page_size")
    if raw_page_size is None:
        page_size = default_page_size
    else:
        try:
            page_size = int(raw_page_size)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="pageSize must be an integer.",
                headers={"x-gemini-error-field": "pageSize"},
            )
    if page_size < 1:
        raise HTTPException(
            status_code=400,
            detail="pageSize must be at least 1.",
            headers={"x-gemini-error-field": "pageSize"},
        )
    if page_size > max_page_size:
        if clamp_page_size:
            page_size = max_page_size
        else:
            raise HTTPException(
                status_code=400,
                detail=f"pageSize must be between 1 and {max_page_size}.",
                headers={"x-gemini-error-field": "pageSize"},
            )
    return page_size, query.get("pageToken") or query.get("page_token")


def _gemini_query_bool(request: Request, camel_name: str, snake_name: str, default: bool = False) -> bool:
    value = request.query_params.get(camel_name)
    if value is None:
        value = request.query_params.get(snake_name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _gemini_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gemini_batches_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", "data/gemini_batches")).expanduser()


def _gemini_batches_index_path() -> Path:
    return _gemini_batches_root() / "index.json"


def _gemini_load_batches_index() -> dict[str, dict[str, Any]]:
    path = _gemini_batches_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini batches index; starting empty.")
        return {}


def _gemini_save_batches_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_batches_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_batches_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_batch_name(name: str) -> str:
    key = name.strip().strip("/")
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("batches/"):
        return key
    return "batches/" + key


def _gemini_batch_body(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    for key in ("batch", "generateContentBatch", "embedContentBatch"):
        if isinstance(body.get(key), dict):
            merged = dict(body[key])
            for outer_key in ("model", "displayName", "inputConfig", "outputConfig", "requests", "config", "priority", "updateMask"):
                if outer_key in body and outer_key not in merged:
                    merged[outer_key] = body[outer_key]
            merged["_batchKind"] = "embed" if key == "embedContentBatch" else "generate"
            return merged
    return body


def _gemini_batch_stats(request_count: int, *, successful: int | None = None, failed: int = 0, pending: int = 0) -> dict[str, str]:
    success_count = request_count - failed - pending if successful is None else successful
    return {
        "requestCount": str(max(0, request_count)),
        "successfulRequestCount": str(max(0, success_count)),
        "failedRequestCount": str(max(0, failed)),
        "pendingRequestCount": str(max(0, pending)),
    }


def _gemini_batch_update_fields(update_mask: str | None, body: dict[str, Any]) -> set[str]:
    aliases = {
        "display_name": "displayName",
        "displayName": "displayName",
        "batch.displayName": "displayName",
        "batch.display_name": "displayName",
        "generateContentBatch.displayName": "displayName",
        "embedContentBatch.displayName": "displayName",
        "generate_content_batch.display_name": "displayName",
        "embed_content_batch.display_name": "displayName",
        "priority": "priority",
        "batch.priority": "priority",
        "generateContentBatch.priority": "priority",
        "embedContentBatch.priority": "priority",
        "generate_content_batch.priority": "priority",
        "embed_content_batch.priority": "priority",
    }
    if not update_mask:
        return {key for key in ("displayName", "priority") if key in body}
    fields: set[str] = set()
    for raw in update_mask.split(","):
        key = raw.strip()
        if not key:
            continue
        fields.add(aliases.get(key, aliases.get(key.rsplit(".", 1)[-1], key)))
    unsupported = fields - {"displayName", "priority"}
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail="batch update supports updateMask fields: displayName, priority.",
        )
    return fields


def _gemini_batch_optional_fields(body: dict[str, Any]) -> dict[str, Any]:
    return {
        key: body[key]
        for key in ("inputConfig", "outputConfig", "priority")
        if key in body and body[key] is not None
    }


def _gemini_store_batch(batch: dict[str, Any]) -> dict[str, Any]:
    index = _gemini_load_batches_index()
    index[batch["name"]] = batch
    _gemini_save_batches_index(index)
    return batch


def _gemini_get_batch(name: str) -> dict[str, Any] | None:
    return _gemini_load_batches_index().get(_gemini_batch_name(name))


def _gemini_batch_operation(batch: dict[str, Any]) -> dict[str, Any]:
    operation = _gemini_get_operation(str(batch.get("operation") or "")) or {
        "name": batch.get("operation") or batch["name"],
        "done": batch.get("state") in {
            "BATCH_STATE_SUCCEEDED",
            "BATCH_STATE_FAILED",
            "BATCH_STATE_CANCELLED",
            "BATCH_STATE_EXPIRED",
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        },
    }
    out = dict(operation)
    metadata = dict(out.get("metadata") if isinstance(out.get("metadata"), dict) else {})
    metadata.update({
        "batch": batch["name"],
        "batchResource": batch,
        "displayName": batch.get("displayName"),
        "model": batch.get("model"),
        "state": batch.get("state"),
        "stats": batch.get("stats"),
        "batchStats": batch.get("batchStats") or batch.get("stats"),
        "createTime": batch.get("createTime"),
        "updateTime": batch.get("updateTime"),
        "endTime": batch.get("endTime"),
        "operation": batch.get("operation"),
    })
    out["name"] = batch["name"]
    out["metadata"] = {key: value for key, value in metadata.items() if value is not None}
    out["done"] = bool(out.get("done"))
    if "response" not in out and isinstance(batch.get("response"), dict):
        out["response"] = batch["response"]
    return out


def _gemini_get_path_value(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _gemini_batch_filter_matches(batch: dict[str, Any], operation: dict[str, Any], filter_expr: str | None) -> bool:
    if not filter_expr:
        return True
    aliases = {
        "batch": "name",
        "batch.name": "name",
        "batch.display_name": "displayName",
        "display_name": "displayName",
        "batch.displayName": "displayName",
        "batch.state": "state",
        "batch.model": "model",
        "operation.name": "operation",
        "metadata.batch": "metadata.batch",
        "metadata.batchResource.name": "metadata.batchResource.name",
        "metadata.batchResource.display_name": "metadata.batchResource.displayName",
        "metadata.batchResource.displayName": "metadata.batchResource.displayName",
        "metadata.batchResource.state": "metadata.batchResource.state",
        "metadata.batchResource.model": "metadata.batchResource.model",
    }
    searchable = dict(batch)
    searchable["done"] = operation.get("done")
    searchable["metadata"] = operation.get("metadata") if isinstance(operation.get("metadata"), dict) else {}
    terms = re.split(r"\s+(?:AND|and)\s+", filter_expr.strip())
    for term in terms:
        term = term.strip()
        if not term:
            continue
        match = re.match(r"^([\w.]+)\s*(!=|=|:)\s*(.+)$", term)
        if not match:
            continue
        field, operator, expected_raw = match.groups()
        expected = expected_raw.strip().strip("\"'")
        actual = _gemini_get_path_value(searchable, aliases.get(field, field))
        if actual is None and "." in field:
            actual = _gemini_get_path_value(operation, aliases.get(field, field))
        if isinstance(actual, bool):
            expected_value: Any = expected.lower() == "true"
        else:
            expected_value = expected
        actual_text = str(actual or "")
        expected_text = str(expected_value)
        if operator == "=" and actual != expected_value and actual_text != expected_text:
            return False
        if operator == "!=" and (actual == expected_value or actual_text == expected_text):
            return False
        if operator == ":" and expected_text.lower() not in actual_text.lower():
            return False
    return True


def _gemini_operation_filter_matches(operation: dict[str, Any], filter_expr: str | None) -> bool:
    if not filter_expr:
        return True
    aliases = {
        "operation": "name",
        "operation.name": "name",
        "metadata.display_name": "metadata.displayName",
        "metadata.state": "metadata.state",
        "metadata.batch_resource": "metadata.batchResource",
        "metadata.batch_resource.state": "metadata.batchResource.state",
        "metadata.batch_resource.display_name": "metadata.batchResource.displayName",
        "error.status": "error.status",
    }
    terms = re.split(r"\s+(?:AND|and)\s+", filter_expr.strip())
    for term in terms:
        term = term.strip()
        if not term:
            continue
        match = re.match(r"^([\w.]+)\s*(!=|=|:)\s*(.+)$", term)
        if not match:
            continue
        field, operator, expected_raw = match.groups()
        expected = expected_raw.strip().strip("\"'")
        actual = _gemini_get_path_value(operation, aliases.get(field, field))
        if isinstance(actual, bool):
            expected_value: Any = expected.lower() == "true"
        else:
            expected_value = expected
        actual_text = str(actual or "")
        expected_text = str(expected_value)
        if operator == "=" and actual != expected_value and actual_text != expected_text:
            return False
        if operator == "!=" and (actual == expected_value or actual_text == expected_text):
            return False
        if operator == ":" and expected_text.lower() not in actual_text.lower():
            return False
    return True


def _gemini_interactions_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_INTERACTIONS_DIR", "data/gemini_interactions")).expanduser()


def _gemini_interactions_index_path() -> Path:
    return _gemini_interactions_root() / "index.json"


def _gemini_load_interactions_index() -> dict[str, dict[str, Any]]:
    path = _gemini_interactions_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini interactions index; starting empty.")
        return {}


def _gemini_save_interactions_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_interactions_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_interactions_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_interaction_name(name: str) -> str:
    key = name.strip().strip("/")
    if key.startswith("v1/"):
        key = key[len("v1/"):]
    if key.startswith("v1beta/"):
        key = key[len("v1beta/"):]
    if key.startswith("interactions/"):
        return key
    return "interactions/" + key


def _gemini_store_interaction(interaction: dict[str, Any]) -> dict[str, Any]:
    interaction = _gemini_interaction_resource(interaction)
    index = _gemini_load_interactions_index()
    index[interaction["name"]] = interaction
    _gemini_save_interactions_index(index)
    return interaction


def _gemini_get_interaction(name: str) -> dict[str, Any] | None:
    interaction = _gemini_load_interactions_index().get(_gemini_interaction_name(name))
    return _gemini_interaction_resource(interaction) if isinstance(interaction, dict) else None


def _gemini_webhooks_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_WEBHOOKS_DIR", "data/gemini_webhooks")).expanduser()


def _gemini_webhooks_index_path() -> Path:
    return _gemini_webhooks_root() / "index.json"


def _gemini_load_webhooks_index() -> dict[str, dict[str, Any]]:
    path = _gemini_webhooks_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini webhooks index; starting empty.")
        return {}


def _gemini_save_webhooks_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_webhooks_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_webhooks_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_webhook_name(name: str) -> str:
    key = name.strip().strip("/")
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("webhooks/"):
        return key
    return "webhooks/" + key


def _gemini_webhook_body(body: Any) -> dict[str, Any]:
    normalized = _gemini_normalize_request(body)
    if not isinstance(normalized, dict):
        return {}
    config = normalized.get("config") if isinstance(normalized.get("config"), dict) else {}
    webhook = normalized.get("webhook") if isinstance(normalized.get("webhook"), dict) else {}
    if webhook:
        merged = dict(config)
        merged.update(webhook)
        for key, value in normalized.items():
            if key not in {"config", "webhook"} and key not in merged:
                merged[key] = value
        normalized = merged
    elif config:
        merged = dict(config)
        for key, value in normalized.items():
            if key != "config":
                merged[key] = value
        normalized = merged
    state = normalized.get("state")
    if isinstance(state, str):
        normalized["state"] = state.strip().lower().replace("_", "-")
    return normalized


def _gemini_store_webhook(webhook: dict[str, Any]) -> dict[str, Any]:
    index = _gemini_load_webhooks_index()
    index[webhook["name"]] = webhook
    _gemini_save_webhooks_index(index)
    return webhook


def _gemini_get_webhook(name: str) -> dict[str, Any] | None:
    return _gemini_load_webhooks_index().get(_gemini_webhook_name(name))


def _gemini_webhook_uri(webhook: dict[str, Any]) -> str:
    return str(webhook.get("uri") or webhook.get("targetUri") or "").strip()


def _gemini_webhook_events(webhook: dict[str, Any]) -> list[str]:
    raw = webhook.get("subscribedEvents")
    if raw is None:
        raw = webhook.get("eventTypes") or webhook.get("events") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(item).strip() for item in raw if str(item).strip()] if isinstance(raw, list) else []


_GEMINI_WEBHOOK_EVENT_ALIASES = {
    "batches.completed": "batch.succeeded",
    "batch.completed": "batch.succeeded",
    "interactions.completed": "interaction.completed",
    "generatedFiles.completed": "video.generated",
    "generated_files.completed": "video.generated",
}


def _gemini_canonical_webhook_event(event_type: str) -> str:
    event = str(event_type).strip()
    return _GEMINI_WEBHOOK_EVENT_ALIASES.get(event, event)


def _gemini_webhook_event_candidates(event_type: str) -> set[str]:
    canonical = _gemini_canonical_webhook_event(event_type)
    candidates = {canonical, str(event_type).strip()}
    candidates.update(alias for alias, target in _GEMINI_WEBHOOK_EVENT_ALIASES.items() if target == canonical)
    return {candidate for candidate in candidates if candidate}


def _gemini_webhook_enabled(webhook: dict[str, Any]) -> bool:
    state = str(webhook.get("state") or "enabled").strip().lower()
    return state in {"enabled", "active", "state_enabled"}


def _gemini_webhook_matches_event(webhook: dict[str, Any], event_type: str) -> bool:
    subscribed = _gemini_webhook_events(webhook)
    if not subscribed:
        return False
    normalized_subscriptions = {_gemini_canonical_webhook_event(item) for item in subscribed}
    normalized_subscriptions.update(subscribed)
    for event in _gemini_webhook_event_candidates(event_type):
        event_family = event.split(".", 1)[0] + ".*" if "." in event else event + ".*"
        if "*" in normalized_subscriptions or event in normalized_subscriptions or event_family in normalized_subscriptions:
            return True
    return False


def _gemini_new_signing_secret() -> dict[str, Any]:
    now = _gemini_now_iso()
    return {
        "id": "secret_" + uuid.uuid4().hex,
        "secret": secrets.token_urlsafe(32),
        "createTime": now,
        "state": "active",
    }


def _gemini_webhook_public_resource(webhook: dict[str, Any], *, include_new_secret: bool = False) -> dict[str, Any]:
    resource = dict(webhook)
    resource["uri"] = _gemini_webhook_uri(resource)
    resource["targetUri"] = resource["uri"]
    resource["subscribedEvents"] = _gemini_webhook_events(resource)
    resource["eventTypes"] = list(resource["subscribedEvents"])
    if isinstance(resource.get("signingSecrets"), list):
        resource["signingSecrets"] = [
            {key: value for key, value in item.items() if key != "secret"}
            for item in resource["signingSecrets"]
            if isinstance(item, dict)
        ]
    if not include_new_secret or not isinstance(resource.get("newSigningSecret"), dict):
        resource.pop("newSigningSecret", None)
    return resource


async def _gemini_post_webhook_json(uri: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[int, str]:
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)) as client:
        response = await client.post(uri, json=payload, headers=headers)
        return response.status_code, response.text[:1000]


async def _gemini_deliver_webhook(webhook: dict[str, Any], event_type: str, resource: dict[str, Any]) -> dict[str, Any]:
    uri = _gemini_webhook_uri(webhook)
    now = _gemini_now_iso()
    payload = {
        "eventType": event_type,
        "eventTime": now,
        "webhook": webhook["name"],
        "resource": resource,
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Webhook-Id": webhook["name"],
        "X-Goog-Webhook-Event": event_type,
        "X-Goog-Webhook-Timestamp": now,
    }
    secrets_list = webhook.get("signingSecrets") if isinstance(webhook.get("signingSecrets"), list) else []
    active_secret = next((item for item in secrets_list if isinstance(item, dict) and item.get("secret")), None)
    if active_secret:
        signature = hmac.new(str(active_secret["secret"]).encode("utf-8"), body, hashlib.sha256).hexdigest()
        headers["X-Goog-Webhook-Signature"] = "sha256=" + signature
    attempt = {
        "eventType": event_type,
        "eventTime": now,
        "uri": uri,
        "status": "pending",
    }
    try:
        status_code, response_text = await _gemini_post_webhook_json(uri, payload, headers)
        attempt.update({
            "status": "delivered" if 200 <= status_code < 300 else "failed",
            "statusCode": status_code,
            "response": response_text,
        })
    except Exception as exc:
        attempt.update({"status": "failed", "error": str(exc)})
    return attempt


async def _gemini_emit_webhook_event(event_type: str, resource: dict[str, Any]) -> None:
    event_type = _gemini_canonical_webhook_event(event_type)
    index = _gemini_load_webhooks_index()
    changed = False
    for name, webhook in list(index.items()):
        if not _gemini_webhook_enabled(webhook):
            continue
        if not _gemini_webhook_uri(webhook):
            continue
        if not _gemini_webhook_matches_event(webhook, event_type):
            continue
        attempt = await _gemini_deliver_webhook(webhook, event_type, resource)
        webhook.setdefault("deliveryAttempts", []).append(attempt)
        webhook["updateTime"] = _gemini_now_iso()
        index[name] = webhook
        changed = True
    if changed:
        _gemini_save_webhooks_index(index)


def _gemini_generated_files_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_GENERATED_FILES_DIR", "data/gemini_generated_files")).expanduser()


def _gemini_generated_files_index_path() -> Path:
    return _gemini_generated_files_root() / "index.json"


def _gemini_load_generated_files_index() -> dict[str, dict[str, Any]]:
    path = _gemini_generated_files_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini generatedFiles index; starting empty.")
        return {}


def _gemini_save_generated_files_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_generated_files_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_generated_files_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_generated_file_name(name: str) -> str:
    key = name.strip().strip("/")
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("generatedFiles/"):
        return key
    return "generatedFiles/" + key


def _gemini_generated_file_resource(meta: dict[str, Any]) -> dict[str, Any]:
    sha256_hash = str(meta.get("sha256Hash") or "")
    if len(sha256_hash) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in sha256_hash):
        try:
            sha256_hash = base64.b64encode(bytes.fromhex(sha256_hash)).decode("ascii")
        except ValueError:
            pass
    return {
        "name": meta["name"],
        "displayName": meta.get("displayName") or meta["name"].split("/", 1)[-1],
        "mimeType": meta.get("mimeType") or "application/octet-stream",
        "sizeBytes": str(int(meta.get("sizeBytes") or 0)),
        "createTime": meta.get("createTime") or _gemini_now_iso(),
        "updateTime": meta.get("updateTime") or meta.get("createTime") or _gemini_now_iso(),
        "expirationTime": meta.get("expirationTime"),
        "sha256Hash": sha256_hash,
        "uri": meta.get("uri") or meta["name"],
        "downloadUri": meta.get("downloadUri") or f"{meta.get('uri') or meta['name']}:download",
        "state": _gemini_file_state(meta.get("state")),
        "source": _gemini_file_source(meta.get("source") or "GENERATED", registered=False),
    }


def _gemini_store_generated_file(
    data: bytes,
    *,
    mime_type: str | None = None,
    display_name: str | None = None,
    source_operation: str | None = None,
) -> dict[str, Any]:
    if not data:
        raise HTTPException(status_code=400, detail="Generated file is empty.")
    root = _gemini_generated_files_root()
    root.mkdir(parents=True, exist_ok=True)
    file_id = "generated_" + uuid.uuid4().hex
    mime = (mime_type or "image/png").split(";", 1)[0].strip()
    extension = mimetypes.guess_extension(mime) or ".bin"
    blob_path = root / f"{file_id}{extension}"
    blob_path.write_bytes(data)
    now = _gemini_now_iso()
    meta = {
        "name": f"generatedFiles/{file_id}",
        "displayName": display_name or f"{file_id}{extension}",
        "mimeType": mime,
        "sizeBytes": len(data),
        "createTime": now,
        "updateTime": now,
        "sha256Hash": hashlib.sha256(data).hexdigest(),
        "uri": f"generatedFiles/{file_id}",
        "downloadUri": f"generatedFiles/{file_id}:download",
        "state": "ACTIVE",
        "source": "GENERATED",
        "path": str(blob_path),
        "sourceOperation": source_operation,
    }
    index = _gemini_load_generated_files_index()
    index[meta["name"]] = meta
    _gemini_save_generated_files_index(index)
    return _gemini_generated_file_resource(meta)


def _gemini_get_generated_file_meta(name: str) -> dict[str, Any] | None:
    return _gemini_load_generated_files_index().get(_gemini_generated_file_name(name))


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


def _gemini_file_enum(value: Any, *, default: str, allowed: set[str]) -> str:
    if value is None:
        return default
    normalized = str(value).strip().upper().replace("-", "_")
    if not normalized:
        return default
    if normalized in allowed:
        return normalized
    if normalized.startswith("FILE_STATE_"):
        candidate = normalized.removeprefix("FILE_STATE_")
    elif normalized.startswith("STATE_"):
        candidate = normalized.removeprefix("STATE_")
    elif normalized.startswith("FILE_SOURCE_"):
        candidate = normalized.removeprefix("FILE_SOURCE_")
    elif normalized.startswith("SOURCE_"):
        candidate = normalized.removeprefix("SOURCE_")
    else:
        candidate = normalized
    return candidate if candidate in allowed else normalized


def _gemini_file_state(value: Any) -> str:
    return _gemini_file_enum(
        value,
        default="ACTIVE",
        allowed={"STATE_UNSPECIFIED", "PROCESSING", "ACTIVE", "FAILED"},
    )


def _gemini_file_source(value: Any, *, registered: bool = False) -> str:
    return _gemini_file_enum(
        value,
        default="REGISTERED" if registered else "UPLOADED",
        allowed={"SOURCE_UNSPECIFIED", "UPLOADED", "GENERATED", "REGISTERED"},
    )


def _gemini_file_resource(meta: dict[str, Any]) -> dict[str, Any]:
    now = int(meta.get("createTime") or time.time())
    sha256_hash = str(meta.get("sha256Hash") or "")
    if len(sha256_hash) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in sha256_hash):
        try:
            sha256_hash = base64.b64encode(bytes.fromhex(sha256_hash)).decode("ascii")
        except ValueError:
            pass
    resource = {
        "name": meta["name"],
        "displayName": meta.get("displayName") or meta["name"].split("/", 1)[-1],
        "mimeType": meta.get("mimeType") or "application/octet-stream",
        "sizeBytes": str(int(meta.get("sizeBytes") or 0)),
        "createTime": meta.get("createTimeIso") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "updateTime": meta.get("updateTimeIso") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "expirationTime": meta.get("expirationTimeIso"),
        "sha256Hash": sha256_hash,
        "uri": meta.get("uri") or meta["name"],
        "downloadUri": meta.get("downloadUri") or (meta.get("uri") or meta["name"]),
        "state": _gemini_file_state(meta.get("state")),
        "source": _gemini_file_source(meta.get("source"), registered=bool(meta.get("registered"))),
    }
    if isinstance(meta.get("error"), dict):
        resource["error"] = meta["error"]
    if isinstance(meta.get("videoMetadata"), dict):
        resource["videoMetadata"] = meta["videoMetadata"]
    elif str(resource["mimeType"]).startswith("video/"):
        resource["videoMetadata"] = {}
    if meta.get("customMetadata") is not None:
        resource["customMetadata"] = meta["customMetadata"]
    return resource


def _gemini_file_expiration_iso(now: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + 48 * 60 * 60))


def _gemini_registered_file_hash(file_meta: dict[str, Any], uri: str) -> str:
    explicit = file_meta.get("sha256Hash") or file_meta.get("sha256_hash")
    if explicit:
        value = str(explicit)
        if len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value):
            try:
                return base64.b64encode(bytes.fromhex(value)).decode("ascii")
            except ValueError:
                pass
        return value
    seed = json.dumps(
        {
            "uri": uri,
            "mimeType": file_meta.get("mimeType") or file_meta.get("mime_type") or "application/octet-stream",
            "sizeBytes": file_meta.get("sizeBytes") or file_meta.get("size_bytes") or 0,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return base64.b64encode(hashlib.sha256(seed).digest()).decode("ascii")


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
    digest = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
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
        "expirationTimeIso": _gemini_file_expiration_iso(now),
        "sha256Hash": digest,
        "uri": f"files/{file_id}",
        "downloadUri": f"files/{file_id}:download",
        "state": "ACTIVE",
        "source": "UPLOADED",
        "path": str(blob_path),
    }
    index = _gemini_load_files_index()
    index[meta["name"]] = meta
    _gemini_save_files_index(index)
    return _gemini_file_resource(meta)


def _gemini_file_metadata(body: dict[str, Any]) -> dict[str, Any]:
    normalized = _gemini_normalize_request(body)
    file_meta = normalized.get("file") if isinstance(normalized.get("file"), dict) else normalized
    if not isinstance(file_meta, dict):
        return {}
    merged = dict(file_meta)
    config = normalized.get("config") if isinstance(normalized.get("config"), dict) else file_meta.get("config")
    if isinstance(config, dict):
        for key in ("displayName", "mimeType", "name", "uri", "downloadUri", "sizeBytes", "videoMetadata", "customMetadata", "state", "source", "expirationTime", "sha256Hash"):
            if key in config and key not in merged:
                merged[key] = config[key]
    return merged


def _gemini_register_file(body: dict[str, Any]) -> dict[str, Any]:
    file_meta = _gemini_file_metadata(body)
    if not file_meta:
        raise HTTPException(status_code=400, detail="files.register requires file metadata.")
    root = _gemini_files_root()
    root.mkdir(parents=True, exist_ok=True)
    file_id = str(file_meta.get("name") or "").strip().strip("/")
    if file_id.startswith("files/"):
        file_id = file_id.split("/", 1)[1]
    if not file_id:
        file_id = "file_" + uuid.uuid4().hex
    now = int(time.time())
    iso = _gemini_now_iso()
    uri = str(file_meta.get("uri") or file_meta.get("fileUri") or f"files/{file_id}")
    meta = {
        "name": f"files/{file_id}",
        "displayName": file_meta.get("displayName") or file_meta.get("display_name") or file_id,
        "mimeType": file_meta.get("mimeType") or file_meta.get("mime_type") or "application/octet-stream",
        "sizeBytes": int(file_meta.get("sizeBytes") or file_meta.get("size_bytes") or 0),
        "createTime": now,
        "updateTime": now,
        "createTimeIso": iso,
        "updateTimeIso": iso,
        "expirationTimeIso": file_meta.get("expirationTime") or file_meta.get("expiration_time"),
        "sha256Hash": _gemini_registered_file_hash(file_meta, uri),
        "uri": uri,
        "downloadUri": file_meta.get("downloadUri") or file_meta.get("download_uri") or uri,
        "state": _gemini_file_state(file_meta.get("state")),
        "source": _gemini_file_source(file_meta.get("source"), registered=True),
        "videoMetadata": file_meta.get("videoMetadata") or file_meta.get("video_metadata"),
        "customMetadata": file_meta.get("customMetadata") or file_meta.get("custom_metadata") or file_meta.get("metadata"),
        "registered": True,
    }
    index = _gemini_load_files_index()
    index[meta["name"]] = meta
    _gemini_save_files_index(index)
    return _gemini_file_resource(meta)


def _gemini_register_files_from_uris(body: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = _gemini_normalize_request(body)
    uris = normalized.get("uris")
    if not isinstance(uris, list) or not uris:
        raise HTTPException(status_code=400, detail="files.register requires a non-empty uris array.")
    config = normalized.get("config") if isinstance(normalized.get("config"), dict) else {}
    files_config = normalized.get("files") if isinstance(normalized.get("files"), list) else []
    files = []
    for idx, uri in enumerate(uris):
        if not isinstance(uri, str) or not uri.strip():
            raise HTTPException(status_code=400, detail="files.register uris must be non-empty strings.")
        item_config = files_config[idx] if idx < len(files_config) and isinstance(files_config[idx], dict) else {}
        normalized_item = _gemini_normalize_request(item_config)
        merged = dict(config)
        merged.update(normalized_item)
        file_name = Path(urlparse(uri).path).name or "registered-file"
        display_name = merged.get("displayName") if len(uris) == 1 or "displayName" in normalized_item else file_name
        files.append(_gemini_register_file({
            **merged,
            "displayName": display_name or file_name,
            "uri": uri,
            "downloadUri": merged.get("downloadUri") or uri,
            "source": merged.get("source") or "REGISTERED",
        }))
    return files


def _gemini_get_file_meta(file_name: str) -> dict[str, Any] | None:
    key = file_name.strip().strip("/")
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
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
    if not path.is_file():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inlineData": {"mimeType": meta.get("mimeType") or "application/octet-stream", "data": data}}


def _gemini_file_data_reference(value: Any) -> tuple[str, str | None]:
    if not isinstance(value, dict):
        return str(value or "").strip(), None
    file_obj = value.get("file") if isinstance(value.get("file"), dict) else None
    uri_value = value.get("fileUri") or value.get("uri") or value.get("name")
    mime_type = value.get("mimeType")
    if not uri_value and file_obj:
        uri_value = file_obj.get("uri") or file_obj.get("fileUri") or file_obj.get("name")
        mime_type = mime_type or file_obj.get("mimeType")
    if isinstance(uri_value, dict):
        mime_type = mime_type or uri_value.get("mimeType")
        uri_value = uri_value.get("uri") or uri_value.get("fileUri") or uri_value.get("name")
    return str(uri_value or "").strip(), str(mime_type).strip() if mime_type else None


def _gemini_remote_file_uri_to_inline(file_uri: str, mime_type: str | None = None) -> dict[str, Any] | None:
    parsed = urlparse(file_uri)
    if parsed.scheme not in {"http", "https"}:
        return None
    max_bytes = int(os.getenv("ANTIGRAVITY_GEMINI_REMOTE_FILE_MAX_BYTES", str(20 * 1024 * 1024)))
    try:
        import httpx

        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
        ) as client:
            response = client.get(file_uri)
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                raise HTTPException(status_code=400, detail=f"Remote file is too large: {content_length} bytes.")
            raw = response.content
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch remote fileData URI: {exc}") from exc
    if len(raw) > max_bytes:
        raise HTTPException(status_code=400, detail=f"Remote file is too large: {len(raw)} bytes.")
    resolved_mime = mime_type or response.headers.get("content-type", "").split(";", 1)[0].strip()
    if not resolved_mime:
        resolved_mime = mimetypes.guess_type(parsed.path)[0] or "application/octet-stream"
    if not (resolved_mime.startswith("image/") or resolved_mime.startswith("video/") or resolved_mime.startswith("audio/") or resolved_mime == "application/pdf"):
        raise HTTPException(status_code=400, detail=f"Unsupported remote file MIME type: {resolved_mime}")
    return {"inlineData": {"mimeType": resolved_mime, "data": base64.b64encode(raw).decode("ascii")}}


def _gemini_inline_local_files(value: Any) -> Any:
    if isinstance(value, list):
        return [_gemini_inline_local_files(item) for item in value]
    if not isinstance(value, dict):
        return value
    file_data = value.get("fileData")
    if isinstance(file_data, dict):
        uri, mime_type = _gemini_file_data_reference(file_data)
        inline = _gemini_file_uri_to_inline(uri)
        if inline:
            return inline
        inline = _gemini_remote_file_uri_to_inline(uri, mime_type)
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
    file_meta = _gemini_file_metadata(metadata) if isinstance(metadata, dict) else {}
    if file_meta:
        display_name = file_meta.get("displayName") or file_meta.get("display_name") or display_name
        mime_type = file_meta.get("mimeType") or file_meta.get("mime_type") or mime_type
    return _gemini_store_file(media, mime_type=mime_type or content_type, display_name=display_name)


def _is_gemini_metadata_file_create(request: Request) -> bool:
    path = request.url.path.rstrip("/")
    if not (path.endswith("/v1beta/files") or path.endswith("/v1/files")):
        return False
    if path.endswith("/upload/v1beta/files") or path.endswith("/upload/v1/files"):
        return False
    if request.query_params.get("uploadType"):
        return False
    if request.headers.get("x-goog-upload-protocol", "").lower() == "resumable":
        return False
    return "application/json" in request.headers.get("content-type", "").lower()


def _is_gemini_resumable_upload_start(request: Request) -> bool:
    upload_type = request.query_params.get("uploadType", "").lower()
    protocol = request.headers.get("x-goog-upload-protocol", "").lower()
    command = request.headers.get("x-goog-upload-command", "").lower()
    return upload_type == "resumable" or protocol == "resumable" or "start" in {
        part.strip() for part in command.split(",") if part.strip()
    }


async def _gemini_start_resumable_upload(request: Request, upload_version: str | None = None) -> JSONResponse:
    body = await request.body()
    metadata: dict[str, Any] = {}
    if body:
        try:
            decoded = json.loads(body.decode("utf-8"))
            metadata = decoded if isinstance(decoded, dict) else {}
        except Exception:
            metadata = {}
    file_meta = _gemini_file_metadata(metadata) if isinstance(metadata, dict) else {}
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
        "offset": 0,
        "data": "",
    }
    _gemini_save_upload_sessions(sessions)
    if upload_version is None:
        raw_path = request.scope.get("raw_path")
        upload_path = (raw_path.decode("ascii", "ignore") if isinstance(raw_path, bytes) else request.url.path).rstrip("/")
        upload_version = "v1" if "/upload/v1/files" in upload_path or upload_path.endswith("/v1/files") else "v1beta"
    upload_url = str(request.base_url).rstrip("/") + f"/upload/{upload_version}/files/{session_id}"
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
    existing = base64.b64decode(str(session.get("data") or ""), validate=False) if session.get("data") else b""
    data = existing + data
    file_resource = _gemini_store_file(
        data,
        mime_type=session.get("mimeType") or request.headers.get("content-type"),
        display_name=session.get("displayName"),
    )
    sessions.pop(session_id, None)
    _gemini_save_upload_sessions(sessions)
    return file_resource


async def _gemini_resumable_upload_command(session_id: str, request: Request) -> Response | dict[str, Any]:
    sessions = _gemini_load_upload_sessions()
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Upload session '{session_id}' not found.")
    command = request.headers.get("x-goog-upload-command", "").lower()
    commands = {part.strip() for part in command.split(",") if part.strip()} or {"upload", "finalize"}
    if "query" in commands:
        return JSONResponse(
            {},
            headers={
                "X-Goog-Upload-Status": "active",
                "X-Goog-Upload-Size-Received": str(int(session.get("offset") or 0)),
            },
        )
    unsupported = commands - {"upload", "finalize", "cancel"}
    if unsupported:
        raise HTTPException(status_code=400, detail=f"Unsupported upload command: {command}")
    if "cancel" in commands:
        sessions.pop(session_id, None)
        _gemini_save_upload_sessions(sessions)
        return JSONResponse({}, headers={"X-Goog-Upload-Status": "cancelled"})

    current_offset = int(session.get("offset") or 0)
    header_offset = request.headers.get("x-goog-upload-offset")
    if header_offset not in (None, ""):
        try:
            requested_offset = int(header_offset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="X-Goog-Upload-Offset must be an integer.") from exc
        if requested_offset != current_offset:
            raise HTTPException(
                status_code=400,
                detail=f"Upload offset mismatch: got {requested_offset}, expected {current_offset}.",
            )

    chunk = await request.body()
    existing = base64.b64decode(str(session.get("data") or ""), validate=False) if session.get("data") else b""
    combined = existing + chunk
    session["data"] = base64.b64encode(combined).decode("ascii")
    session["offset"] = len(combined)
    sessions[session_id] = session

    if "finalize" in commands:
        file_resource = _gemini_store_file(
            combined,
            mime_type=session.get("mimeType") or request.headers.get("content-type"),
            display_name=session.get("displayName"),
        )
        sessions.pop(session_id, None)
        _gemini_save_upload_sessions(sessions)
        return {"file": file_resource}

    _gemini_save_upload_sessions(sessions)
    return JSONResponse(
        {},
        headers={
            "X-Goog-Upload-Status": "active",
            "X-Goog-Upload-Size-Received": str(session["offset"]),
        },
    )


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
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("cachedContents/"):
        return key
    return "cachedContents/" + key


def _gemini_cached_resource(meta: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in meta.items() if k not in {"payload"} and v is not None}
    payload = meta.get("payload")
    if isinstance(payload, dict):
        for key in ("contents", "systemInstruction", "tools", "toolConfig", "safetySettings"):
            if key in payload and key not in out:
                out[key] = payload[key]
    return out


def _gemini_duration_seconds(value: Any) -> int | None:
    if not isinstance(value, str) or not value.endswith("s"):
        return None
    try:
        return max(1, int(float(value[:-1])))
    except ValueError:
        return None


def _gemini_cached_body(body: Any) -> dict[str, Any]:
    normalized = _gemini_normalize_request(body)
    if not isinstance(normalized, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    wrapped = normalized.get("cachedContent")
    if isinstance(wrapped, dict):
        merged = dict(wrapped)
        for key, value in normalized.items():
            if key != "cachedContent" and key not in merged:
                merged[key] = value
        normalized = merged
    normalized = _gemini_apply_generate_config(normalized)
    normalized = _gemini_apply_response_format(normalized)
    return _gemini_normalize_generate_body(normalized)


def _gemini_cached_update_mask(update_mask: str | None) -> set[str] | None:
    if not update_mask:
        return None
    aliases = {
        "ttl": "ttl",
        "expire_time": "expireTime",
        "expiretime": "expireTime",
        "expireTime": "expireTime",
        "cachedContent.ttl": "ttl",
        "cached_content.ttl": "ttl",
        "cachedContent.expireTime": "expireTime",
        "cached_content.expire_time": "expireTime",
    }
    fields: set[str] = set()
    for raw in update_mask.split(","):
        key = raw.strip()
        if not key:
            continue
        fields.add(aliases.get(key, aliases.get(key.rsplit(".", 1)[-1], key)))
    unsupported = fields - {"ttl", "expireTime"}
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail="cachedContents.patch supports updateMask fields: ttl, expireTime.",
        )
    return fields


def _gemini_create_cached_content(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_cached_body(body)
    usage = _gemini_count_tokens_response(_gemini_count_tokens_request(body))
    now = int(time.time())
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    cache_id = "cache_" + uuid.uuid4().hex
    expire_seconds = _gemini_duration_seconds(body.get("ttl")) or 3600
    expire_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + expire_seconds))
    meta = {
        "name": f"cachedContents/{cache_id}",
        "model": body.get("model"),
        "displayName": body.get("displayName") or body.get("display_name"),
        "createTime": iso,
        "updateTime": iso,
        "ttl": body.get("ttl") if not body.get("expireTime") else None,
        "expireTime": body.get("expireTime") or expire_iso,
        "usageMetadata": {
            "totalTokenCount": usage.get("totalTokens", 0),
            "promptTokensDetails": usage.get("promptTokensDetails", []),
        },
        "payload": body,
    }
    index = _gemini_load_cached_index()
    index[meta["name"]] = meta
    _gemini_save_cached_index(index)
    return _gemini_cached_resource(meta)


def _gemini_patch_cached_meta(meta: dict[str, Any], body: dict[str, Any], update_mask: str | None) -> dict[str, Any]:
    update_mask = update_mask or body.pop("updateMask", None)
    fields = _gemini_cached_update_mask(update_mask)
    if fields is None:
        fields = {key for key in ("ttl", "expireTime") if key in body}
    if not fields:
        raise HTTPException(status_code=400, detail="cachedContents.patch requires ttl or expireTime.")
    if "ttl" in fields:
        if "ttl" not in body:
            raise HTTPException(status_code=400, detail="ttl is required by updateMask.")
        seconds = _gemini_duration_seconds(body["ttl"])
        if seconds is None:
            raise HTTPException(status_code=400, detail="ttl must be a duration string such as '3600s'.")
        meta["ttl"] = body["ttl"]
        meta["expireTime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(time.time()) + seconds))
    if "expireTime" in fields:
        if "expireTime" not in body:
            raise HTTPException(status_code=400, detail="expireTime is required by updateMask.")
        meta.pop("ttl", None)
        meta["expireTime"] = body["expireTime"]
    meta["updateTime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    index = _gemini_load_cached_index()
    index[meta["name"]] = meta
    _gemini_save_cached_index(index)
    return _gemini_cached_resource(meta)


def _gemini_get_cached_meta(name: str) -> dict[str, Any] | None:
    return _gemini_load_cached_index().get(_gemini_cached_name(name))


def _gemini_cached_content_reference(value: Any) -> str:
    if isinstance(value, dict):
        candidate = value.get("name") or value.get("cachedContent") or value.get("cached_content")
        return str(candidate or "").strip()
    return str(value or "").strip()


def _gemini_apply_cached_content(body: dict[str, Any]) -> dict[str, Any]:
    cached_name = _gemini_cached_content_reference(body.get("cachedContent"))
    if not cached_name:
        return body
    meta = _gemini_get_cached_meta(cached_name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Cached content '{cached_name}' not found.")
    payload = meta.get("payload") if isinstance(meta.get("payload"), dict) else {}
    merged = dict(body)
    cached_contents = payload.get("contents") if isinstance(payload.get("contents"), list) else []
    current_contents = merged.get("contents") if isinstance(merged.get("contents"), list) else []
    if cached_contents:
        merged["contents"] = cached_contents + current_contents
    for key in ("systemInstruction", "tools", "toolConfig", "safetySettings"):
        if key not in merged and key in payload:
            merged[key] = payload[key]
    merged.pop("cachedContent", None)
    return merged


def _gemini_corpora_root() -> Path:
    return Path(os.getenv("ANTIGRAVITY_GEMINI_CORPORA_DIR", "data/gemini_corpora")).expanduser()


def _gemini_corpora_index_path() -> Path:
    return _gemini_corpora_root() / "index.json"


def _gemini_load_corpora_index() -> dict[str, dict[str, Any]]:
    path = _gemini_corpora_index_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        log.warning("Failed to read Gemini corpora index; starting empty.")
        return {}


def _gemini_save_corpora_index(index: dict[str, dict[str, Any]]) -> None:
    root = _gemini_corpora_root()
    root.mkdir(parents=True, exist_ok=True)
    path = _gemini_corpora_index_path()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _gemini_corpus_name(name: str) -> str:
    key = name.strip().strip("/")
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("corpora/"):
        return key.split("/documents/", 1)[0].split("/permissions/", 1)[0]
    return "corpora/" + key


def _gemini_corpus_document_name(corpus_name: str, document: str) -> str:
    corpus_key = _gemini_corpus_name(corpus_name)
    key = document.strip().strip("/")
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith(corpus_key + "/documents/"):
        return key.split("/chunks/", 1)[0]
    if "/documents/" in key:
        key = key.rsplit("/documents/", 1)[-1].split("/chunks/", 1)[0]
    return f"{corpus_key}/documents/{key}"


def _gemini_corpus_chunk_name(corpus_name: str, document: str, chunk: str) -> str:
    doc_key = _gemini_corpus_document_name(corpus_name, document)
    key = chunk.strip().strip("/")
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith(doc_key + "/chunks/"):
        return key
    if "/chunks/" in key:
        key = key.rsplit("/chunks/", 1)[-1]
    return f"{doc_key}/chunks/{key}"


def _gemini_corpus_permission_name(corpus_name: str, permission: str) -> str:
    corpus_key = _gemini_corpus_name(corpus_name)
    key = permission.strip().strip("/")
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith(corpus_key + "/permissions/"):
        return key
    if "/permissions/" in key:
        key = key.rsplit("/permissions/", 1)[-1]
    return f"{corpus_key}/permissions/{key}"


def _gemini_corpus_resource(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k not in {"documents", "permissions"} and v is not None}


def _gemini_permission_body(body: Any) -> dict[str, Any]:
    normalized = _gemini_normalize_request(body)
    if not isinstance(normalized, dict):
        return {}
    wrapped = normalized.get("permission")
    if isinstance(wrapped, dict):
        merged = dict(wrapped)
        for key, value in normalized.items():
            if key != "permission" and key not in merged:
                merged[key] = value
        normalized = merged
    for key in ("role", "granteeType"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = value.strip().upper().replace("-", "_").replace(" ", "_")
    return normalized


def _gemini_permission_public_resource(name: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "granteeType": body.get("granteeType") or "USER",
        "emailAddress": body.get("emailAddress"),
        "role": body.get("role") or "READER",
    }


def _gemini_permission_update_fields(update_mask: str | None, body: dict[str, Any]) -> set[str]:
    aliases = {
        "role": "role",
        "permission.role": "role",
        "grantee_type": "granteeType",
        "granteeType": "granteeType",
        "permission.grantee_type": "granteeType",
        "permission.granteeType": "granteeType",
        "email_address": "emailAddress",
        "emailAddress": "emailAddress",
        "permission.email_address": "emailAddress",
        "permission.emailAddress": "emailAddress",
    }
    if not update_mask:
        return {key for key in ("role", "granteeType", "emailAddress") if key in body}
    fields: set[str] = set()
    for raw in update_mask.split(","):
        key = raw.strip()
        if not key:
            continue
        fields.add(aliases.get(key, aliases.get(key.rsplit(".", 1)[-1], key)))
    unsupported = fields - {"role", "granteeType", "emailAddress"}
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail="permissions.patch supports updateMask fields: role, granteeType, emailAddress.",
        )
    return fields


def _gemini_simple_update_fields(
    *,
    update_mask: str | None,
    body: dict[str, Any],
    allowed: set[str],
    aliases: dict[str, str],
    resource: str,
) -> set[str]:
    if not update_mask:
        return {key for key in allowed if key in body}
    fields: set[str] = set()
    for raw in update_mask.split(","):
        key = raw.strip()
        if not key:
            continue
        fields.add(aliases.get(key, aliases.get(key.rsplit(".", 1)[-1], key)))
    unsupported = fields - allowed
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"{resource}.patch supports updateMask fields: {', '.join(sorted(allowed))}.",
        )
    return fields


def _gemini_document_resource(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k not in {"chunks"} and v is not None}


def _gemini_chunk_resource(chunk: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in chunk.items() if v is not None}


def _gemini_create_corpus(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_normalize_request(body)
    corpus_id = str(body.get("name") or body.get("corpusId") or body.get("corpus_id") or "").strip().strip("/")
    if corpus_id.startswith("corpora/"):
        corpus_id = corpus_id.split("/", 1)[1]
    if not corpus_id:
        corpus_id = "corpus_" + uuid.uuid4().hex
    now = _gemini_now_iso()
    meta = {
        "name": f"corpora/{corpus_id}",
        "displayName": body.get("displayName") or body.get("display_name") or corpus_id,
        "createTime": now,
        "updateTime": now,
        "documents": {},
        "permissions": {},
    }
    index = _gemini_load_corpora_index()
    index[meta["name"]] = meta
    _gemini_save_corpora_index(index)
    return _gemini_corpus_resource(meta)


def _gemini_corpus_query(meta: dict[str, Any], query: str, *, top_k: int = 10) -> dict[str, Any]:
    terms = {part.lower() for part in re.findall(r"\w+", query or "") if part}
    ranked: list[tuple[float, dict[str, Any]]] = []
    for doc in (meta.get("documents") or {}).values():
        for chunk in (doc.get("chunks") or {}).values():
            text = str(((chunk.get("data") or {}).get("stringValue")) or chunk.get("text") or "")
            words = {part.lower() for part in re.findall(r"\w+", text) if part}
            score = (len(terms & words) / max(1, len(terms))) if terms else 0.0
            if query.lower() in text.lower() and query:
                score = max(score, 1.0)
            if score > 0:
                ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return {
        "relevantChunks": [
            {"chunk": _gemini_chunk_resource(chunk), "chunkRelevanceScore": score}
            for score, chunk in ranked[:top_k]
        ]
    }


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
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("fileSearchStores/"):
        return key
    return "fileSearchStores/" + key


def _gemini_document_name(store_name: str, document_id: str) -> str:
    doc = document_id.strip().strip("/")
    if "/documents/" in doc:
        doc = doc.rsplit("/documents/", 1)[-1]
    return f"{_gemini_fss_name(store_name)}/documents/{doc}"


def _gemini_config_body(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_normalize_request(body)
    config = body.get("config") if isinstance(body.get("config"), dict) else {}
    if not config:
        return body
    merged = dict(config)
    for key, value in body.items():
        if key != "config":
            merged[key] = value
    return _gemini_normalize_request(merged)


def _gemini_fss_body(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        return body
    for key in ("fileSearchStore", "file_search_store"):
        wrapped = body.get(key)
        if isinstance(wrapped, dict):
            merged = _gemini_config_body(wrapped)
            outer = _gemini_config_body({k: v for k, v in body.items() if k != key})
            for outer_key in ("displayName", "embeddingModel", "chunkingConfig", "customMetadata"):
                if outer_key in outer and outer_key not in merged:
                    merged[outer_key] = outer[outer_key]
            return _gemini_normalize_fss_body(merged)
    return _gemini_normalize_fss_body(_gemini_config_body(body))


def _gemini_normalize_fss_body(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_normalize_request(body)
    chunking = body.get("chunkingConfig")
    if isinstance(chunking, dict):
        for config_key in ("whiteSpaceConfig", "sentenceConfig"):
            config = chunking.get(config_key)
            if isinstance(config, dict):
                for key in ("maxTokensPerChunk", "maxOverlapTokens"):
                    if key in config:
                        config[key] = _gemini_int_value(config[key])
        body["chunkingConfig"] = chunking
    return body


def _gemini_fss_document_body(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        return body
    body = _gemini_config_body(body)
    for key in ("fileMetadata", "file_metadata", "file"):
        wrapped = body.get(key)
        if isinstance(wrapped, dict):
            merged = _gemini_config_body(wrapped)
            for outer_key in ("fileName", "fileUri", "displayName", "mimeType", "customMetadata", "metadata", "chunkingConfig"):
                if outer_key in body and outer_key not in merged:
                    merged[outer_key] = body[outer_key]
            body = merged
            break
    return _gemini_normalize_fss_body(body)


def _gemini_fss_resource(meta: dict[str, Any]) -> dict[str, Any]:
    documents = list((meta.get("documents") or {}).values())
    active_count = sum(1 for doc in documents if str(doc.get("state") or "ACTIVE").upper() == "ACTIVE")
    pending_count = sum(1 for doc in documents if str(doc.get("state") or "").upper() in {"PENDING", "PROCESSING"})
    failed_count = sum(1 for doc in documents if str(doc.get("state") or "").upper() == "FAILED")
    total_size = 0
    for doc in documents:
        try:
            total_size += int(doc.get("sizeBytes") or 0)
        except (TypeError, ValueError):
            continue
    resource = {
        "name": meta["name"],
        "displayName": meta.get("displayName") or meta["name"].split("/", 1)[-1],
        "createTime": meta.get("createTime"),
        "updateTime": meta.get("updateTime"),
        "fileCount": len(documents),
        "activeDocumentsCount": active_count,
        "pendingDocumentsCount": pending_count,
        "failedDocumentsCount": failed_count,
        "sizeBytes": str(total_size),
    }
    for key in ("embeddingModel", "chunkingConfig", "customMetadata"):
        if meta.get(key) is not None:
            resource[key] = meta[key]
    return resource


def _gemini_document_resource(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k not in {"content"} and v is not None}


def _gemini_create_fss(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_fss_body(body)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    store_id = "fs_" + uuid.uuid4().hex
    meta = {
        "name": f"fileSearchStores/{store_id}",
        "displayName": body.get("displayName") or body.get("display_name") or store_id,
        "createTime": now,
        "updateTime": now,
        "documents": {},
    }
    for key in ("embeddingModel", "chunkingConfig", "customMetadata"):
        if body.get(key) is not None:
            meta[key] = body[key]
    index = _gemini_load_fss_index()
    index[meta["name"]] = meta
    _gemini_save_fss_index(index)
    return _gemini_fss_resource(meta)


def _gemini_get_fss_meta(store_name: str) -> dict[str, Any] | None:
    return _gemini_load_fss_index().get(_gemini_fss_name(store_name))


def _gemini_store_document(
    store_name: str,
    *,
    display_name: str | None,
    mime_type: str | None,
    content: bytes,
    custom_metadata: Any | None = None,
    chunking_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    if custom_metadata is not None:
        doc["customMetadata"] = custom_metadata
    if chunking_config is not None:
        doc["chunkingConfig"] = chunking_config
    store.setdefault("documents", {})[doc["name"]] = doc
    store["updateTime"] = now
    index[store_key] = store
    _gemini_save_fss_index(index)
    return _gemini_document_resource(doc)


def _gemini_import_file_to_fss(store_name: str, body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_fss_document_body(body)
    file_name = str(body.get("fileName") or body.get("file") or body.get("fileUri") or "")
    meta = _gemini_get_file_meta(file_name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found.")
    path = Path(str(meta.get("path") or ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File blob for '{file_name}' not found.")
    document = _gemini_store_document(
        store_name,
        display_name=body.get("displayName") or body.get("display_name") or meta.get("displayName"),
        mime_type=meta.get("mimeType"),
        content=path.read_bytes(),
        custom_metadata=body.get("customMetadata") or body.get("metadata"),
        chunking_config=body.get("chunkingConfig") if isinstance(body.get("chunkingConfig"), dict) else None,
    )
    return document


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


def _gemini_reject_unsupported_builtin_tools(body: dict[str, Any]) -> None:
    tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    unsupported = {
        "codeExecution": "code_execution",
        "url_context": "url_context",
    }
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        for key, display in unsupported.items():
            if key in tool:
                raise HTTPException(
                    status_code=501,
                    detail=(
                        f"Gemini tool '{display}' is recognized but not implemented by this "
                        "Antigravity-backed proxy."
                    ),
                )


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
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("tunedModels/"):
        return key
    return "tunedModels/" + key


def _gemini_tuned_resource(meta: dict[str, Any]) -> dict[str, Any]:
    resource = {
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
        "supportedGenerationMethods": [
            "generateContent",
            "streamGenerateContent",
            "generateText",
            "batchGenerateContent",
            "countTokens",
            "computeTokens",
            "embedContent",
            "batchEmbedContents",
            "asyncBatchEmbedContent",
        ],
    }
    for key in (
        "tunedModelSource",
        "tuningTask",
        "readerProjectNumbers",
        "hyperparameters",
        "trainingData",
        "validationData",
    ):
        if meta.get(key) is not None:
            resource[key] = meta[key]
    return {key: value for key, value in resource.items() if value is not None}


def _gemini_tuned_scalar_fields(meta: dict[str, Any]) -> None:
    for key in ("temperature", "topP"):
        if key in meta:
            meta[key] = _gemini_float_value(meta[key])
    for key in ("topK",):
        if key in meta:
            meta[key] = _gemini_int_value(meta[key])


def _gemini_tuned_update_fields(update_mask: str | None, body: dict[str, Any]) -> set[str]:
    allowed = {
        "displayName",
        "description",
        "temperature",
        "topP",
        "topK",
        "baseModel",
        "tunedModelSource",
        "tuningTask",
        "readerProjectNumbers",
        "hyperparameters",
        "trainingData",
        "validationData",
    }
    aliases = {
        "display_name": "displayName",
        "tunedModel.displayName": "displayName",
        "tunedModel.display_name": "displayName",
        "tuned_model.display_name": "displayName",
        "tuned_model.displayName": "displayName",
        "base_model": "baseModel",
        "tunedModel.baseModel": "baseModel",
        "tunedModel.base_model": "baseModel",
        "tuned_model.base_model": "baseModel",
        "tuned_model.baseModel": "baseModel",
        "top_p": "topP",
        "top_k": "topK",
        "tuned_model_source": "tunedModelSource",
        "tunedModel.tunedModelSource": "tunedModelSource",
        "tunedModel.tuned_model_source": "tunedModelSource",
        "tuned_model.tuned_model_source": "tunedModelSource",
        "tuned_model.tunedModelSource": "tunedModelSource",
        "tuning_task": "tuningTask",
        "tunedModel.tuningTask": "tuningTask",
        "tunedModel.tuning_task": "tuningTask",
        "tuned_model.tuning_task": "tuningTask",
        "tuned_model.tuningTask": "tuningTask",
        "reader_project_numbers": "readerProjectNumbers",
        "tunedModel.readerProjectNumbers": "readerProjectNumbers",
        "tunedModel.reader_project_numbers": "readerProjectNumbers",
        "tuned_model.reader_project_numbers": "readerProjectNumbers",
        "tuned_model.readerProjectNumbers": "readerProjectNumbers",
        "training_data": "trainingData",
        "tunedModel.trainingData": "trainingData",
        "tunedModel.training_data": "trainingData",
        "tuned_model.training_data": "trainingData",
        "tuned_model.trainingData": "trainingData",
        "validation_data": "validationData",
        "tunedModel.validationData": "validationData",
        "tunedModel.validation_data": "validationData",
        "tuned_model.validation_data": "validationData",
        "tuned_model.validationData": "validationData",
    }
    if not update_mask:
        return {key for key in allowed if key in body}
    fields: set[str] = set()
    for raw in update_mask.split(","):
        key = raw.strip()
        if not key:
            continue
        suffix = key.rsplit(".", 1)[-1]
        normalized = aliases.get(key, aliases.get(suffix, suffix if suffix in allowed else key))
        if normalized.startswith("tuningTask."):
            normalized = "tuningTask"
        fields.add(normalized)
    unsupported = fields - allowed
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail="tunedModels.patch supports updateMask fields: baseModel, description, displayName, "
            "hyperparameters, readerProjectNumbers, temperature, topK, topP, trainingData, "
            "tunedModelSource, tuningTask, validationData.",
        )
    return fields


def _gemini_create_tuned_model(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_normalize_request(body)
    tuned_model = body.get("tunedModel") if isinstance(body.get("tunedModel"), dict) else body
    config = body.get("config") if isinstance(body.get("config"), dict) else {}
    source_id = str(body.get("tunedModelId") or "").strip()
    model_id = source_id or ("tuned_" + uuid.uuid4().hex)
    name = _gemini_tuned_name(model_id)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = {
        "name": name,
        "displayName": tuned_model.get("displayName") or config.get("displayName") or model_id,
        "description": tuned_model.get("description") or "",
        "baseModel": tuned_model.get("baseModel") or config.get("baseModel") or body.get("baseModel") or "models/gemini-3-flash-agent",
        "state": "ACTIVE",
        "createTime": now,
        "updateTime": now,
        "temperature": tuned_model.get("temperature") if tuned_model.get("temperature") is not None else config.get("temperature"),
        "topP": tuned_model.get("topP") if tuned_model.get("topP") is not None else config.get("topP"),
        "topK": tuned_model.get("topK") if tuned_model.get("topK") is not None else config.get("topK"),
        "permissions": {},
    }
    for key in (
        "tunedModelSource",
        "tuningTask",
        "readerProjectNumbers",
        "hyperparameters",
        "trainingData",
        "validationData",
    ):
        value = tuned_model.get(key)
        if value is None:
            value = config.get(key)
        if value is not None:
            meta[key] = value
    _gemini_tuned_scalar_fields(meta)
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
    parent_name = _gemini_tuned_name(parent)
    key = permission_id.strip().strip("/")
    for prefix in ("v1beta/", "v1/"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith(parent_name + "/permissions/"):
        return key
    if "/permissions/" in key:
        key = key.rsplit("/permissions/", 1)[-1]
    return f"{parent_name}/permissions/{key}"


def _gemini_permission_resource(parent: str, perm: dict[str, Any]) -> dict[str, Any]:
    parent_name = _gemini_tuned_name(parent)
    perm = _gemini_permission_body(perm)
    pid = str(perm.get("id") or perm.get("name", "").rsplit("/", 1)[-1] or ("perm_" + uuid.uuid4().hex))
    return {
        "name": f"{parent_name}/permissions/{pid}",
        "granteeType": perm.get("granteeType") or "USER",
        "emailAddress": perm.get("emailAddress"),
        "role": perm.get("role") or "READER",
    }


def _gemini_store_permission(parent: str, body: dict[str, Any]) -> dict[str, Any]:
    index = _gemini_load_tuned_index()
    parent_name = _gemini_tuned_name(parent)
    meta = index.get(parent_name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Tuned model '{parent}' not found.")
    permission_id = "perm_" + uuid.uuid4().hex
    perm = _gemini_permission_resource(parent_name, {"id": permission_id, **_gemini_permission_body(body)})
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
    candidates = [
        bearer,
        request.headers.get("x-api-key", "").strip(),
        request.headers.get("x-goog-api-key", "").strip(),
        request.query_params.get("key", "").strip(),
    ]
    return any(candidate and secrets.compare_digest(candidate, expected) for candidate in candidates)


def _websocket_api_key_valid(websocket: WebSocket) -> bool:
    expected = _proxy_api_key()
    if not expected:
        return True
    auth = websocket.headers.get("authorization", "").strip()
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    candidates = [
        bearer,
        websocket.headers.get("x-api-key", "").strip(),
        websocket.headers.get("x-goog-api-key", "").strip(),
        websocket.query_params.get("key", "").strip(),
    ]
    return any(candidate and secrets.compare_digest(candidate, expected) for candidate in candidates)


def _is_gemini_http_path(path: str) -> bool:
    if path.startswith(("/v1beta/", "/upload/v1beta/", "/upload/v1/")):
        return True
    if not path.startswith("/v1/"):
        return False
    suffix = path[len("/v1/"):]
    if ":" in suffix:
        return True
    gemini_prefixes = (
        "cachedContents",
        "corpora",
        "fileSearchStores",
        "files",
        "generatedFiles",
        "interactions",
        "operations",
        "tunedModels",
        "webhooks",
    )
    return suffix.startswith(gemini_prefixes)


def _gemini_stable_alias_path(path: str) -> str:
    if path.startswith("/upload/v1/files") or path.startswith("/upload/v1/fileSearchStores"):
        return path
    if path.startswith("/upload/v1/"):
        return "/upload/v1beta/" + path[len("/upload/v1/"):]
    if not path.startswith("/v1/"):
        return path
    suffix = path[len("/v1/"):]
    stable_prefixes: tuple[str, ...] = ()
    if suffix.startswith(stable_prefixes):
        return "/v1beta/" + suffix
    return path


def _gemini_alias_query_string(query_string: bytes) -> bytes:
    if not query_string:
        return query_string
    pairs = parse_qsl(query_string.decode("utf-8", errors="ignore"), keep_blank_values=True)
    out: list[tuple[str, str]] = []
    for key, value in pairs:
        mapped = {
            "page_size": "pageSize",
            "page_token": "pageToken",
            "update_mask": "updateMask",
            "upload_type": "uploadType",
            "display_name": "displayName",
            "return_partial_success": "returnPartialSuccess",
        }.get(key, key)
        out.append((mapped, value))
    return urlencode(out, doseq=True).encode("ascii")


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
    aliased_path = _gemini_stable_alias_path(request.scope.get("path", ""))
    if aliased_path != request.scope.get("path"):
        request.scope["path"] = aliased_path
        request.scope["raw_path"] = aliased_path.encode("ascii", errors="ignore")
    aliased_query = _gemini_alias_query_string(request.scope.get("query_string", b""))
    if aliased_query != request.scope.get("query_string", b""):
        request.scope["query_string"] = aliased_query
    expected = _proxy_api_key()
    if not expected or request.url.path == "/health":
        return await call_next(request)
    if _request_api_key_valid(request):
        return await call_next(request)
    if _is_gemini_http_path(str(request.scope.get("path") or request.url.path)):
        return _gemini_error_response(
            "Invalid or missing API key.",
            status_code=401,
            status="UNAUTHENTICATED",
        )
    return _openai_error_response(
        "Invalid or missing API key.",
        status_code=401,
        error_type="authentication_error",
        code="invalid_api_key",
    )


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    if _is_gemini_http_path(str(request.scope.get("path") or request.url.path)):
        field = (exc.headers or {}).get("x-gemini-error-field") if exc.headers else None
        return _gemini_error_response(
            exc.detail,
            status_code=exc.status_code,
            status=_gemini_status_for_http(exc.status_code),
            field=field,
        )
    return _openai_error_response(exc.detail, status_code=exc.status_code)


@app.exception_handler(StarletteHTTPException)
async def _starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if _is_gemini_http_path(str(request.scope.get("path") or request.url.path)):
        headers = getattr(exc, "headers", None) or {}
        return _gemini_error_response(
            exc.detail,
            status_code=exc.status_code,
            status=_gemini_status_for_http(exc.status_code),
            field=headers.get("x-gemini-error-field"),
        )
    return _openai_error_response(exc.detail, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    param = None
    if errors:
        loc = errors[0].get("loc") or []
        param = ".".join(str(part) for part in loc if part not in {"body", "query", "path"})
    if _is_gemini_http_path(str(request.scope.get("path") or request.url.path)):
        return _gemini_error_response(
            "Request validation failed.",
            status_code=400,
            status="INVALID_ARGUMENT",
            field=param,
        )
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
    "minItems", "maxItems", "minLength", "maxLength",
    "propertyOrdering", "anyOf",
}

_SCHEMA_KEY_ALIASES = {
    "min_items": "minItems",
    "max_items": "maxItems",
    "min_length": "minLength",
    "max_length": "maxLength",
    "property_ordering": "propertyOrdering",
    "any_of": "anyOf",
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
    for raw_key, v in schema.items():
        k = _SCHEMA_KEY_ALIASES.get(str(raw_key), raw_key)
        if k not in _SCHEMA_KEEP:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _sanitize_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _sanitize_schema(v)
        elif k == "anyOf" and isinstance(v, list):
            out[k] = [_sanitize_schema(item) for item in v]
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


def _gemini_part_modality(part: Any) -> str:
    if not isinstance(part, dict):
        return "TEXT"
    if part.get("text") is not None:
        return "TEXT"
    data = part.get("inlineData") if isinstance(part.get("inlineData"), dict) else part.get("fileData")
    if not isinstance(data, dict):
        return "TEXT"
    mime = str(data.get("mimeType") or "").lower()
    if mime.startswith("image/"):
        return "IMAGE"
    if mime.startswith("video/"):
        return "VIDEO"
    if mime.startswith("audio/"):
        return "AUDIO"
    if mime == "application/pdf" or mime.startswith("text/"):
        return "DOCUMENT"
    return "DOCUMENT"


def _gemini_count_tokens_response(messages: list[ChatMessage], *, cached_tokens: int = 0) -> dict[str, Any]:
    prompt_estimate = _estimate_prompt_tokens(messages)
    by_modality: dict[str, int] = {}
    for message in messages:
        content = message.content
        parts = content if isinstance(content, list) else [{"text": content}]
        for part in parts:
            modality = _gemini_part_modality(part)
            token_count = _estimate_tokens(part)
            by_modality[modality] = by_modality.get(modality, 0) + token_count

    if not by_modality:
        by_modality["TEXT"] = prompt_estimate
    detail_sum = sum(by_modality.values())
    total_tokens = max(prompt_estimate, detail_sum)
    if detail_sum != total_tokens:
        by_modality["TEXT"] = by_modality.get("TEXT", 0) + (total_tokens - detail_sum)

    response = {
        "totalTokens": total_tokens,
        "promptTokensDetails": [
            {"modality": modality, "tokenCount": count}
            for modality, count in sorted(by_modality.items())
            if count > 0
        ],
        "cachedContentTokenCount": cached_tokens,
        "cacheTokensDetails": [{"modality": "TEXT", "tokenCount": cached_tokens}] if cached_tokens > 0 else [],
    }
    return response


def _gemini_token_texts(value: Any) -> list[str]:
    if isinstance(value, dict):
        for key in ("text", "outputText", "output_text"):
            if value.get(key) is not None:
                text = str(value[key])
                tokens = re.findall(r"\S+", text)
                return tokens or ([text] if text else [])
    text = _msg_text(value)
    tokens = re.findall(r"\S+", text)
    return tokens or ([text] if text else [])


def _gemini_compute_token_id(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7fffffff


def _gemini_compute_tokens_response(messages: list[ChatMessage]) -> dict[str, Any]:
    tokens_info: list[dict[str, Any]] = []
    for message in messages:
        token_texts: list[str] = []
        content = message.content
        parts = content if isinstance(content, list) else [{"text": content}]
        for part in parts:
            if isinstance(part, dict) and (part.get("inlineData") or part.get("inline_data")):
                inline = part.get("inlineData") or part.get("inline_data") or {}
                mime_type = inline.get("mimeType") or inline.get("mime_type") or "application/octet-stream"
                token_texts.append(f"<inline:{mime_type}>")
                continue
            if isinstance(part, dict) and (part.get("fileData") or part.get("file_data")):
                file_data = part.get("fileData") or part.get("file_data") or {}
                file_uri = file_data.get("fileUri") or file_data.get("file_uri") or file_data.get("uri") or "file"
                token_texts.append(f"<file:{file_uri}>")
                continue
            token_texts.extend(_gemini_token_texts(part))
        if not token_texts:
            continue
        tokens_info.append({
            "role": "model" if message.role == "assistant" else message.role,
            "tokenIds": [_gemini_compute_token_id(token) for token in token_texts],
            "tokens": [base64.b64encode(token.encode("utf-8")).decode("ascii") for token in token_texts],
        })
    return {"tokensInfo": tokens_info}


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
        declarations = tool.get("functionDeclarations")
        if declarations is None:
            declarations = tool.get("functionDeclaration")
        if isinstance(declarations, dict):
            declarations = [declarations]
        if isinstance(declarations, list):
            for declaration in declarations:
                if isinstance(declaration, dict) and declaration.get("name"):
                    names.add(str(declaration["name"]))
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
def _gemini_models_list_response(page_size: int, page_token: str | None) -> dict[str, Any]:
    models = [
        _gemini_model_resource(model)
        for model in _MODELS
        if not _model_capabilities(model)["internal"]
    ]
    start = int(page_token or 0) if page_token and page_token.isdigit() else 0
    end = start + page_size
    return {
        "models": models[start:end],
        "nextPageToken": str(end) if end < len(models) else "",
    }


@app.get("/v1/models")
async def list_models(request: Request):
    """Return the list of supported models (OpenAI-compatible)."""
    query = request.query_params
    if "pageSize" in query or "pageToken" in query or "page_size" in query or "page_token" in query:
        try:
            page_size, page_token = _gemini_list_query_params(request, default_page_size=50, max_page_size=1000)
        except HTTPException as exc:
            return _gemini_error_response(exc.detail, status_code=exc.status_code, status="INVALID_ARGUMENT", field="pageSize")
        return _gemini_models_list_response(page_size, page_token)
    return ModelListResponse(data=_MODELS)


@app.get("/v1beta/models")
async def gemini_list_models(request: Request):
    """Gemini-compatible model listing."""
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=50, max_page_size=1000)
    return _gemini_models_list_response(pageSize, pageToken)


@app.get("/v1/models/{model_name:path}/operations")
@app.get("/v1beta/models/{model_name:path}/operations")
async def gemini_list_model_operations(model_name: str, request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    if _is_gemini_video_model_id(model_name):
        model_resource = "models/" + _gemini_resource_model_id(model_name)
    else:
        model = _resolve_gemini_model(model_name)
        model_resource = _gemini_model_name(model)
    return _gemini_operation_list_response(
        _gemini_operations_for_scope("model", model_resource),
        pageSize,
        pageToken,
    )


@app.get("/v1/models/{model_name:path}/operations/{operation_id:path}")
@app.get("/v1beta/models/{model_name:path}/operations/{operation_id:path}")
async def gemini_get_model_operation(model_name: str, operation_id: str):
    model_resource = "models/" + _gemini_resource_model_id(model_name) if _is_gemini_video_model_id(model_name) else _gemini_model_name(_resolve_gemini_model(model_name))
    operation = _gemini_get_scoped_operation("model", model_resource, operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.post("/v1/models/{model_name:path}/operations/{operation_id:path}:wait")
@app.post("/v1beta/models/{model_name:path}/operations/{operation_id:path}:wait")
async def gemini_wait_model_operation(model_name: str, operation_id: str):
    model_resource = "models/" + _gemini_resource_model_id(model_name) if _is_gemini_video_model_id(model_name) else _gemini_model_name(_resolve_gemini_model(model_name))
    operation = _gemini_get_scoped_operation("model", model_resource, operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.post("/v1/models/{model_name:path}/operations/{operation_id:path}:cancel")
@app.post("/v1beta/models/{model_name:path}/operations/{operation_id:path}:cancel")
async def gemini_cancel_model_operation(model_name: str, operation_id: str):
    model_resource = "models/" + _gemini_resource_model_id(model_name) if _is_gemini_video_model_id(model_name) else _gemini_model_name(_resolve_gemini_model(model_name))
    operation = _gemini_get_scoped_operation("model", model_resource, operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    _gemini_cancel_operation(operation)
    return JSONResponse({})


@app.delete("/v1/models/{model_name:path}/operations/{operation_id:path}")
@app.delete("/v1beta/models/{model_name:path}/operations/{operation_id:path}")
async def gemini_delete_model_operation(model_name: str, operation_id: str):
    model_resource = "models/" + _gemini_resource_model_id(model_name) if _is_gemini_video_model_id(model_name) else _gemini_model_name(_resolve_gemini_model(model_name))
    operation = _gemini_get_scoped_operation("model", model_resource, operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    index = _gemini_load_operations_index()
    index.pop(operation["name"], None)
    _gemini_save_operations_index(index)
    return JSONResponse({})


@app.get("/v1/models/{model_name:path}")
@app.get("/v1beta/models/{model_name:path}")
async def gemini_get_model(model_name: str):
    """Gemini-compatible model retrieval."""
    try:
        model = _resolve_gemini_model(model_name)
    except HTTPException as exc:
        if _is_gemini_video_model_id(model_name):
            return _gemini_video_model_resource(model_name)
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="NOT_FOUND")
    return _gemini_model_resource(model)


@app.get("/v1/files")
@app.get("/v1beta/files")
async def gemini_list_files(request: Request):
    """Gemini-compatible local Files API listing."""
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=10, max_page_size=100)
    index = _gemini_load_files_index()
    files = [_gemini_file_resource(meta) for meta in index.values()]
    files.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {
        "files": files[start:end],
        "nextPageToken": str(end) if end < len(files) else "",
    }


@app.post("/v1/files:register")
@app.post("/v1beta/files:register")
async def gemini_register_file(request: Request):
    """Gemini-compatible metadata-only file registration."""
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        if "uris" in body:
            return {"files": _gemini_register_files_from_uris(body)}
        return {"file": _gemini_register_file(body)}
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="INVALID_ARGUMENT")
    except Exception as exc:
        log.exception("Gemini file register failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.get("/v1/files/{file_id:path}:download")
@app.get("/v1beta/files/{file_id:path}:download")
async def gemini_download_file(file_id: str):
    meta = _gemini_get_file_meta(file_id)
    if not meta:
        return _gemini_error_response(f"File '{file_id}' not found.", status_code=404, status="NOT_FOUND")
    path = Path(str(meta.get("path") or ""))
    if not path.is_file():
        return _gemini_error_response(f"File '{file_id}' has no local media blob.", status_code=404, status="NOT_FOUND")
    return Response(content=path.read_bytes(), media_type=meta.get("mimeType") or "application/octet-stream")


@app.get("/v1/files/{file_id:path}")
@app.get("/v1beta/files/{file_id:path}")
async def gemini_get_file(file_id: str):
    meta = _gemini_get_file_meta(file_id)
    if not meta:
        return _gemini_error_response(f"File '{file_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_file_resource(meta)


@app.delete("/v1/files/{file_id:path}")
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


async def _gemini_upload_file_response(request: Request, upload_version: str) -> Any:
    """Gemini-compatible simple media/multipart file upload."""
    try:
        if _is_gemini_metadata_file_create(request):
            body = _gemini_normalize_request(await request.json())
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
            return {"file": _gemini_register_file(body)}
        if _is_gemini_resumable_upload_start(request):
            return await _gemini_start_resumable_upload(request, upload_version=upload_version)
        file_resource = await _gemini_upload_file_from_request(request)
        return {"file": file_resource}
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="INVALID_ARGUMENT")
    except Exception as exc:
        log.exception("Gemini file upload failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/files")
@app.post("/upload/v1/files")
async def gemini_upload_file_v1(request: Request):
    return await _gemini_upload_file_response(request, upload_version="v1")


@app.post("/v1beta/files")
@app.post("/upload/v1beta/files")
async def gemini_upload_file(request: Request):
    return await _gemini_upload_file_response(request, upload_version="v1beta")


@app.get("/v1/generatedFiles")
@app.get("/v1beta/generatedFiles")
async def gemini_list_generated_files(request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    files = [_gemini_generated_file_resource(meta) for meta in _gemini_load_generated_files_index().values()]
    files.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"generatedFiles": files[start:end], "nextPageToken": str(end) if end < len(files) else ""}


@app.get("/v1/generatedFiles/operations")
@app.get("/v1beta/generatedFiles/operations")
async def gemini_list_generated_file_operations(request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    return _gemini_operation_list_response(_gemini_operations_for_scope("generatedFile"), pageSize, pageToken)


@app.get("/v1/generatedFiles/{generated_file_id:path}:download")
@app.get("/v1beta/generatedFiles/{generated_file_id:path}:download")
async def gemini_download_generated_file(generated_file_id: str):
    meta = _gemini_get_generated_file_meta(generated_file_id)
    if not meta:
        return _gemini_error_response(f"Generated file '{generated_file_id}' not found.", status_code=404, status="NOT_FOUND")
    path = Path(str(meta.get("path") or ""))
    if not path.is_file():
        return _gemini_error_response(
            f"Generated file '{generated_file_id}' has no local media blob.",
            status_code=404,
            status="NOT_FOUND",
        )
    return Response(content=path.read_bytes(), media_type=meta.get("mimeType") or "application/octet-stream")


@app.get("/v1/generatedFiles/operations/{operation_id:path}")
@app.get("/v1beta/generatedFiles/operations/{operation_id:path}")
async def gemini_get_generated_file_operation(operation_id: str):
    operation = _gemini_get_scoped_operation("generatedFile", None, operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.get("/v1/generatedFiles/{generated_file_id:path}/operations/{operation_id:path}")
@app.get("/v1beta/generatedFiles/{generated_file_id:path}/operations/{operation_id:path}")
async def gemini_get_generated_file_scoped_operation(generated_file_id: str, operation_id: str):
    generated_file_name = _gemini_generated_file_name(generated_file_id)
    operation = _gemini_get_scoped_operation("generatedFile", generated_file_name, operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.post("/v1/generatedFiles/operations/{operation_id:path}:wait")
@app.post("/v1beta/generatedFiles/operations/{operation_id:path}:wait")
async def gemini_wait_generated_file_operation(operation_id: str):
    operation = _gemini_get_scoped_operation("generatedFile", None, operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.post("/v1/generatedFiles/operations/{operation_id:path}:cancel")
@app.post("/v1beta/generatedFiles/operations/{operation_id:path}:cancel")
async def gemini_cancel_generated_file_operation(operation_id: str):
    operation = _gemini_get_scoped_operation("generatedFile", None, operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    _gemini_cancel_operation(operation)
    return JSONResponse({})


@app.delete("/v1/generatedFiles/operations/{operation_id:path}")
@app.delete("/v1beta/generatedFiles/operations/{operation_id:path}")
async def gemini_delete_generated_file_operation(operation_id: str):
    operation = _gemini_get_scoped_operation("generatedFile", None, operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    index = _gemini_load_operations_index()
    index.pop(operation["name"], None)
    _gemini_save_operations_index(index)
    return JSONResponse({})


@app.get("/v1/generatedFiles/{generated_file_id:path}")
@app.get("/v1beta/generatedFiles/{generated_file_id:path}")
async def gemini_get_generated_file(generated_file_id: str):
    meta = _gemini_get_generated_file_meta(generated_file_id)
    if not meta:
        return _gemini_error_response(f"Generated file '{generated_file_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_generated_file_resource(meta)


@app.delete("/v1/generatedFiles/{generated_file_id:path}")
@app.delete("/v1beta/generatedFiles/{generated_file_id:path}")
async def gemini_delete_generated_file(generated_file_id: str):
    name = _gemini_generated_file_name(generated_file_id)
    index = _gemini_load_generated_files_index()
    meta = index.get(name)
    if not meta:
        return _gemini_error_response(f"Generated file '{generated_file_id}' not found.", status_code=404, status="NOT_FOUND")
    index.pop(name, None)
    _gemini_save_generated_files_index(index)
    path = Path(str(meta.get("path") or ""))
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            log.warning("Failed to remove Gemini generatedFile blob: %s", path)
    return JSONResponse({})


@app.post("/upload/v1/files/{session_id}")
@app.post("/upload/v1beta/files/{session_id}")
@app.put("/upload/v1/files/{session_id}")
@app.put("/upload/v1beta/files/{session_id}")
async def gemini_resumable_upload(session_id: str, request: Request):
    try:
        return await _gemini_resumable_upload_command(session_id, request)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini resumable upload failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/cachedContents")
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


@app.get("/v1/cachedContents")
@app.get("/v1beta/cachedContents")
async def gemini_list_cached_contents(request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000, clamp_page_size=True)
    index = _gemini_load_cached_index()
    items = [_gemini_cached_resource(meta) for meta in index.values()]
    items.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"cachedContents": items[start:end], "nextPageToken": str(end) if end < len(items) else ""}


@app.get("/v1/cachedContents/{cache_id:path}")
@app.get("/v1beta/cachedContents/{cache_id:path}")
async def gemini_get_cached_content(cache_id: str):
    meta = _gemini_get_cached_meta(cache_id)
    if not meta:
        return _gemini_error_response(f"Cached content '{cache_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_cached_resource(meta)


@app.patch("/v1/cachedContents/{cache_id:path}")
@app.patch("/v1beta/cachedContents/{cache_id:path}")
async def gemini_patch_cached_content(cache_id: str, request: Request, updateMask: str | None = None):
    try:
        meta = _gemini_get_cached_meta(cache_id)
        if not meta:
            raise HTTPException(status_code=404, detail=f"Cached content '{cache_id}' not found.")
        body = _gemini_cached_body(await request.json())
        return _gemini_patch_cached_meta(meta, body, updateMask)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini cachedContents patch failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.delete("/v1/cachedContents/{cache_id:path}")
@app.delete("/v1beta/cachedContents/{cache_id:path}")
async def gemini_delete_cached_content(cache_id: str):
    meta = _gemini_get_cached_meta(cache_id)
    if not meta:
        return _gemini_error_response(f"Cached content '{cache_id}' not found.", status_code=404, status="NOT_FOUND")
    index = _gemini_load_cached_index()
    index.pop(meta["name"], None)
    _gemini_save_cached_index(index)
    return JSONResponse({})


@app.post("/v1/corpora")
@app.post("/v1beta/corpora")
async def gemini_create_corpus(request: Request):
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return _gemini_create_corpus(body)
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="INVALID_ARGUMENT")


@app.get("/v1/corpora")
@app.get("/v1beta/corpora")
async def gemini_list_corpora(request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    corpora = [_gemini_corpus_resource(meta) for meta in _gemini_load_corpora_index().values()]
    corpora.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"corpora": corpora[start:end], "nextPageToken": str(end) if end < len(corpora) else ""}


@app.post("/v1/corpora/{corpus_id}:query")
@app.post("/v1beta/corpora/{corpus_id}:query")
async def gemini_query_corpus(corpus_id: str, request: Request):
    meta = _gemini_load_corpora_index().get(_gemini_corpus_name(corpus_id))
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    body = _gemini_normalize_request(await request.json())
    query = str((body or {}).get("query") or (body or {}).get("text") or "")
    top_k = int((body or {}).get("resultsCount") or (body or {}).get("results_count") or 10)
    return _gemini_corpus_query(meta, query, top_k=top_k)


@app.get("/v1/corpora/{corpus_id}")
@app.get("/v1beta/corpora/{corpus_id}")
async def gemini_get_corpus(corpus_id: str):
    meta = _gemini_load_corpora_index().get(_gemini_corpus_name(corpus_id))
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_corpus_resource(meta)


@app.get("/v1/corpora/{corpus_id:path}/operations/{operation_id:path}")
@app.get("/v1beta/corpora/{corpus_id:path}/operations/{operation_id:path}")
async def gemini_get_corpus_operation(corpus_id: str, operation_id: str):
    operation = _gemini_get_scoped_operation("corpus", _gemini_corpus_name(corpus_id), operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.patch("/v1/corpora/{corpus_id}")
@app.patch("/v1beta/corpora/{corpus_id}")
async def gemini_patch_corpus(corpus_id: str, request: Request, updateMask: str | None = None):
    index = _gemini_load_corpora_index()
    name = _gemini_corpus_name(corpus_id)
    meta = index.get(name)
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    body = _gemini_normalize_request(await request.json())
    if isinstance(body, dict):
        updateMask = updateMask or body.pop("updateMask", None)
        fields = _gemini_simple_update_fields(
            update_mask=updateMask,
            body=body,
            allowed={"displayName"},
            aliases={"display_name": "displayName", "corpus.display_name": "displayName", "corpus.displayName": "displayName"},
            resource="corpora",
        )
        for key in fields:
            meta[key] = body[key]
    meta["updateTime"] = _gemini_now_iso()
    index[name] = meta
    _gemini_save_corpora_index(index)
    return _gemini_corpus_resource(meta)


@app.delete("/v1/corpora/{corpus_id}")
@app.delete("/v1beta/corpora/{corpus_id}")
async def gemini_delete_corpus(corpus_id: str):
    name = _gemini_corpus_name(corpus_id)
    index = _gemini_load_corpora_index()
    if name not in index:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    index.pop(name, None)
    _gemini_save_corpora_index(index)
    return JSONResponse({})


@app.post("/v1/corpora/{corpus_id}/documents")
@app.post("/v1beta/corpora/{corpus_id}/documents")
async def gemini_create_corpus_document(corpus_id: str, request: Request):
    index = _gemini_load_corpora_index()
    corpus_name = _gemini_corpus_name(corpus_id)
    meta = index.get(corpus_name)
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    try:
        body = _gemini_document_create_body(await request.json(), request)
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="INVALID_ARGUMENT")
    doc_id = str(body.get("documentId") or body.get("document_id") or body.get("name") or "").strip().strip("/")
    if doc_id.startswith(corpus_name + "/documents/"):
        doc_id = doc_id.rsplit("/documents/", 1)[-1]
    if not doc_id:
        doc_id = "doc_" + uuid.uuid4().hex
    now = _gemini_now_iso()
    doc = {
        "name": f"{corpus_name}/documents/{doc_id}",
        "displayName": body.get("displayName") or body.get("display_name") or doc_id,
        "customMetadata": body.get("customMetadata") or body.get("custom_metadata") or [],
        "createTime": now,
        "updateTime": now,
        "chunks": {},
    }
    meta.setdefault("documents", {})[doc["name"]] = doc
    meta["updateTime"] = now
    index[corpus_name] = meta
    _gemini_save_corpora_index(index)
    return _gemini_document_resource(doc)


@app.get("/v1/corpora/{corpus_id}/documents")
@app.get("/v1beta/corpora/{corpus_id}/documents")
async def gemini_list_corpus_documents(corpus_id: str, request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    meta = _gemini_load_corpora_index().get(_gemini_corpus_name(corpus_id))
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    docs = [_gemini_document_resource(doc) for doc in (meta.get("documents") or {}).values()]
    docs.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"documents": docs[start:end], "nextPageToken": str(end) if end < len(docs) else ""}


@app.post("/v1/corpora/{corpus_id}/documents/{document_id:path}:query")
@app.post("/v1beta/corpora/{corpus_id}/documents/{document_id:path}:query")
async def gemini_query_corpus_document(corpus_id: str, document_id: str, request: Request):
    meta = _gemini_load_corpora_index().get(_gemini_corpus_name(corpus_id))
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    doc = (meta.get("documents") or {}).get(_gemini_corpus_document_name(corpus_id, document_id))
    if not doc:
        return _gemini_error_response(f"Document '{document_id}' not found.", status_code=404, status="NOT_FOUND")
    body = _gemini_normalize_request(await request.json())
    tmp_meta = {"documents": {doc["name"]: doc}}
    query = str((body or {}).get("query") or (body or {}).get("text") or "")
    top_k = int((body or {}).get("resultsCount") or (body or {}).get("results_count") or 10)
    return _gemini_corpus_query(tmp_meta, query, top_k=top_k)


def _gemini_get_corpus_doc_for_chunks(corpus_id: str, document_id: str) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    index = _gemini_load_corpora_index()
    corpus_name = _gemini_corpus_name(corpus_id)
    meta = index.get(corpus_name)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Corpus '{corpus_id}' not found.")
    doc_name = _gemini_corpus_document_name(corpus_id, document_id)
    doc = (meta.get("documents") or {}).get(doc_name)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return index, corpus_name, doc, doc_name


def _gemini_document_create_body(raw_body: Any, request: Request) -> dict[str, Any]:
    body = _gemini_normalize_request(raw_body)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    document = body.get("document") if isinstance(body.get("document"), dict) else body
    merged = dict(document)
    for key in ("documentId", "displayName", "customMetadata"):
        if body.get(key) is not None:
            merged[key] = body[key]
    query = request.query_params
    document_id = query.get("documentId") or query.get("document_id")
    if document_id:
        merged["documentId"] = document_id
    return _gemini_normalize_request(merged)


def _gemini_chunk_body(raw_body: Any, request: Request | None = None) -> dict[str, Any]:
    body = _gemini_normalize_request(raw_body)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    chunk = body.get("chunk") if isinstance(body.get("chunk"), dict) else body
    merged = dict(chunk)
    for key in ("chunkId", "name", "data", "customMetadata", "text", "createTime", "updateMask"):
        if body.get(key) is not None:
            merged[key] = body[key]
    if request is not None:
        query = request.query_params
        chunk_id = query.get("chunkId") or query.get("chunk_id")
        if chunk_id:
            merged["chunkId"] = chunk_id
        update_mask = query.get("updateMask") or query.get("update_mask")
        if update_mask:
            merged["updateMask"] = update_mask
    return _gemini_normalize_request(merged)


def _gemini_upsert_corpus_chunk(
    doc: dict[str, Any],
    doc_name: str,
    body: dict[str, Any],
    *,
    update_mask: str | None = None,
) -> dict[str, Any]:
    body = _gemini_normalize_request(body)
    chunk_id = str(body.get("chunkId") or body.get("chunk_id") or body.get("name") or "").strip().strip("/")
    if chunk_id.startswith(doc_name + "/chunks/"):
        chunk_id = chunk_id.rsplit("/chunks/", 1)[-1]
    if "/chunks/" in chunk_id:
        chunk_id = chunk_id.rsplit("/chunks/", 1)[-1]
    if not chunk_id:
        chunk_id = "chunk_" + uuid.uuid4().hex
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    text = body.get("text")
    if text is not None and "stringValue" not in data:
        data = {"stringValue": str(text)}
    now = _gemini_now_iso()
    chunk_name = f"{doc_name}/chunks/{chunk_id}"
    existing = dict((doc.get("chunks") or {}).get(chunk_name) or {})
    if update_mask:
        if not existing:
            raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found.")
        patch_body = dict(body)
        if data:
            patch_body["data"] = data
        fields = _gemini_simple_update_fields(
            update_mask=update_mask,
            body=patch_body,
            allowed={"data", "customMetadata"},
            aliases={
                "chunk.data": "data",
                "chunk.custom_metadata": "customMetadata",
                "chunk.customMetadata": "customMetadata",
                "custom_metadata": "customMetadata",
            },
            resource="chunks",
        )
        chunk = existing
        for key in fields:
            if update_mask and key not in patch_body:
                raise HTTPException(status_code=400, detail=f"{key} is required by updateMask.")
            chunk[key] = patch_body[key]
        chunk["updateTime"] = now
    else:
        chunk = {
            "name": chunk_name,
            "data": data,
            "customMetadata": body.get("customMetadata") or body.get("custom_metadata") or [],
            "createTime": existing.get("createTime") or body.get("createTime") or now,
            "updateTime": now,
        }
    doc.setdefault("chunks", {})[chunk["name"]] = chunk
    doc["updateTime"] = now
    return chunk


@app.post("/v1/corpora/{corpus_id}/documents/{document_id:path}/chunks:batchCreate")
@app.post("/v1beta/corpora/{corpus_id}/documents/{document_id:path}/chunks:batchCreate")
async def gemini_batch_create_corpus_chunks(corpus_id: str, document_id: str, request: Request):
    try:
        index, corpus_name, doc, doc_name = _gemini_get_corpus_doc_for_chunks(corpus_id, document_id)
        body = _gemini_normalize_request(await request.json())
        raw_requests = (body or {}).get("requests") or []
        chunks = []
        for item in raw_requests if isinstance(raw_requests, list) else []:
            chunk_body = _gemini_chunk_body(item)
            if isinstance(chunk_body, dict):
                chunks.append(_gemini_chunk_resource(_gemini_upsert_corpus_chunk(doc, doc_name, chunk_body)))
        index[corpus_name]["documents"][doc_name] = doc
        _gemini_save_corpora_index(index)
        return {"chunks": chunks}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.post("/v1/corpora/{corpus_id}/documents/{document_id:path}/chunks:batchUpdate")
@app.post("/v1beta/corpora/{corpus_id}/documents/{document_id:path}/chunks:batchUpdate")
async def gemini_batch_update_corpus_chunks(corpus_id: str, document_id: str, request: Request):
    try:
        index, corpus_name, doc, doc_name = _gemini_get_corpus_doc_for_chunks(corpus_id, document_id)
        body = _gemini_normalize_request(await request.json())
        raw_requests = (body or {}).get("requests") or []
        chunks = []
        for item in raw_requests if isinstance(raw_requests, list) else []:
            chunk_body = _gemini_chunk_body(item)
            update_mask = chunk_body.pop("updateMask", None)
            chunks.append(_gemini_chunk_resource(_gemini_upsert_corpus_chunk(
                doc,
                doc_name,
                chunk_body,
                update_mask=update_mask,
            )))
        index[corpus_name]["documents"][doc_name] = doc
        _gemini_save_corpora_index(index)
        return {"chunks": chunks}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.post("/v1/corpora/{corpus_id}/documents/{document_id:path}/chunks:batchDelete")
@app.post("/v1beta/corpora/{corpus_id}/documents/{document_id:path}/chunks:batchDelete")
async def gemini_batch_delete_corpus_chunks(corpus_id: str, document_id: str, request: Request):
    try:
        index, corpus_name, doc, doc_name = _gemini_get_corpus_doc_for_chunks(corpus_id, document_id)
        body = _gemini_normalize_request(await request.json())
        names = (body or {}).get("names") or []
        if not isinstance(names, list):
            names = []
        for name in names:
            doc.setdefault("chunks", {}).pop(_gemini_corpus_chunk_name(corpus_id, document_id, str(name)), None)
        doc["updateTime"] = _gemini_now_iso()
        index[corpus_name]["documents"][doc_name] = doc
        _gemini_save_corpora_index(index)
        return JSONResponse({})
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.post("/v1/corpora/{corpus_id}/documents/{document_id:path}/chunks")
@app.post("/v1beta/corpora/{corpus_id}/documents/{document_id:path}/chunks")
async def gemini_create_corpus_chunk(corpus_id: str, document_id: str, request: Request):
    try:
        index, corpus_name, doc, doc_name = _gemini_get_corpus_doc_for_chunks(corpus_id, document_id)
        body = _gemini_chunk_body(await request.json(), request)
        chunk = _gemini_upsert_corpus_chunk(doc, doc_name, body)
        index[corpus_name]["documents"][doc_name] = doc
        _gemini_save_corpora_index(index)
        return _gemini_chunk_resource(chunk)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.get("/v1/corpora/{corpus_id}/documents/{document_id:path}/chunks")
@app.get("/v1beta/corpora/{corpus_id}/documents/{document_id:path}/chunks")
async def gemini_list_corpus_chunks(corpus_id: str, document_id: str, request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    try:
        _index, _corpus_name, doc, _doc_name = _gemini_get_corpus_doc_for_chunks(corpus_id, document_id)
        chunks = [_gemini_chunk_resource(chunk) for chunk in (doc.get("chunks") or {}).values()]
        chunks.sort(key=lambda item: item.get("createTime") or "")
        start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
        end = start + pageSize
        return {"chunks": chunks[start:end], "nextPageToken": str(end) if end < len(chunks) else ""}
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="NOT_FOUND")


@app.get("/v1/corpora/{corpus_id}/documents/{document_id:path}/chunks/{chunk_id:path}")
@app.get("/v1beta/corpora/{corpus_id}/documents/{document_id:path}/chunks/{chunk_id:path}")
async def gemini_get_corpus_chunk(corpus_id: str, document_id: str, chunk_id: str):
    try:
        _index, _corpus_name, doc, _doc_name = _gemini_get_corpus_doc_for_chunks(corpus_id, document_id)
        chunk = (doc.get("chunks") or {}).get(_gemini_corpus_chunk_name(corpus_id, document_id, chunk_id))
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found.")
        return _gemini_chunk_resource(chunk)
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="NOT_FOUND")


@app.patch("/v1/corpora/{corpus_id}/documents/{document_id:path}/chunks/{chunk_id:path}")
@app.patch("/v1beta/corpora/{corpus_id}/documents/{document_id:path}/chunks/{chunk_id:path}")
async def gemini_patch_corpus_chunk(corpus_id: str, document_id: str, chunk_id: str, request: Request, updateMask: str | None = None):
    try:
        index, corpus_name, doc, doc_name = _gemini_get_corpus_doc_for_chunks(corpus_id, document_id)
        body = _gemini_chunk_body(await request.json())
        updateMask = updateMask or body.pop("updateMask", None)
        chunk_name = _gemini_corpus_chunk_name(corpus_id, document_id, chunk_id)
        current = dict((doc.get("chunks") or {}).get(chunk_name) or {})
        if not current:
            raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found.")
        fields = _gemini_simple_update_fields(
            update_mask=updateMask,
            body=body,
            allowed={"data", "customMetadata"},
            aliases={
                "chunk.data": "data",
                "chunk.custom_metadata": "customMetadata",
                "chunk.customMetadata": "customMetadata",
                "custom_metadata": "customMetadata",
            },
            resource="chunks",
        )
        for key in fields:
            current[key] = body[key]
        current["updateTime"] = _gemini_now_iso()
        doc.setdefault("chunks", {})[chunk_name] = current
        index[corpus_name]["documents"][doc_name] = doc
        _gemini_save_corpora_index(index)
        return _gemini_chunk_resource(current)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.delete("/v1/corpora/{corpus_id}/documents/{document_id:path}/chunks/{chunk_id:path}")
@app.delete("/v1beta/corpora/{corpus_id}/documents/{document_id:path}/chunks/{chunk_id:path}")
async def gemini_delete_corpus_chunk(corpus_id: str, document_id: str, chunk_id: str):
    try:
        index, corpus_name, doc, doc_name = _gemini_get_corpus_doc_for_chunks(corpus_id, document_id)
        chunk_name = _gemini_corpus_chunk_name(corpus_id, document_id, chunk_id)
        if chunk_name not in (doc.get("chunks") or {}):
            raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found.")
        doc["chunks"].pop(chunk_name, None)
        doc["updateTime"] = _gemini_now_iso()
        index[corpus_name]["documents"][doc_name] = doc
        _gemini_save_corpora_index(index)
        return JSONResponse({})
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status="NOT_FOUND")


@app.get("/v1/corpora/{corpus_id}/documents/{document_id:path}")
@app.get("/v1beta/corpora/{corpus_id}/documents/{document_id:path}")
async def gemini_get_corpus_document(corpus_id: str, document_id: str):
    meta = _gemini_load_corpora_index().get(_gemini_corpus_name(corpus_id))
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    doc = (meta.get("documents") or {}).get(_gemini_corpus_document_name(corpus_id, document_id))
    if not doc:
        return _gemini_error_response(f"Document '{document_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_document_resource(doc)


@app.patch("/v1/corpora/{corpus_id}/documents/{document_id:path}")
@app.patch("/v1beta/corpora/{corpus_id}/documents/{document_id:path}")
async def gemini_patch_corpus_document(corpus_id: str, document_id: str, request: Request, updateMask: str | None = None):
    index = _gemini_load_corpora_index()
    corpus_name = _gemini_corpus_name(corpus_id)
    meta = index.get(corpus_name)
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    doc_name = _gemini_corpus_document_name(corpus_id, document_id)
    doc = (meta.get("documents") or {}).get(doc_name)
    if not doc:
        return _gemini_error_response(f"Document '{document_id}' not found.", status_code=404, status="NOT_FOUND")
    body = _gemini_normalize_request(await request.json())
    if isinstance(body, dict):
        updateMask = updateMask or body.pop("updateMask", None)
        fields = _gemini_simple_update_fields(
            update_mask=updateMask,
            body=body,
            allowed={"displayName", "customMetadata"},
            aliases={
                "document.display_name": "displayName",
                "document.displayName": "displayName",
                "display_name": "displayName",
                "document.custom_metadata": "customMetadata",
                "document.customMetadata": "customMetadata",
                "custom_metadata": "customMetadata",
            },
            resource="documents",
        )
        for key in fields:
            if key in body:
                doc[key] = body[key]
    doc["updateTime"] = _gemini_now_iso()
    meta.setdefault("documents", {})[doc_name] = doc
    index[corpus_name] = meta
    _gemini_save_corpora_index(index)
    return _gemini_document_resource(doc)


@app.delete("/v1/corpora/{corpus_id}/documents/{document_id:path}")
@app.delete("/v1beta/corpora/{corpus_id}/documents/{document_id:path}")
async def gemini_delete_corpus_document(corpus_id: str, document_id: str):
    index = _gemini_load_corpora_index()
    corpus_name = _gemini_corpus_name(corpus_id)
    meta = index.get(corpus_name)
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    doc_name = _gemini_corpus_document_name(corpus_id, document_id)
    if doc_name not in (meta.get("documents") or {}):
        return _gemini_error_response(f"Document '{document_id}' not found.", status_code=404, status="NOT_FOUND")
    meta["documents"].pop(doc_name, None)
    meta["updateTime"] = _gemini_now_iso()
    index[corpus_name] = meta
    _gemini_save_corpora_index(index)
    return JSONResponse({})


@app.get("/v1/corpora/{corpus_id}/permissions")
@app.get("/v1beta/corpora/{corpus_id}/permissions")
async def gemini_list_corpus_permissions(corpus_id: str, request: Request):
    meta = _gemini_load_corpora_index().get(_gemini_corpus_name(corpus_id))
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_permission_list_response(meta.get("permissions") or {}, request)


@app.post("/v1/corpora/{corpus_id}/permissions")
@app.post("/v1beta/corpora/{corpus_id}/permissions")
async def gemini_create_corpus_permission(corpus_id: str, request: Request):
    index = _gemini_load_corpora_index()
    corpus_name = _gemini_corpus_name(corpus_id)
    meta = index.get(corpus_name)
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    body = _gemini_permission_body(await request.json())
    pid = "perm_" + uuid.uuid4().hex
    perm_name = f"{corpus_name}/permissions/{pid}"
    perm = _gemini_permission_public_resource(perm_name, body or {})
    meta.setdefault("permissions", {})[perm["name"]] = perm
    index[corpus_name] = meta
    _gemini_save_corpora_index(index)
    return perm


@app.get("/v1/corpora/{corpus_id}/permissions/{permission_id:path}")
@app.get("/v1beta/corpora/{corpus_id}/permissions/{permission_id:path}")
async def gemini_get_corpus_permission(corpus_id: str, permission_id: str):
    meta = _gemini_load_corpora_index().get(_gemini_corpus_name(corpus_id))
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    perm = (meta.get("permissions") or {}).get(_gemini_corpus_permission_name(corpus_id, permission_id))
    if not perm:
        return _gemini_error_response(f"Permission '{permission_id}' not found.", status_code=404, status="NOT_FOUND")
    return perm


@app.patch("/v1/corpora/{corpus_id}/permissions/{permission_id:path}")
@app.patch("/v1beta/corpora/{corpus_id}/permissions/{permission_id:path}")
async def gemini_patch_corpus_permission(corpus_id: str, permission_id: str, request: Request, updateMask: str | None = None):
    index = _gemini_load_corpora_index()
    corpus_name = _gemini_corpus_name(corpus_id)
    meta = index.get(corpus_name)
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    perm_name = _gemini_corpus_permission_name(corpus_id, permission_id)
    perm = (meta.get("permissions") or {}).get(perm_name)
    if not perm:
        return _gemini_error_response(f"Permission '{permission_id}' not found.", status_code=404, status="NOT_FOUND")
    raw_body = await request.json()
    body = _gemini_permission_body(raw_body)
    updateMask = updateMask or request.query_params.get("update_mask")
    if isinstance(raw_body, dict):
        updateMask = updateMask or raw_body.get("updateMask")
    if isinstance(body, dict):
        updateMask = updateMask or body.pop("updateMask", None)
        fields = _gemini_permission_update_fields(updateMask, body)
        for key in fields:
            if key in body:
                perm[key] = body[key]
    meta["updateTime"] = _gemini_now_iso()
    meta.setdefault("permissions", {})[perm_name] = perm
    index[corpus_name] = meta
    _gemini_save_corpora_index(index)
    return perm


@app.delete("/v1/corpora/{corpus_id}/permissions/{permission_id:path}")
@app.delete("/v1beta/corpora/{corpus_id}/permissions/{permission_id:path}")
async def gemini_delete_corpus_permission(corpus_id: str, permission_id: str):
    index = _gemini_load_corpora_index()
    corpus_name = _gemini_corpus_name(corpus_id)
    meta = index.get(corpus_name)
    if not meta:
        return _gemini_error_response(f"Corpus '{corpus_id}' not found.", status_code=404, status="NOT_FOUND")
    perm_name = _gemini_corpus_permission_name(corpus_id, permission_id)
    if perm_name not in (meta.get("permissions") or {}):
        return _gemini_error_response(f"Permission '{permission_id}' not found.", status_code=404, status="NOT_FOUND")
    meta["permissions"].pop(perm_name, None)
    index[corpus_name] = meta
    _gemini_save_corpora_index(index)
    return JSONResponse({})


@app.post("/v1/fileSearchStores")
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


@app.get("/v1/fileSearchStores")
@app.get("/v1beta/fileSearchStores")
async def gemini_list_file_search_stores(request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    stores = [_gemini_fss_resource(meta) for meta in _gemini_load_fss_index().values()]
    stores.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"fileSearchStores": stores[start:end], "nextPageToken": str(end) if end < len(stores) else ""}


@app.get("/v1/fileSearchStores/{store_id}")
@app.get("/v1beta/fileSearchStores/{store_id}")
async def gemini_get_file_search_store(store_id: str):
    meta = _gemini_get_fss_meta(store_id)
    if not meta:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_fss_resource(meta)


@app.delete("/v1/fileSearchStores/{store_id}")
@app.delete("/v1beta/fileSearchStores/{store_id}")
async def gemini_delete_file_search_store(store_id: str, request: Request):
    name = _gemini_fss_name(store_id)
    index = _gemini_load_fss_index()
    meta = index.get(name)
    if not meta:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    force = _gemini_query_bool(request, "force", "force")
    if (meta.get("documents") or {}) and not force:
        return _gemini_error_response(
            "File search store contains documents. Set force=true to delete it.",
            status_code=400,
            status="FAILED_PRECONDITION",
            field="force",
        )
    index.pop(name, None)
    _gemini_save_fss_index(index)
    return JSONResponse({})


@app.post("/v1/fileSearchStores/{store_id}:importFile")
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


@app.post("/upload/v1/fileSearchStores/{store_id}:uploadToFileSearchStore")
@app.post("/upload/v1beta/fileSearchStores/{store_id}:uploadToFileSearchStore")
@app.post("/v1/fileSearchStores/{store_id}:uploadToFileSearchStore")
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
        elif content_type.split(";", 1)[0].strip().lower() == "application/json":
            try:
                decoded = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                decoded = {}
            if isinstance(decoded, dict):
                metadata = decoded
                raw_content = decoded.get("content", decoded.get("text", decoded.get("data", "")))
                if isinstance(raw_content, str):
                    media = raw_content.encode("utf-8")
                elif isinstance(raw_content, (bytes, bytearray)):
                    media = bytes(raw_content)
                else:
                    media = json.dumps(raw_content, ensure_ascii=False).encode("utf-8")
                media_type = (
                    decoded.get("mimeType")
                    or decoded.get("mime_type")
                    or (decoded.get("file", {}) if isinstance(decoded.get("file"), dict) else {}).get("mimeType")
                    or "text/plain"
                )
        metadata = _gemini_fss_document_body(metadata) if isinstance(metadata, dict) else {}
        file_meta = metadata
        display_name = request.query_params.get("displayName")
        if isinstance(file_meta, dict):
            display_name = file_meta.get("displayName") or display_name
            media_type = file_meta.get("mimeType") or media_type
        custom_metadata = metadata.get("customMetadata") or metadata.get("metadata")
        chunking_config = metadata.get("chunkingConfig") if isinstance(metadata.get("chunkingConfig"), dict) else None
        if isinstance(file_meta, dict):
            custom_metadata = (
                file_meta.get("customMetadata")
                or file_meta.get("metadata")
                or custom_metadata
            )
            if isinstance(file_meta.get("chunkingConfig"), dict):
                chunking_config = file_meta["chunkingConfig"]
        document = _gemini_store_document(
            store_id,
            display_name=display_name,
            mime_type=media_type,
            content=media,
            custom_metadata=custom_metadata,
            chunking_config=chunking_config,
        )
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


@app.get("/v1/fileSearchStores/{store_id}/documents")
@app.get("/v1beta/fileSearchStores/{store_id}/documents")
async def gemini_list_file_search_documents(store_id: str, request: Request):
    meta = _gemini_get_fss_meta(store_id)
    if not meta:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    query = request.query_params
    page_size_raw = query.get("pageSize") or query.get("page_size") or "20"
    try:
        page_size = max(1, min(20, int(page_size_raw)))
    except (TypeError, ValueError):
        page_size = 20
    page_token = query.get("pageToken") or query.get("page_token") or "0"
    try:
        start = max(0, int(page_token))
    except (TypeError, ValueError):
        start = 0
    docs = [_gemini_document_resource(doc) for doc in (meta.get("documents") or {}).values()]
    docs.sort(key=lambda item: item.get("createTime") or "")
    end = start + page_size
    return {"documents": docs[start:end], "nextPageToken": str(end) if end < len(docs) else ""}


@app.get("/v1/fileSearchStores/{store_id}/media/{document_id:path}")
@app.get("/v1beta/fileSearchStores/{store_id}/media/{document_id:path}")
async def gemini_download_file_search_document_media(store_id: str, document_id: str):
    meta = _gemini_get_fss_meta(store_id)
    if not meta:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    doc = (meta.get("documents") or {}).get(_gemini_document_name(store_id, document_id))
    if not doc:
        return _gemini_error_response(f"Document '{document_id}' not found.", status_code=404, status="NOT_FOUND")
    try:
        content = base64.b64decode(str(doc.get("content") or ""))
    except Exception:
        content = b""
    return Response(content=content, media_type=doc.get("mimeType") or "application/octet-stream")


@app.get("/v1/fileSearchStores/{store_id}/operations")
@app.get("/v1beta/fileSearchStores/{store_id}/operations")
async def gemini_list_file_search_operations(store_id: str, request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    if not _gemini_get_fss_meta(store_id):
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_operation_list_response(
        _gemini_operations_for_scope("fileSearchStore", _gemini_fss_name(store_id)),
        pageSize,
        pageToken,
    )


@app.get("/v1/fileSearchStores/{store_id}/operations/{operation_id:path}")
@app.get("/v1beta/fileSearchStores/{store_id}/operations/{operation_id:path}")
async def gemini_get_file_search_operation(store_id: str, operation_id: str):
    operation = _gemini_get_scoped_operation("fileSearchStore", _gemini_fss_name(store_id), operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.get("/v1/fileSearchStores/{store_id}/upload/operations/{operation_id:path}")
@app.get("/v1beta/fileSearchStores/{store_id}/upload/operations/{operation_id:path}")
async def gemini_get_file_search_upload_operation(store_id: str, operation_id: str):
    return await gemini_get_file_search_operation(store_id, operation_id)


@app.post("/v1/fileSearchStores/{store_id}/operations/{operation_id:path}:wait")
@app.post("/v1beta/fileSearchStores/{store_id}/operations/{operation_id:path}:wait")
async def gemini_wait_file_search_operation(store_id: str, operation_id: str):
    operation = _gemini_get_scoped_operation("fileSearchStore", _gemini_fss_name(store_id), operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.post("/v1/fileSearchStores/{store_id}/upload/operations/{operation_id:path}:wait")
@app.post("/v1beta/fileSearchStores/{store_id}/upload/operations/{operation_id:path}:wait")
async def gemini_wait_file_search_upload_operation(store_id: str, operation_id: str):
    return await gemini_wait_file_search_operation(store_id, operation_id)


@app.post("/v1/fileSearchStores/{store_id}/operations/{operation_id:path}:cancel")
@app.post("/v1beta/fileSearchStores/{store_id}/operations/{operation_id:path}:cancel")
async def gemini_cancel_file_search_operation(store_id: str, operation_id: str):
    operation = _gemini_get_scoped_operation("fileSearchStore", _gemini_fss_name(store_id), operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    _gemini_cancel_operation(operation)
    return JSONResponse({})


@app.post("/v1/fileSearchStores/{store_id}/upload/operations/{operation_id:path}:cancel")
@app.post("/v1beta/fileSearchStores/{store_id}/upload/operations/{operation_id:path}:cancel")
async def gemini_cancel_file_search_upload_operation(store_id: str, operation_id: str):
    return await gemini_cancel_file_search_operation(store_id, operation_id)


@app.delete("/v1/fileSearchStores/{store_id}/operations/{operation_id:path}")
@app.delete("/v1beta/fileSearchStores/{store_id}/operations/{operation_id:path}")
async def gemini_delete_file_search_operation(store_id: str, operation_id: str):
    operation = _gemini_get_scoped_operation("fileSearchStore", _gemini_fss_name(store_id), operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    name = operation["name"]
    index = _gemini_load_operations_index()
    index.pop(name, None)
    _gemini_save_operations_index(index)
    return JSONResponse({})


@app.delete("/v1/fileSearchStores/{store_id}/upload/operations/{operation_id:path}")
@app.delete("/v1beta/fileSearchStores/{store_id}/upload/operations/{operation_id:path}")
async def gemini_delete_file_search_upload_operation(store_id: str, operation_id: str):
    return await gemini_delete_file_search_operation(store_id, operation_id)


@app.get("/v1/fileSearchStores/{store_id}/documents/{document_id:path}")
@app.get("/v1beta/fileSearchStores/{store_id}/documents/{document_id:path}")
async def gemini_get_file_search_document(store_id: str, document_id: str):
    meta = _gemini_get_fss_meta(store_id)
    if not meta:
        return _gemini_error_response(f"File search store '{store_id}' not found.", status_code=404, status="NOT_FOUND")
    doc = (meta.get("documents") or {}).get(_gemini_document_name(store_id, document_id))
    if not doc:
        return _gemini_error_response(f"Document '{document_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_document_resource(doc)


@app.delete("/v1/fileSearchStores/{store_id}/documents/{document_id:path}")
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


@app.post("/v1/tunedModels")
@app.post("/v1beta/tunedModels")
async def gemini_create_tuned_model(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        query_tuned_model_id = request.query_params.get("tunedModelId") or request.query_params.get("tuned_model_id")
        if query_tuned_model_id and "tunedModelId" not in body and "tuned_model_id" not in body:
            body = {**body, "tunedModelId": query_tuned_model_id}
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


@app.get("/v1/tunedModels")
@app.get("/v1beta/tunedModels")
async def gemini_list_tuned_models(request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    models = [_gemini_tuned_resource(meta) for meta in _gemini_load_tuned_index().values()]
    models.sort(key=lambda item: item.get("createTime") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {"tunedModels": models[start:end], "nextPageToken": str(end) if end < len(models) else ""}


@app.get("/v1/tunedModels/{tuned_model_id:path}/operations")
@app.get("/v1beta/tunedModels/{tuned_model_id:path}/operations")
async def gemini_list_tuned_model_operations(tuned_model_id: str, request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    tuned_name = _gemini_tuned_name(tuned_model_id)
    return _gemini_operation_list_response(
        _gemini_operations_for_scope("tunedModel", tuned_name),
        pageSize,
        pageToken,
    )


@app.get("/v1/tunedModels/{tuned_model_id:path}/operations/{operation_id:path}")
@app.get("/v1beta/tunedModels/{tuned_model_id:path}/operations/{operation_id:path}")
async def gemini_get_tuned_model_operation(tuned_model_id: str, operation_id: str):
    operation = _gemini_get_scoped_operation("tunedModel", _gemini_tuned_name(tuned_model_id), operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.post("/v1/tunedModels/{tuned_model_id:path}/operations/{operation_id:path}:wait")
@app.post("/v1beta/tunedModels/{tuned_model_id:path}/operations/{operation_id:path}:wait")
async def gemini_wait_tuned_model_operation(tuned_model_id: str, operation_id: str):
    operation = _gemini_get_scoped_operation("tunedModel", _gemini_tuned_name(tuned_model_id), operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.post("/v1/tunedModels/{tuned_model_id:path}/operations/{operation_id:path}:cancel")
@app.post("/v1beta/tunedModels/{tuned_model_id:path}/operations/{operation_id:path}:cancel")
async def gemini_cancel_tuned_model_operation(tuned_model_id: str, operation_id: str):
    operation = _gemini_get_scoped_operation("tunedModel", _gemini_tuned_name(tuned_model_id), operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    _gemini_cancel_operation(operation)
    return JSONResponse({})


@app.delete("/v1/tunedModels/{tuned_model_id:path}/operations/{operation_id:path}")
@app.delete("/v1beta/tunedModels/{tuned_model_id:path}/operations/{operation_id:path}")
async def gemini_delete_tuned_model_operation(tuned_model_id: str, operation_id: str):
    operation = _gemini_get_scoped_operation("tunedModel", _gemini_tuned_name(tuned_model_id), operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    index = _gemini_load_operations_index()
    index.pop(operation["name"], None)
    _gemini_save_operations_index(index)
    return JSONResponse({})


@app.get("/v1/tunedModels/{tuned_model_id}")
@app.get("/v1beta/tunedModels/{tuned_model_id}")
async def gemini_get_tuned_model(tuned_model_id: str):
    meta = _gemini_get_tuned_meta(tuned_model_id)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_tuned_resource(meta)


@app.patch("/v1/tunedModels/{tuned_model_id}")
@app.patch("/v1beta/tunedModels/{tuned_model_id}")
async def gemini_patch_tuned_model(tuned_model_id: str, request: Request, updateMask: str | None = None):
    index = _gemini_load_tuned_index()
    name = _gemini_tuned_name(tuned_model_id)
    meta = index.get(name)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    raw_body = await request.json()
    body = _gemini_normalize_request(raw_body)
    if isinstance(body, dict):
        patch_body = body.get("tunedModel") if isinstance(body.get("tunedModel"), dict) else body
        patch_config = body.get("config") if isinstance(body.get("config"), dict) else {}
        merged_patch = dict(patch_config)
        merged_patch.update(patch_body)
        if isinstance(raw_body, dict):
            updateMask = updateMask or raw_body.get("updateMask") or raw_body.get("update_mask")
            raw_patch_body = raw_body.get("tunedModel") or raw_body.get("tuned_model")
            if isinstance(raw_patch_body, dict):
                updateMask = updateMask or raw_patch_body.get("updateMask") or raw_patch_body.get("update_mask")
        updateMask = updateMask or body.pop("updateMask", None) or merged_patch.pop("updateMask", None)
        fields = _gemini_tuned_update_fields(updateMask, merged_patch)
        for key in fields:
            if updateMask and key not in merged_patch:
                raise HTTPException(status_code=400, detail=f"{key} is required by updateMask.")
            if key in merged_patch:
                meta[key] = merged_patch[key]
        _gemini_tuned_scalar_fields(meta)
        meta["updateTime"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        index[name] = meta
        _gemini_save_tuned_index(index)
    return _gemini_tuned_resource(meta)


@app.delete("/v1/tunedModels/{tuned_model_id}")
@app.delete("/v1beta/tunedModels/{tuned_model_id}")
async def gemini_delete_tuned_model(tuned_model_id: str):
    index = _gemini_load_tuned_index()
    name = _gemini_tuned_name(tuned_model_id)
    if name not in index:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    index.pop(name, None)
    _gemini_save_tuned_index(index)
    return JSONResponse({})


@app.post("/v1/tunedModels/{tuned_model_id}:generateContent")
@app.post("/v1beta/tunedModels/{tuned_model_id}:generateContent")
async def gemini_tuned_generate_content(tuned_model_id: str, request: Request):
    try:
        model = _gemini_tuned_base_model(tuned_model_id)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        body = _gemini_apply_generate_config(body)
        body = _gemini_apply_response_format(body)
        body = _gemini_normalize_generate_body(body)
        body = _gemini_apply_cached_content(body)
        body = _gemini_apply_file_search(body)
        _gemini_reject_unsupported_builtin_tools(body)
        body = _gemini_inline_local_files(body)
        body.pop("model", None)
        data = await asyncio.to_thread(_get_client().generate_raw, request=body, model=str(model["antigravity_model"]))
        return JSONResponse(_gemini_finalize_generate_response(
            _gemini_unwrap_response(data),
            model_name=str(model.get("name") or tuned_model_id),
            request_body=body,
        ))
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini tuned model generateContent failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1/tunedModels/{tuned_model_id}:generateText")
@app.post("/v1beta/tunedModels/{tuned_model_id}:generateText")
async def gemini_tuned_generate_text(tuned_model_id: str, request: Request):
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return await _gemini_tuned_legacy_generate(tuned_model_id, body)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini tuned model generateText failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1/tunedModels/{tuned_model_id}:streamGenerateContent")
@app.post("/v1beta/tunedModels/{tuned_model_id}:streamGenerateContent")
async def gemini_tuned_stream_generate_content(tuned_model_id: str, request: Request):
    try:
        model = _gemini_tuned_base_model(tuned_model_id)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        body = _gemini_apply_generate_config(body)
        body = _gemini_apply_response_format(body)
        body = _gemini_normalize_generate_body(body)
        body = _gemini_apply_cached_content(body)
        body = _gemini_apply_file_search(body)
        _gemini_reject_unsupported_builtin_tools(body)
        body = _gemini_inline_local_files(body)
        body.pop("model", None)
        return _gemini_streaming_response(body=body, antigravity_model=str(model["antigravity_model"]))
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini tuned model streamGenerateContent failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1/tunedModels/{tuned_model_id}:batchGenerateContent")
@app.post("/v1beta/tunedModels/{tuned_model_id}:batchGenerateContent")
async def gemini_tuned_batch_generate_content(tuned_model_id: str, request: Request):
    try:
        body = _gemini_batch_body(_gemini_normalize_request(await request.json()))
        if isinstance(body, dict) and body.pop("_batchKind", None) == "embed":
            raise HTTPException(status_code=400, detail="batchGenerateContent does not accept embedContentBatch.")
        base_model = _gemini_tuned_base_model(tuned_model_id)
        operation, _batch = await _gemini_create_completed_batch(_gemini_model_name(base_model), body)
        return operation
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini tuned model batchGenerateContent failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1/tunedModels/{tuned_model_id}:countTokens")
@app.post("/v1beta/tunedModels/{tuned_model_id}:countTokens")
async def gemini_tuned_count_tokens(tuned_model_id: str, request: Request):
    try:
        _gemini_tuned_base_model(tuned_model_id)
        body = _gemini_normalize_request(await request.json())
        cached_tokens = _gemini_cached_content_tokens(body if isinstance(body, dict) else {})
        if isinstance(body, dict):
            body = _gemini_apply_file_search(_gemini_apply_cached_content(body))
        return _gemini_count_tokens_response(
            _gemini_count_tokens_request(body if isinstance(body, dict) else {}),
            cached_tokens=cached_tokens,
        )
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.post("/v1/tunedModels/{tuned_model_id}:computeTokens")
@app.post("/v1beta/tunedModels/{tuned_model_id}:computeTokens")
async def gemini_tuned_compute_tokens(tuned_model_id: str, request: Request):
    try:
        _gemini_tuned_base_model(tuned_model_id)
        body = _gemini_normalize_request(await request.json())
        if isinstance(body, dict):
            body = _gemini_apply_file_search(_gemini_apply_cached_content(body))
        return _gemini_compute_tokens_response(_gemini_count_tokens_request(body if isinstance(body, dict) else {}))
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.post("/v1/tunedModels/{tuned_model_id}:embedContent")
@app.post("/v1beta/tunedModels/{tuned_model_id}:embedContent")
async def gemini_tuned_embed_content(tuned_model_id: str, request: Request):
    try:
        _gemini_tuned_base_model(tuned_model_id)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return _gemini_embedding_from_request(body)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini tuned model embedContent failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/tunedModels/{tuned_model_id}:batchEmbedContents")
@app.post("/v1beta/tunedModels/{tuned_model_id}:batchEmbedContents")
async def gemini_tuned_batch_embed_contents(tuned_model_id: str, request: Request):
    try:
        _gemini_tuned_base_model(tuned_model_id)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        return _gemini_batch_embedding_from_request(body)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini tuned model batchEmbedContents failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/tunedModels/{tuned_model_id}:asyncBatchEmbedContent")
@app.post("/v1beta/tunedModels/{tuned_model_id}:asyncBatchEmbedContent")
async def gemini_tuned_async_batch_embed_content(tuned_model_id: str, request: Request):
    try:
        model = _gemini_tuned_base_model(tuned_model_id)
        body = _gemini_batch_body(_gemini_normalize_request(await request.json()))
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        operation, _batch = _gemini_create_completed_embed_batch(model, body)
        return operation
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini tuned model asyncBatchEmbedContent failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/tunedModels/{tuned_model_id}:transferOwnership")
@app.post("/v1beta/tunedModels/{tuned_model_id}:transferOwnership")
async def gemini_transfer_tuned_model_ownership(tuned_model_id: str, request: Request):
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        email = body.get("emailAddress")
        if not isinstance(email, str) or not email.strip():
            raise HTTPException(status_code=400, detail="transferOwnership requires emailAddress.")
        email = email.strip()
        index = _gemini_load_tuned_index()
        name = _gemini_tuned_name(tuned_model_id)
        meta = index.get(name)
        if not meta:
            raise HTTPException(status_code=404, detail=f"Tuned model '{tuned_model_id}' not found.")
        permissions = meta.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            permissions = {}
            meta["permissions"] = permissions
        now = _gemini_now_iso()
        for perm in permissions.values():
            if isinstance(perm, dict) and perm.get("role") == "OWNER":
                perm["role"] = "WRITER"
        owner_id = "owner_" + hashlib.sha256(email.lower().encode("utf-8")).hexdigest()[:16]
        owner_name = f"{name}/permissions/{owner_id}"
        permissions[owner_name] = {
            "name": owner_name,
            "role": "OWNER",
            "granteeType": "USER",
            "emailAddress": email,
        }
        meta["owner"] = email
        meta["updateTime"] = now
        index[name] = meta
        _gemini_save_tuned_index(index)
        return {"name": name, "owner": email, "permission": permissions[owner_name]}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini tuned model transferOwnership failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.get("/v1/tunedModels/{tuned_model_id}/permissions")
@app.get("/v1beta/tunedModels/{tuned_model_id}/permissions")
async def gemini_list_tuned_model_permissions(tuned_model_id: str, request: Request):
    meta = _gemini_get_tuned_meta(tuned_model_id)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_permission_list_response(meta.get("permissions") or {}, request)


@app.post("/v1/tunedModels/{tuned_model_id}/permissions")
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


@app.get("/v1/tunedModels/{tuned_model_id}/permissions/{permission_id:path}")
@app.get("/v1beta/tunedModels/{tuned_model_id}/permissions/{permission_id:path}")
async def gemini_get_tuned_model_permission(tuned_model_id: str, permission_id: str):
    meta = _gemini_get_tuned_meta(tuned_model_id)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    perm = (meta.get("permissions") or {}).get(_gemini_permission_name(tuned_model_id, permission_id))
    if not perm:
        return _gemini_error_response(f"Permission '{permission_id}' not found.", status_code=404, status="NOT_FOUND")
    return perm


@app.patch("/v1/tunedModels/{tuned_model_id}/permissions/{permission_id:path}")
@app.patch("/v1beta/tunedModels/{tuned_model_id}/permissions/{permission_id:path}")
async def gemini_patch_tuned_model_permission(tuned_model_id: str, permission_id: str, request: Request, updateMask: str | None = None):
    index = _gemini_load_tuned_index()
    name = _gemini_tuned_name(tuned_model_id)
    meta = index.get(name)
    if not meta:
        return _gemini_error_response(f"Tuned model '{tuned_model_id}' not found.", status_code=404, status="NOT_FOUND")
    perm_name = _gemini_permission_name(tuned_model_id, permission_id)
    perm = (meta.get("permissions") or {}).get(perm_name)
    if not perm:
        return _gemini_error_response(f"Permission '{permission_id}' not found.", status_code=404, status="NOT_FOUND")
    raw_body = await request.json()
    body = _gemini_permission_body(raw_body)
    updateMask = updateMask or request.query_params.get("update_mask")
    if isinstance(raw_body, dict):
        updateMask = updateMask or raw_body.get("updateMask")
    if isinstance(body, dict):
        updateMask = updateMask or body.pop("updateMask", None)
        fields = _gemini_permission_update_fields(updateMask, body)
        for key in fields:
            if key in body:
                perm[key] = body[key]
    meta["updateTime"] = _gemini_now_iso()
    meta.setdefault("permissions", {})[perm_name] = perm
    index[name] = meta
    _gemini_save_tuned_index(index)
    return perm


@app.post("/v1/tunedModels/{tuned_model_id}/permissions/{permission_id:path}:transferOwnership")
@app.post("/v1beta/tunedModels/{tuned_model_id}/permissions/{permission_id:path}:transferOwnership")
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
    meta["updateTime"] = _gemini_now_iso()
    meta.setdefault("permissions", {})[perm_name] = perm
    index[name] = meta
    _gemini_save_tuned_index(index)
    return JSONResponse({})


@app.delete("/v1/tunedModels/{tuned_model_id}/permissions/{permission_id:path}")
@app.delete("/v1beta/tunedModels/{tuned_model_id}/permissions/{permission_id:path}")
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


@app.post("/v1/models/{model_name:path}:countTokens")
@app.post("/v1beta/models/{model_name:path}:countTokens")
async def gemini_count_tokens(model_name: str, request: Request):
    """Gemini-compatible approximate countTokens endpoint."""
    try:
        _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if isinstance(body, dict):
            cached_tokens = _gemini_cached_content_tokens(body)
            body = _gemini_apply_cached_content(body)
            body = _gemini_apply_file_search(body)
        else:
            cached_tokens = 0
        messages = _gemini_count_tokens_request(body if isinstance(body, dict) else {})
        return _gemini_count_tokens_response(messages, cached_tokens=cached_tokens)
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code)
    except Exception as exc:
        log.exception("Gemini countTokens failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/models/{model_name:path}:computeTokens")
@app.post("/v1beta/models/{model_name:path}:computeTokens")
async def gemini_compute_tokens(model_name: str, request: Request):
    """Gemini-compatible approximate computeTokens endpoint."""
    try:
        _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if isinstance(body, dict):
            body = _gemini_apply_cached_content(body)
            body = _gemini_apply_file_search(body)
        messages = _gemini_count_tokens_request(body if isinstance(body, dict) else {})
        return _gemini_compute_tokens_response(messages)
    except HTTPException as exc:
        return _gemini_error_response(exc.detail, status_code=exc.status_code)
    except Exception as exc:
        log.exception("Gemini computeTokens failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/models/{model_name:path}:embedContent")
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


@app.post("/v1/models/{model_name:path}:batchEmbedContents")
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


@app.post("/v1/models/{model_name:path}:embedText")
@app.post("/v1beta/models/{model_name:path}:embedText")
async def gemini_embed_text(model_name: str, request: Request):
    """Legacy Gemini embedText endpoint mapped to deterministic local vectors."""
    try:
        _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        text = str(body.get("text") or _gemini_legacy_prompt_text(body))
        output_dim = int(body.get("outputDimensionality") or 768)
        seed_parts = [text]
        if body.get("taskType"):
            seed_parts.append(f"taskType:{body['taskType']}")
        if body.get("title"):
            seed_parts.append(f"title:{body['title']}")
        return {"embedding": {"value": _gemini_embedding_values("\n".join(seed_parts), dimensions=output_dim)}}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini embedText failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/models/{model_name:path}:batchEmbedText")
@app.post("/v1beta/models/{model_name:path}:batchEmbedText")
async def gemini_batch_embed_text(model_name: str, request: Request):
    """Legacy Gemini batchEmbedText endpoint mapped to deterministic local vectors."""
    try:
        _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        texts = body.get("texts") or body.get("requests") or []
        if not isinstance(texts, list):
            raise HTTPException(status_code=400, detail="batchEmbedText requires texts or requests array.")
        embeddings = []
        for item in texts:
            if isinstance(item, dict):
                text = str(item.get("text") or _gemini_legacy_prompt_text(item))
                output_dim = int(item.get("outputDimensionality") or 768)
                seed_parts = [text]
                if item.get("taskType"):
                    seed_parts.append(f"taskType:{item['taskType']}")
                if item.get("title"):
                    seed_parts.append(f"title:{item['title']}")
                text = "\n".join(seed_parts)
            else:
                text = str(item)
                output_dim = int(body.get("outputDimensionality") or 768)
            embeddings.append({"value": _gemini_embedding_values(text, dimensions=output_dim)})
        return {"embeddings": embeddings}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini batchEmbedText failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/models/{model_name:path}:asyncBatchEmbedContent")
@app.post("/v1beta/models/{model_name:path}:asyncBatchEmbedContent")
async def gemini_async_batch_embed_content(model_name: str, request: Request):
    """Gemini-compatible asyncBatchEmbedContent as an immediately completed operation."""
    try:
        model = _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        operation, _batch = _gemini_create_completed_embed_batch(model, body)
        return operation
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini asyncBatchEmbedContent failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


def _gemini_create_completed_embed_batch(model: dict[str, Any], body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    body = _gemini_batch_body(_gemini_normalize_request(body))
    embedding_response = _gemini_batch_embedding_from_request(body)
    now = _gemini_now_iso()
    batch_name = "batches/batch_" + uuid.uuid4().hex
    operation_name = "operations/asyncBatchEmbedContent-" + uuid.uuid4().hex
    model_resource = _gemini_model_name(model)
    request_count = len(body.get("requests") or [])
    stats = _gemini_batch_stats(request_count)
    response_payload = {
        "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.AsyncBatchEmbedContentResponse",
        **embedding_response,
    }
    batch = {
        "name": batch_name,
        "displayName": body.get("displayName") or batch_name.rsplit("/", 1)[-1],
        "model": model_resource,
        "state": "BATCH_STATE_SUCCEEDED",
        "createTime": now,
        "updateTime": now,
        "endTime": now,
        "requestCount": request_count,
        "stats": stats,
        "batchStats": stats,
        "operation": operation_name,
        "response": response_payload,
        **_gemini_batch_optional_fields(body),
        "metadata": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.AsyncBatchEmbedContentMetadata",
            "model": model_resource,
            "requestCount": request_count,
            "stats": stats,
            "batchStats": stats,
        },
    }
    operation = {
        "name": operation_name,
        "metadata": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.AsyncBatchEmbedContentMetadata",
            "model": model_resource,
            "requestCount": request_count,
            "stats": stats,
            "batchStats": stats,
            "batch": batch_name,
            "batchResource": batch,
            "displayName": batch["displayName"],
            "state": batch["state"],
            "createTime": now,
            "updateTime": now,
            "endTime": now,
            "operation": operation_name,
        },
        "done": True,
        "response": response_payload,
    }
    stored_batch = _gemini_store_batch(batch)
    stored_operation = _gemini_store_operation(operation)
    return stored_operation, stored_batch


async def _gemini_predict_payload(model_name: str, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _resolve_gemini_model(model_name)
    body = _gemini_normalize_request(body)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    body = _gemini_apply_generate_config(body)
    body = _gemini_apply_response_format(body)
    body = _gemini_normalize_generate_body(body)
    if _model_capabilities(model)["image_generation"]:
        image = await _gemini_generate_image_payload(model_name, body)
        prediction = {
            "predictions": [{
                "bytesBase64Encoded": image["base64"],
                "mimeType": image["mimeType"],
                "generatedFile": image["generatedFile"]["name"],
            }],
            "deployedModelId": _gemini_model_name(model),
        }
        return prediction, body
    request_body = _gemini_predict_to_generate_body(body)
    request_body = _gemini_apply_response_format(request_body)
    request_body = _gemini_apply_cached_content(request_body)
    request_body = _gemini_apply_file_search(request_body)
    request_body = _gemini_inline_local_files(request_body)
    data = await asyncio.to_thread(
        _get_client().generate_raw,
        request=request_body,
        model=str(model["antigravity_model"]),
    )
    response = _gemini_finalize_generate_response(
        _gemini_unwrap_response(data),
        model_name=model_name,
        request_body=request_body,
    )
    return {"predictions": [response], "deployedModelId": _gemini_model_name(model)}, request_body


@app.post("/v1/models/{model_name:path}:predict")
@app.post("/v1beta/models/{model_name:path}:predict")
async def gemini_predict(model_name: str, request: Request):
    """Gemini/Vertex-compatible predict endpoint mapped to generateContent."""
    try:
        prediction, _request_body = await _gemini_predict_payload(model_name, await request.json())
        return prediction
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini predict failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1/models/{model_name:path}:generateImages")
@app.post("/v1beta/models/{model_name:path}:generateImages")
async def gemini_generate_images(model_name: str, request: Request):
    """Gemini/Imagen-compatible generateImages endpoint backed by Antigravity image generation."""
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        generated_images = []
        for _ in range(_gemini_image_count(body)):
            image = await _gemini_generate_image_payload(model_name, body)
            generated_images.append({
                "image": {
                    "imageBytes": image["base64"],
                    "mimeType": image["mimeType"],
                },
                "generatedFile": image["generatedFile"],
            })
        return {
            "generatedImages": generated_images
        }
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini generateImages failed")
        return _gemini_error_response(f"Antigravity image generation error: {exc}", status_code=502, status="UNAVAILABLE")


def _gemini_video_unimplemented_operation(model_name: str, body: dict[str, Any]) -> dict[str, Any]:
    model_id = _gemini_resource_model_id(model_name)
    now = _gemini_now_iso()
    return _gemini_store_operation({
        "name": "operations/generateVideos-" + uuid.uuid4().hex,
        "metadata": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.GenerateVideosOperationMetadata",
            "model": "models/" + model_id,
            "createTime": now,
            "endTime": now,
            "request": {k: v for k, v in body.items() if k in {"prompt", "instances", "parameters", "config"}},
        },
        "done": True,
        "error": {
            "code": 12,
            "message": "Video generation is recognized by the Gemini compatibility layer, but the current Antigravity backend does not expose native video generation.",
            "status": "UNIMPLEMENTED",
        },
    })


@app.post("/v1/models/{model_name:path}:generateVideos")
@app.post("/v1beta/models/{model_name:path}:generateVideos")
async def gemini_generate_videos(model_name: str, request: Request):
    """Gemini/Veo-compatible video generation operation placeholder."""
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        if not _is_gemini_video_model_id(model_name):
            _resolve_gemini_model(model_name)
        return _gemini_video_unimplemented_operation(model_name, body)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini generateVideos failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.post("/v1/models/{model_name:path}:predictLongRunning")
@app.post("/v1beta/models/{model_name:path}:predictLongRunning")
async def gemini_predict_long_running(model_name: str, request: Request):
    """Gemini/Vertex-compatible predictLongRunning as a completed operation."""
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        if _is_gemini_video_model_id(model_name):
            return _gemini_video_unimplemented_operation(model_name, body)
        prediction, request_body = await _gemini_predict_payload(model_name, body)
        model = _resolve_gemini_model(model_name)
        now = _gemini_now_iso()
        operation = {
            "name": "operations/predictLongRunning-" + uuid.uuid4().hex,
            "metadata": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.PredictLongRunningMetadata",
                "model": _gemini_model_name(model),
                "deployedModelId": _gemini_model_name(model),
                "createTime": now,
                "endTime": now,
                "request": {
                    key: request_body[key]
                    for key in ("contents", "generationConfig", "systemInstruction", "safetySettings", "tools", "toolConfig")
                    if key in request_body
                },
            },
            "done": True,
            "response": {
                "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.PredictLongRunningResponse",
                **prediction,
            },
        }
        return _gemini_store_operation(operation)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini predictLongRunning failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1/models/{model_name:path}:countTextTokens")
@app.post("/v1beta/models/{model_name:path}:countTextTokens")
async def gemini_count_text_tokens(model_name: str, request: Request):
    """Legacy Gemini countTextTokens endpoint mapped to local token estimation."""
    try:
        _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        text = _gemini_legacy_prompt_text(body)
        return {"tokenCount": _estimate_tokens(text)}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.post("/v1/models/{model_name:path}:countMessageTokens")
@app.post("/v1beta/models/{model_name:path}:countMessageTokens")
async def gemini_count_message_tokens(model_name: str, request: Request):
    """Legacy Gemini countMessageTokens endpoint mapped to local token estimation."""
    return await gemini_count_text_tokens(model_name, request)


@app.post("/v1/models/{model_name:path}:generateText")
@app.post("/v1beta/models/{model_name:path}:generateText")
async def gemini_generate_text(model_name: str, request: Request):
    """Legacy Gemini generateText endpoint mapped to generateContent."""
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        response = await _gemini_legacy_generate(model_name, body)
        candidates = []
        for candidate in response.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            text = _gemini_content_text(candidate.get("content") or {})
            candidates.append({
                "output": text,
                "safetyRatings": candidate.get("safetyRatings") or candidate.get("safety_ratings") or [],
            })
        return {"candidates": candidates, "filters": [], "usageMetadata": response.get("usageMetadata") or {}}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini generateText failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1/models/{model_name:path}:generateMessage")
@app.post("/v1beta/models/{model_name:path}:generateMessage")
async def gemini_generate_message(model_name: str, request: Request):
    """Legacy Gemini generateMessage endpoint mapped to generateContent."""
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        response = await _gemini_legacy_generate(model_name, body)
        messages = []
        for candidate in response.get("candidates") or []:
            if isinstance(candidate, dict):
                messages.append({"author": "1", "content": _gemini_content_text(candidate.get("content") or {})})
        return {"candidates": messages, "messages": messages, "usageMetadata": response.get("usageMetadata") or {}}
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini generateMessage failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1/models/{model_name:path}:generateAnswer")
@app.post("/v1beta/models/{model_name:path}:generateAnswer")
async def gemini_generate_answer(model_name: str, request: Request):
    """Legacy Semantic Retriever generateAnswer endpoint mapped to generateContent."""
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        response = await _gemini_legacy_generate(model_name, body)
        text = _gemini_response_text(response)
        return {
            "answer": {"content": text},
            "answerableProbability": 1.0 if text else 0.0,
            "inputFeedback": {},
            "usageMetadata": response.get("usageMetadata") or {},
        }
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini generateAnswer failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.post("/v1/models/{model_name:path}:generateContent")
@app.post("/v1beta/models/{model_name:path}:generateContent")
@app.post("/v1/dynamic/{model_name:path}:generateContent")
@app.post("/v1beta/dynamic/{model_name:path}:generateContent")
async def gemini_generate_content(model_name: str, request: Request):
    """Gemini REST-compatible generateContent endpoint backed by Antigravity."""
    try:
        model = _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        body = _gemini_apply_generate_config(body)
        body = _gemini_apply_response_format(body)
        body = _gemini_normalize_generate_body(body)
        if _model_capabilities(model)["image_generation"]:
            image = await _gemini_generate_image_payload(model_name, body)
            return JSONResponse(_gemini_finalize_generate_response({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "inlineData": {
                                "mimeType": image["mimeType"],
                                "data": image["base64"],
                            }
                        }],
                    },
                    "finishReason": "STOP",
                }],
                "generatedFile": image["generatedFile"]["name"],
            }, model_name=model_name, request_body=body))
        body = _gemini_apply_cached_content(body)
        body = _gemini_apply_file_search(body)
        _gemini_reject_unsupported_builtin_tools(body)
        body = _gemini_inline_local_files(body)
        body.pop("model", None)
        if request.query_params.get("alt") == "sse" or request.query_params.get("stream", "").lower() == "true":
            return _gemini_streaming_response(body=body, antigravity_model=str(model["antigravity_model"]))
        data = await asyncio.to_thread(
            _get_client().generate_raw,
            request=body,
            model=str(model["antigravity_model"]),
        )
        return JSONResponse(_gemini_finalize_generate_response(
            _gemini_unwrap_response(data),
            model_name=model_name,
            request_body=body,
        ))
    except HTTPException as exc:
        status = _gemini_status_for_http(exc.status_code)
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


def _gemini_streaming_response(*, body: dict[str, Any], antigravity_model: str) -> StreamingResponse:
    async def _gen():
        got_any = False
        try:
            async for chunk in _get_client().generate_raw_stream_async(
                request=body,
                model=antigravity_model,
            ):
                payload = _gemini_unwrap_response(chunk)
                payload = _gemini_finalize_generate_response(payload, model_name=antigravity_model, request_body=body)
                got_any = True
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            log.warning("Gemini streamGenerateContent failed; falling back to non-streaming: %s", exc)
        if not got_any:
            try:
                data = await asyncio.to_thread(
                    _get_client().generate_raw,
                    request=body,
                    model=antigravity_model,
                )
                payload = _gemini_finalize_generate_response(
                    _gemini_unwrap_response(data),
                    model_name=antigravity_model,
                    request_body=body,
                )
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception as inner:
                payload = _gemini_error_payload(
                    f"Antigravity upstream error: {inner}",
                    status_code=502,
                    status="UNAVAILABLE",
                )
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/v1/models/{model_name:path}:batchGenerateContent")
@app.post("/v1beta/models/{model_name:path}:batchGenerateContent")
async def gemini_batch_generate_content(model_name: str, request: Request):
    """Gemini-compatible batchGenerateContent as an immediately completed operation."""
    try:
        body = _gemini_batch_body(_gemini_normalize_request(await request.json()))
        if isinstance(body, dict) and body.pop("_batchKind", None) == "embed":
            raise HTTPException(status_code=400, detail="batchGenerateContent does not accept embedContentBatch.")
        operation, _batch = await _gemini_create_completed_batch(model_name, body)
        return operation
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini batchGenerateContent failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


async def _gemini_create_completed_batch(model_name: str, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _resolve_gemini_model(model_name)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    requests = body.get("requests")
    if not isinstance(requests, list):
        raise HTTPException(status_code=400, detail="batchGenerateContent requires a requests array.")

    responses: list[dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="batchGenerateContent request items must be objects.")
        req_body = _gemini_normalize_request(
            _gemini_batch_request_item(item, "request", "generateContentRequest")
        )
        req_body = _gemini_apply_generate_config(req_body)
        req_body = _gemini_apply_response_format(req_body)
        req_body = _gemini_normalize_generate_body(req_body)
        req_body = _gemini_apply_cached_content(req_body)
        req_body = _gemini_apply_file_search(req_body)
        req_body = _gemini_inline_local_files(req_body)
        req_body.pop("model", None)
        data = await asyncio.to_thread(
            _get_client().generate_raw,
            request=req_body,
            model=str(model["antigravity_model"]),
        )
        responses.append(_gemini_finalize_generate_response(
            _gemini_unwrap_response(data),
            model_name=model_name,
            request_body=req_body,
        ))

    now = _gemini_now_iso()
    model_resource = _gemini_model_name(model)
    batch_name = "batches/batch_" + uuid.uuid4().hex
    operation_name = "operations/batchGenerateContent-" + uuid.uuid4().hex
    stats = _gemini_batch_stats(len(requests))
    response_payload = {
        "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.BatchGenerateContentResponse",
        "responses": responses,
    }
    batch = {
        "name": batch_name,
        "displayName": body.get("displayName") or body.get("display_name") or batch_name.rsplit("/", 1)[-1],
        "model": model_resource,
        "state": "BATCH_STATE_SUCCEEDED",
        "createTime": now,
        "updateTime": now,
        "endTime": now,
        "requestCount": len(requests),
        "stats": stats,
        "batchStats": stats,
        "operation": operation_name,
        "response": response_payload,
        **_gemini_batch_optional_fields(body),
        "metadata": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.BatchGenerateContentMetadata",
            "model": model_resource,
            "requestCount": len(requests),
            "stats": stats,
            "batchStats": stats,
        },
    }
    operation = {
        "name": operation_name,
        "metadata": {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.BatchGenerateContentMetadata",
            "model": model_resource,
            "requestCount": len(requests),
            "stats": stats,
            "batchStats": stats,
            "batch": batch_name,
            "batchResource": batch,
            "displayName": batch["displayName"],
            "state": batch["state"],
            "createTime": now,
            "updateTime": now,
            "endTime": now,
            "operation": operation_name,
        },
        "done": True,
        "response": response_payload,
    }
    stored_operation = _gemini_store_operation(operation)
    stored_batch = _gemini_store_batch(batch)
    await _gemini_emit_webhook_event("batch.succeeded", stored_batch)
    return stored_operation, stored_batch


@app.post("/v1/batches")
@app.post("/v1beta/batches")
async def gemini_create_batch(request: Request):
    """Gemini-compatible batches.create using immediate local execution."""
    try:
        body = _gemini_batch_body(_gemini_normalize_request(await request.json()))
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        model_name = body.get("model") or body.get("modelName") or body.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise HTTPException(status_code=400, detail="batches.create requires a model.")
        batch_kind = body.pop("_batchKind", None)
        if batch_kind == "embed":
            model = _resolve_gemini_model(model_name)
            _operation, batch = _gemini_create_completed_embed_batch(model, body)
        else:
            _operation, batch = await _gemini_create_completed_batch(model_name, body)
        return _gemini_batch_operation(batch)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini batches create failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.get("/v1/batches")
@app.get("/v1beta/batches")
async def gemini_list_batches(request: Request, filter: str | None = None):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    returnPartialSuccess = _gemini_query_bool(request, "returnPartialSuccess", "return_partial_success")
    index = _gemini_load_batches_index()
    batch_items = []
    for batch in index.values():
        operation = _gemini_batch_operation(batch)
        if _gemini_batch_filter_matches(batch, operation, filter):
            batch_items.append((batch, operation))
    batch_items.sort(key=lambda item: item[0].get("name") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    page = batch_items[start:end]
    response = {
        "operations": [operation for _batch, operation in page],
        "batches": [batch for batch, _operation in page],
        "nextPageToken": str(end) if end < len(batch_items) else "",
    }
    if returnPartialSuccess:
        response["unreachable"] = []
    return response


@app.get("/v1/batches/{batch_id:path}")
@app.get("/v1beta/batches/{batch_id:path}")
async def gemini_get_batch(batch_id: str):
    batch = _gemini_get_batch(batch_id)
    if not batch:
        return _gemini_error_response(f"Batch '{batch_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_batch_operation(batch)


@app.post("/v1/batches/{batch_id:path}:cancel")
@app.post("/v1beta/batches/{batch_id:path}:cancel")
async def gemini_cancel_batch(batch_id: str):
    batch = _gemini_get_batch(batch_id)
    if not batch:
        return _gemini_error_response(f"Batch '{batch_id}' not found.", status_code=404, status="NOT_FOUND")
    terminal_states = {
        "BATCH_STATE_SUCCEEDED",
        "BATCH_STATE_FAILED",
        "BATCH_STATE_CANCELLED",
        "BATCH_STATE_EXPIRED",
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
    }
    if batch.get("state") not in terminal_states:
        now = _gemini_now_iso()
        batch["state"] = "BATCH_STATE_CANCELLED"
        batch["updateTime"] = now
        batch["endTime"] = now
        _gemini_store_batch(batch)
        operation = _gemini_get_operation(str(batch.get("operation") or ""))
        if operation and not operation.get("done"):
            operation["done"] = True
            operation["error"] = {"code": 1, "message": "Operation cancelled.", "status": "CANCELLED"}
            _gemini_store_operation(operation)
    return JSONResponse({})


def _gemini_patch_batch(batch_id: str, body: dict[str, Any], update_mask: str | None = None) -> dict[str, Any]:
    name = _gemini_batch_name(batch_id)
    index = _gemini_load_batches_index()
    batch = index.get(name)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")
    body = _gemini_batch_body(body)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    update_mask = update_mask or body.pop("updateMask", None)
    fields = _gemini_batch_update_fields(update_mask, body)
    if not fields:
        raise HTTPException(status_code=400, detail="batch update requires displayName or priority.")
    for key in fields:
        if key not in body:
            raise HTTPException(status_code=400, detail=f"{key} is required by updateMask.")
    for key in ("displayName", "priority"):
        if key in body:
            batch[key] = body[key]
    batch["updateTime"] = _gemini_now_iso()
    index[name] = batch
    _gemini_save_batches_index(index)
    return batch


@app.patch("/v1/batches/{batch_id:path}:updateGenerateContentBatch")
@app.patch("/v1beta/batches/{batch_id:path}:updateGenerateContentBatch")
@app.post("/v1/batches/{batch_id:path}:updateGenerateContentBatch")
@app.post("/v1beta/batches/{batch_id:path}:updateGenerateContentBatch")
async def gemini_update_generate_content_batch(batch_id: str, request: Request, updateMask: str | None = None):
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        updateMask = updateMask or request.query_params.get("update_mask")
        return _gemini_batch_operation(_gemini_patch_batch(batch_id, body, updateMask))
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.patch("/v1/batches/{batch_id:path}:updateEmbedContentBatch")
@app.patch("/v1beta/batches/{batch_id:path}:updateEmbedContentBatch")
@app.post("/v1/batches/{batch_id:path}:updateEmbedContentBatch")
@app.post("/v1beta/batches/{batch_id:path}:updateEmbedContentBatch")
async def gemini_update_embed_content_batch(batch_id: str, request: Request, updateMask: str | None = None):
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        updateMask = updateMask or request.query_params.get("update_mask")
        return _gemini_batch_operation(_gemini_patch_batch(batch_id, body, updateMask))
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)


@app.delete("/v1/batches/{batch_id:path}")
@app.delete("/v1beta/batches/{batch_id:path}")
async def gemini_delete_batch(batch_id: str):
    name = _gemini_batch_name(batch_id)
    index = _gemini_load_batches_index()
    if name not in index:
        return _gemini_error_response(f"Batch '{batch_id}' not found.", status_code=404, status="NOT_FOUND")
    operation_name = index[name].get("operation")
    index.pop(name, None)
    _gemini_save_batches_index(index)
    if operation_name:
        operations = _gemini_load_operations_index()
        operations.pop(str(operation_name), None)
        _gemini_save_operations_index(operations)
    return JSONResponse({})


@app.post("/v1/webhooks")
@app.post("/v1beta/webhooks")
async def gemini_create_webhook(request: Request):
    """Gemini-compatible webhooks.create stored locally.

    The proxy preserves webhook configuration for clients that manage Gemini
    polling alternatives. It does not independently deliver callback events.
    """
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        webhook = _gemini_webhook_body(body)
        name = webhook.get("name")
        if isinstance(name, str) and name.strip():
            resource_name = _gemini_webhook_name(name)
        else:
            resource_name = "webhooks/webhook_" + uuid.uuid4().hex
        now = _gemini_now_iso()
        resource = dict(webhook)
        create_secret = resource.pop("newSigningSecret", True)
        resource["name"] = resource_name
        resource.setdefault("displayName", resource_name.rsplit("/", 1)[-1])
        resource.setdefault("createTime", now)
        resource["updateTime"] = now
        resource["uri"] = _gemini_webhook_uri(resource)
        resource["targetUri"] = resource["uri"]
        if not resource["uri"]:
            raise HTTPException(status_code=400, detail="Webhook requires uri or targetUri.")
        resource["subscribedEvents"] = _gemini_webhook_events(resource)
        resource["eventTypes"] = list(resource["subscribedEvents"])
        resource.setdefault("state", "enabled")
        if create_secret is not False:
            secret = _gemini_new_signing_secret()
            resource["newSigningSecret"] = secret
            resource["signingSecrets"] = [secret]
        return _gemini_webhook_public_resource(_gemini_store_webhook(resource), include_new_secret=True)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini webhooks create failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.get("/v1/webhooks")
@app.get("/v1beta/webhooks")
async def gemini_list_webhooks(request: Request):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    webhooks = list(_gemini_load_webhooks_index().values())
    webhooks.sort(key=lambda item: item.get("name") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    return {
        "webhooks": [_gemini_webhook_public_resource(item) for item in webhooks[start:end]],
        "nextPageToken": str(end) if end < len(webhooks) else "",
    }


@app.get("/v1/webhooks/{webhook_id:path}")
@app.get("/v1beta/webhooks/{webhook_id:path}")
async def gemini_get_webhook(webhook_id: str):
    webhook = _gemini_get_webhook(webhook_id)
    if not webhook:
        return _gemini_error_response(f"Webhook '{webhook_id}' not found.", status_code=404, status="NOT_FOUND")
    return _gemini_webhook_public_resource(webhook)


@app.patch("/v1/webhooks/{webhook_id:path}")
@app.patch("/v1beta/webhooks/{webhook_id:path}")
async def gemini_patch_webhook(webhook_id: str, request: Request, updateMask: str | None = None):
    try:
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        patch = _gemini_webhook_body(body)
        name = _gemini_webhook_name(webhook_id)
        index = _gemini_load_webhooks_index()
        webhook = index.get(name)
        if not webhook:
            raise HTTPException(status_code=404, detail=f"Webhook '{webhook_id}' not found.")
        updateMask = updateMask or patch.pop("updateMask", None) or body.pop("updateMask", None)
        field_aliases = {
            "display_name": "displayName",
            "uri": "uri",
            "target_uri": "targetUri",
            "event_types": "eventTypes",
            "subscribed_events": "subscribedEvents",
            "state": "state",
            "webhook.display_name": "displayName",
            "webhook.uri": "uri",
            "webhook.target_uri": "targetUri",
            "webhook.event_types": "eventTypes",
            "webhook.subscribed_events": "subscribedEvents",
            "webhook.state": "state",
        }
        fields = []
        for raw_field in (updateMask or "").split(","):
            raw_field = raw_field.strip()
            if raw_field:
                fields.append(field_aliases.get(raw_field, field_aliases.get(raw_field.rsplit(".", 1)[-1], raw_field)))
        updated_fields: set[str] = set()
        if fields:
            for field in fields:
                if field in patch:
                    webhook[field] = patch[field]
                    updated_fields.add(field)
        else:
            for key, value in patch.items():
                if key not in {"name", "createTime"}:
                    webhook[key] = value
                    updated_fields.add(key)
        if "targetUri" in updated_fields and "uri" not in updated_fields:
            webhook["uri"] = webhook["targetUri"]
        elif "uri" in updated_fields and "targetUri" not in updated_fields:
            webhook["targetUri"] = webhook["uri"]
        if "uri" in webhook or "targetUri" in webhook:
            webhook["uri"] = _gemini_webhook_uri(webhook)
            webhook["targetUri"] = webhook["uri"]
        if "subscribedEvents" in webhook or "eventTypes" in webhook or "events" in webhook:
            webhook["subscribedEvents"] = _gemini_webhook_events(webhook)
            webhook["eventTypes"] = list(webhook["subscribedEvents"])
        webhook["updateTime"] = _gemini_now_iso()
        index[name] = webhook
        _gemini_save_webhooks_index(index)
        return _gemini_webhook_public_resource(webhook)
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini webhooks patch failed")
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")


@app.delete("/v1/webhooks/{webhook_id:path}")
@app.delete("/v1beta/webhooks/{webhook_id:path}")
async def gemini_delete_webhook(webhook_id: str):
    name = _gemini_webhook_name(webhook_id)
    index = _gemini_load_webhooks_index()
    if name not in index:
        return _gemini_error_response(f"Webhook '{webhook_id}' not found.", status_code=404, status="NOT_FOUND")
    index.pop(name, None)
    _gemini_save_webhooks_index(index)
    return JSONResponse({})


@app.post("/v1/webhooks/{webhook_id:path}:ping")
@app.post("/v1beta/webhooks/{webhook_id:path}:ping")
async def gemini_ping_webhook(webhook_id: str):
    name = _gemini_webhook_name(webhook_id)
    index = _gemini_load_webhooks_index()
    webhook = index.get(name)
    if not webhook:
        return _gemini_error_response(f"Webhook '{webhook_id}' not found.", status_code=404, status="NOT_FOUND")
    attempt = await _gemini_deliver_webhook(webhook, "webhooks.ping", {"name": name})
    webhook.setdefault("deliveryAttempts", []).append(attempt)
    webhook["updateTime"] = _gemini_now_iso()
    index[name] = webhook
    _gemini_save_webhooks_index(index)
    return {"deliveryAttempt": attempt}


@app.post("/v1/webhooks/{webhook_id:path}:rotateSigningSecret")
@app.post("/v1beta/webhooks/{webhook_id:path}:rotateSigningSecret")
async def gemini_rotate_webhook_signing_secret(webhook_id: str):
    name = _gemini_webhook_name(webhook_id)
    index = _gemini_load_webhooks_index()
    webhook = index.get(name)
    if not webhook:
        return _gemini_error_response(f"Webhook '{webhook_id}' not found.", status_code=404, status="NOT_FOUND")
    secret = _gemini_new_signing_secret()
    secrets_list = webhook.get("signingSecrets") if isinstance(webhook.get("signingSecrets"), list) else []
    webhook["signingSecrets"] = [secret] + [item for item in secrets_list if isinstance(item, dict)][:1]
    webhook["newSigningSecret"] = secret
    webhook["updateTime"] = _gemini_now_iso()
    index[name] = webhook
    _gemini_save_webhooks_index(index)
    return _gemini_webhook_public_resource(webhook, include_new_secret=True)


def _gemini_interaction_model_output_step(text: str, response: dict[str, Any] | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text, "annotations": []})
    return {
        "type": "model_output",
        "status": "completed",
        "content": content,
        "outputText": text,
        "response": response or {},
    }


async def _gemini_create_interaction(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    body = _gemini_apply_generate_config(body)
    body = _gemini_apply_response_format(body)
    body = _gemini_normalize_generate_body(body)
    model_name = body.get("model") or body.get("modelName") or body.get("model_name") or "models/gemini-3-flash-agent"
    model = _resolve_gemini_model(str(model_name))
    contents = _gemini_interaction_contents(body)

    history: list[dict[str, Any]] = []
    previous_id = body.get("previousInteractionId") or body.get("previous_interaction_id")
    if previous_id:
        previous = _gemini_get_interaction(str(previous_id))
        if previous is None:
            raise HTTPException(status_code=404, detail=f"Interaction '{previous_id}' not found.")
        history = list(previous.get("history") or [])

    request_body: dict[str, Any] = {"contents": history + contents}
    for key in (
        "systemInstruction",
        "generationConfig",
        "safetySettings",
        "tools",
        "toolConfig",
        "cachedContent",
        "agent",
        "agentConfig",
        "environment",
        "webhookConfig",
    ):
        if key in body:
            request_body[key] = body[key]

    now = _gemini_now_iso()
    interaction_name = "interactions/int_" + uuid.uuid4().hex
    if body.get("background") is True:
        usage = _gemini_interaction_usage({})
        interaction = {
            "name": interaction_name,
            "id": interaction_name.rsplit("/", 1)[-1],
            "model": _gemini_model_name(model),
            "agent": body.get("agent"),
            "status": "in_progress",
            "created": now,
            "updated": now,
            "createTime": now,
            "updateTime": now,
            "previousInteractionId": previous_id or None,
            "input": body.get("input", body.get("messages", body.get("contents"))),
            "request": request_body,
            "output": {},
            "outputText": "",
            "history": request_body["contents"],
            "steps": [],
            "usage": usage,
            "usageMetadata": usage,
            "background": True,
        }
        if body.get("store", True) is not False:
            _gemini_store_interaction(interaction)
        return _gemini_interaction_resource(interaction)

    if _model_capabilities(model)["image_generation"]:
        image = await _gemini_generate_image_payload(str(model_name), request_body)
        response = _gemini_finalize_generate_response({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "inlineData": {
                            "mimeType": image["mimeType"],
                            "data": image["base64"],
                        }
                    }],
                },
                "finishReason": "STOP",
            }],
            "generatedFile": image["generatedFile"]["name"],
        }, model_name=str(model_name), request_body=request_body)
        model_content = response["candidates"][0]["content"]
        usage = _gemini_interaction_usage({})
        interaction = {
            "name": interaction_name,
            "id": interaction_name.rsplit("/", 1)[-1],
            "model": _gemini_model_name(model),
            "agent": body.get("agent"),
            "status": "completed",
            "created": now,
            "updated": now,
            "createTime": now,
            "updateTime": now,
            "previousInteractionId": previous_id or None,
            "input": body.get("input", body.get("messages", body.get("contents"))),
            "output": response,
            "outputText": "",
            "generatedFile": image["generatedFile"],
            "history": request_body["contents"] + [model_content],
            "steps": [
                {
                    "type": "image_generation",
                    "status": "completed",
                    "generatedFile": image["generatedFile"]["name"],
                },
                _gemini_interaction_model_output_step("", response),
            ],
            "usage": usage,
            "usageMetadata": usage,
        }
        if body.get("store", True) is not False:
            _gemini_store_interaction(interaction)
            await _gemini_emit_webhook_event("interaction.completed", interaction)
        return _gemini_interaction_resource(interaction)

    request_body = _gemini_apply_cached_content(request_body)
    request_body = _gemini_apply_file_search(request_body)
    request_body = _gemini_inline_local_files(request_body)
    data = await asyncio.to_thread(
        _get_client().generate_raw,
        request=request_body,
        model=str(model["antigravity_model"]),
    )
    response = _gemini_finalize_generate_response(
        _gemini_unwrap_response(data),
        model_name=str(model_name),
        request_body=request_body,
    )
    text = _gemini_response_text(response)
    model_content = None
    candidates = response.get("candidates") or []
    if candidates and isinstance(candidates[0], dict) and isinstance(candidates[0].get("content"), dict):
        model_content = candidates[0]["content"]
    new_history = request_body["contents"] + ([model_content] if model_content else [])
    usage = _gemini_interaction_usage(response.get("usageMetadata") or response.get("usage_metadata") or {})
    interaction = {
        "name": interaction_name,
        "id": interaction_name.rsplit("/", 1)[-1],
        "model": _gemini_model_name(model),
        "agent": body.get("agent"),
        "status": "completed",
        "created": now,
        "updated": now,
        "createTime": now,
        "updateTime": now,
        "previousInteractionId": previous_id or None,
        "input": body.get("input", body.get("messages", body.get("contents"))),
        "output": response,
        "outputText": text,
        "history": new_history,
        "steps": [
            _gemini_interaction_model_output_step(text, response),
            {
                "type": "model_response",
                "status": "completed",
                "outputText": text,
            }
        ],
        "usage": usage,
        "usageMetadata": response.get("usageMetadata") or response.get("usage_metadata") or usage,
    }
    if body.get("store", True) is not False:
        _gemini_store_interaction(interaction)
        await _gemini_emit_webhook_event("interaction.completed", interaction)
    return _gemini_interaction_resource(interaction)


@app.post("/v1/interactions")
@app.post("/v1beta/interactions")
async def gemini_create_interaction(request: Request):
    try:
        body = _gemini_interaction_body(await request.json())
        interaction = await _gemini_create_interaction(body)
        if isinstance(body, dict) and body.get("stream"):
            async def _gen():
                created = dict(interaction)
                created.pop("outputText", None)
                yield f"data: {json.dumps({'type': 'interaction.created', 'interaction': created}, ensure_ascii=False)}\n\n"
                if interaction.get("outputText"):
                    yield f"data: {json.dumps({'type': 'interaction.output_text.delta', 'delta': interaction['outputText']}, ensure_ascii=False)}\n\n"
                for step in interaction.get("steps") or []:
                    if isinstance(step, dict):
                        yield f"data: {json.dumps({'type': 'interaction.step.completed', 'step': step}, ensure_ascii=False)}\n\n"
                generated_file = interaction.get("generatedFile")
                if generated_file:
                    yield f"data: {json.dumps({'type': 'interaction.output_image.done', 'generatedFile': generated_file}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'interaction.completed', 'interaction': interaction}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")
        return interaction
    except HTTPException as exc:
        status = "NOT_FOUND" if exc.status_code == 404 else "INVALID_ARGUMENT"
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        log.exception("Gemini interactions create failed")
        return _gemini_error_response(f"Antigravity upstream error: {exc}", status_code=502, status="UNAVAILABLE")


@app.get("/v1/interactions")
@app.get("/v1beta/interactions")
async def gemini_list_interactions(request: Request):
    query = request.query_params
    page_size_raw = query.get("pageSize") or query.get("page_size") or "100"
    try:
        page_size = max(1, min(1000, int(page_size_raw)))
    except (TypeError, ValueError):
        page_size = 100
    page_token = query.get("pageToken") or query.get("page_token") or "0"
    try:
        start = max(0, int(page_token))
    except (TypeError, ValueError):
        start = 0
    interactions = [
        _gemini_interaction_resource(interaction)
        for interaction in _gemini_load_interactions_index().values()
        if isinstance(interaction, dict)
    ]
    interactions.sort(key=lambda item: item.get("createTime") or item.get("created") or item.get("name") or "")
    end = start + page_size
    return {
        "interactions": interactions[start:end],
        "nextPageToken": str(end) if end < len(interactions) else "",
    }


@app.get("/v1/interactions/{interaction_id:path}")
@app.get("/v1beta/interactions/{interaction_id:path}")
async def gemini_get_interaction(interaction_id: str):
    interaction = _gemini_get_interaction(interaction_id)
    if not interaction:
        return _gemini_error_response(f"Interaction '{interaction_id}' not found.", status_code=404, status="NOT_FOUND")
    return interaction


def _gemini_cancel_interaction_response(interaction_id: str):
    interaction = _gemini_get_interaction(interaction_id)
    if not interaction:
        return _gemini_error_response(f"Interaction '{interaction_id}' not found.", status_code=404, status="NOT_FOUND")
    if interaction.get("status") not in {"completed", "failed", "cancelled"}:
        interaction["status"] = "cancelled"
        now = _gemini_now_iso()
        interaction["updated"] = now
        interaction["updateTime"] = now
        _gemini_store_interaction(interaction)
    return interaction


@app.post("/v1/interactions/{interaction_id:path}:cancel")
@app.post("/v1beta/interactions/{interaction_id:path}:cancel")
async def gemini_cancel_interaction(interaction_id: str):
    return _gemini_cancel_interaction_response(interaction_id)


@app.post("/v1/interactions/{interaction_id:path}/cancel")
@app.post("/v1beta/interactions/{interaction_id:path}/cancel")
async def gemini_cancel_interaction_rest(interaction_id: str):
    return _gemini_cancel_interaction_response(interaction_id)


@app.delete("/v1/interactions/{interaction_id:path}")
@app.delete("/v1beta/interactions/{interaction_id:path}")
async def gemini_delete_interaction(interaction_id: str):
    name = _gemini_interaction_name(interaction_id)
    index = _gemini_load_interactions_index()
    if name not in index:
        return _gemini_error_response(f"Interaction '{interaction_id}' not found.", status_code=404, status="NOT_FOUND")
    index.pop(name, None)
    _gemini_save_interactions_index(index)
    return JSONResponse({})


@app.websocket("/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent")
@app.websocket("/v1beta/live")
@app.websocket("/v1/live")
async def gemini_live_websocket(websocket: WebSocket):
    """Gemini Live API-compatible WebSocket for text turn flows.

    The Antigravity backend is request/response oriented, so this implements
    the Live protocol envelope for setup and text clientContent turns. Realtime
    audio/video chunks are rejected explicitly instead of being silently ignored.
    """
    if not _websocket_api_key_valid(websocket):
        await websocket.close(code=1008, reason="Invalid API key")
        return
    await websocket.accept()
    setup: dict[str, Any] = {}
    model_name = websocket.query_params.get("model") or "models/gemini-3-flash-agent"
    history: list[dict[str, Any]] = []
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = _gemini_normalize_request(json.loads(raw))
            except json.JSONDecodeError:
                await websocket.send_json(_gemini_error_payload(
                    "Live API messages must be JSON.",
                    status_code=400,
                    status="INVALID_ARGUMENT",
                ))
                continue
            if not isinstance(message, dict):
                await websocket.send_json(_gemini_error_payload(
                    "Live API message must be an object.",
                    status_code=400,
                    status="INVALID_ARGUMENT",
                ))
                continue

            setup_msg = message.get("setup")
            if isinstance(setup_msg, dict):
                setup = _gemini_normalize_request(setup_msg)
                model_name = setup.get("model") or setup.get("modelName") or model_name
                await websocket.send_json({"setupComplete": {}})
                continue

            if isinstance(message.get("realtimeInput"), dict) or isinstance(message.get("realtime_input"), dict):
                await websocket.send_json(_gemini_error_payload(
                    "Realtime audio/video input is not supported by this Antigravity-backed Live shim.",
                    status_code=501,
                    status="UNIMPLEMENTED",
                ))
                continue

            if isinstance(message.get("toolResponse"), dict) or isinstance(message.get("tool_response"), dict):
                await websocket.send_json({"toolCallCancellation": {"ids": []}})
                continue

            turns, turn_complete = _gemini_live_turns_from_message(message)
            if not turns:
                await websocket.send_json(_gemini_error_payload(
                    "Unsupported Live API message.",
                    status_code=400,
                    status="INVALID_ARGUMENT",
                ))
                continue
            history.extend(turns)
            if not turn_complete:
                await websocket.send_json({"serverContent": {"turnComplete": False}})
                continue

            try:
                response, model_turn = await _gemini_live_generate(model_name=str(model_name), history=history, setup=setup)
            except HTTPException as exc:
                status = _gemini_status_for_http(exc.status_code)
                await websocket.send_json(_gemini_error_payload(exc.detail, status_code=exc.status_code, status=status))
                continue
            except Exception as exc:
                log.exception("Gemini Live generate failed")
                await websocket.send_json(_gemini_error_payload(
                    f"Antigravity upstream error: {exc}",
                    status_code=502,
                    status="UNAVAILABLE",
                ))
                continue

            server_content: dict[str, Any] = {"turnComplete": True}
            if model_turn:
                history.append(model_turn)
                server_content["modelTurn"] = model_turn
            if isinstance(response.get("usageMetadata"), dict):
                server_content["usageMetadata"] = response["usageMetadata"]
            await websocket.send_json({"serverContent": server_content})
    except WebSocketDisconnect:
        return


@app.post("/v1/models/{model_name:path}:streamGenerateContent")
@app.post("/v1beta/models/{model_name:path}:streamGenerateContent")
@app.post("/v1/dynamic/{model_name:path}:streamGenerateContent")
@app.post("/v1beta/dynamic/{model_name:path}:streamGenerateContent")
async def gemini_stream_generate_content(model_name: str, request: Request):
    """Gemini REST-compatible SSE streamGenerateContent endpoint."""
    try:
        model = _resolve_gemini_model(model_name)
        body = _gemini_normalize_request(await request.json())
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        body = _gemini_apply_generate_config(body)
        body = _gemini_apply_response_format(body)
        body = _gemini_normalize_generate_body(body)
        body = _gemini_apply_cached_content(body)
        body = _gemini_apply_file_search(body)
        _gemini_reject_unsupported_builtin_tools(body)
        body = _gemini_inline_local_files(body)
        body.pop("model", None)
    except HTTPException as exc:
        status = _gemini_status_for_http(exc.status_code)
        return _gemini_error_response(exc.detail, status_code=exc.status_code, status=status)
    except Exception as exc:
        return _gemini_error_response(str(exc), status_code=400, status="INVALID_ARGUMENT")

    return _gemini_streaming_response(body=body, antigravity_model=str(model["antigravity_model"]))


@app.get("/v1/operations")
@app.get("/v1beta/operations")
async def gemini_list_operations(request: Request, filter: str | None = None):
    pageSize, pageToken = _gemini_list_query_params(request, default_page_size=100, max_page_size=1000)
    returnPartialSuccess = _gemini_query_bool(request, "returnPartialSuccess", "return_partial_success")
    index = _gemini_load_operations_index()
    operations = [operation for operation in index.values() if _gemini_operation_filter_matches(operation, filter)]
    operations.sort(key=lambda item: item.get("name") or "")
    start = int(pageToken or 0) if pageToken and pageToken.isdigit() else 0
    end = start + pageSize
    response = {"operations": operations[start:end], "nextPageToken": str(end) if end < len(operations) else ""}
    if returnPartialSuccess:
        response["unreachable"] = []
    return response


@app.get("/v1/operations/{operation_id:path}")
@app.get("/v1beta/operations/{operation_id:path}")
async def gemini_get_operation(operation_id: str):
    operation = _gemini_get_operation(operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.post("/v1/operations/{operation_id:path}:cancel")
@app.post("/v1beta/operations/{operation_id:path}:cancel")
async def gemini_cancel_operation(operation_id: str):
    operation = _gemini_get_operation(operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    _gemini_cancel_operation(operation)
    return JSONResponse({})


@app.post("/v1/operations/{operation_id:path}:wait")
@app.post("/v1beta/operations/{operation_id:path}:wait")
async def gemini_wait_operation(operation_id: str):
    operation = _gemini_get_operation(operation_id)
    if not operation:
        return _gemini_error_response(f"Operation '{operation_id}' not found.", status_code=404, status="NOT_FOUND")
    return operation


@app.delete("/v1/operations/{operation_id:path}")
@app.delete("/v1beta/operations/{operation_id:path}")
async def gemini_delete_operation(operation_id: str):
    index = _gemini_load_operations_index()
    name = _gemini_resolve_operation_key(index, operation_id)
    if not name:
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
            generated_file = _gemini_store_generated_file(
                image_bytes,
                mime_type=mimetypes.guess_type(str(output_path))[0] or "image/png",
                display_name=output_path.name,
            )
            operation = {
                "name": "operations/generateImage-" + uuid.uuid4().hex,
                "metadata": {
                    "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.GenerateImageMetadata",
                    "generatedFile": generated_file["name"],
                },
                "done": True,
                "response": {
                    "@type": "type.googleapis.com/google.ai.generativelanguage.v1beta.GeneratedFile",
                    **generated_file,
                },
            }
            _gemini_store_operation(operation)
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
                "generated_file": generated_file["name"],
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
