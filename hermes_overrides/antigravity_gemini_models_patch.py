"""Hermes overlay: list Antigravity-backed Gemini models live.

This module is loaded from a venv .pth file on the Ubuntu Hermes gateway. It
patches only Hermes' Gemini provider catalog function, leaving the Hermes git
tree untouched so normal `hermes update` operations do not overwrite it.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _dotenv_value(name: str) -> str:
    try:
        path = Path.home() / ".hermes" / ".env"
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _env_value(name: str) -> str:
    return os.getenv(name, "").strip() or _dotenv_value(name).strip()


def _gemini_base_url() -> str:
    cfg = _load_config()
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    base = ""
    if str(model_cfg.get("provider", "")).strip().lower() == "gemini":
        base = str(model_cfg.get("base_url", "") or "").strip()
    base = base or _env_value("GEMINI_BASE_URL")
    base = base or "https://generativelanguage.googleapis.com/v1beta"
    return base.rstrip("/")


def _gemini_api_key() -> str:
    cfg = _load_config()
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return (
        str(model_cfg.get("api_key", "") or "").strip()
        or _env_value("GOOGLE_API_KEY")
        or _env_value("GEMINI_API_KEY")
        or _env_value("ANTIGRAVITY_PROXY_API_KEY")
    )


def _extract_model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    raw_models = payload.get("models")
    if raw_models is None and isinstance(payload.get("data"), list):
        raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_models:
        name = ""
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("id") or "").strip()
        elif isinstance(item, str):
            name = item.strip()
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _fetch_live_gemini_models() -> list[str]:
    base = _gemini_base_url()
    url = base + "/models"
    key = _gemini_api_key()
    headers = {
        "Accept": "application/json",
        "User-Agent": "hermes-antigravity-model-overlay/1",
    }
    if key:
        headers["x-goog-api-key"] = key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return _extract_model_ids(payload)


def install() -> None:
    try:
        from hermes_cli import models as hermes_models
    except Exception:
        return

    original = getattr(hermes_models, "provider_model_ids", None)
    if not callable(original) or getattr(original, "_antigravity_gemini_patched", False):
        return

    def provider_model_ids(provider=None, *, force_refresh: bool = False):
        try:
            normalized = hermes_models.normalize_provider(provider)
        except Exception:
            normalized = (provider or "").strip().lower()
        if normalized == "gemini":
            try:
                live = _fetch_live_gemini_models()
                if live:
                    return live
            except Exception:
                pass
        return original(provider, force_refresh=force_refresh)

    provider_model_ids._antigravity_gemini_patched = True  # type: ignore[attr-defined]
    provider_model_ids._antigravity_original = original  # type: ignore[attr-defined]
    hermes_models.provider_model_ids = provider_model_ids


install()
