import asyncio
import json

from fastapi import HTTPException
from fastapi.testclient import TestClient

import antigravity_proxy as proxy


def test_model_normalization_adds_capabilities_and_hides_internal(monkeypatch):
    monkeypatch.delenv("ANTIGRAVITY_PROXY_INCLUDE_INTERNAL_MODELS", raising=False)
    models = [
        {"id": "Gemini Lite", "object": "model", "antigravity_model": "gemini-lite-a"},
        {"id": "Gemini Lite", "object": "model", "antigravity_model": "gemini-lite-b"},
        {"id": "chat_123", "object": "model", "antigravity_model": "chat_123"},
        {"id": "Gemini Image", "object": "model", "antigravity_model": "gemini-image"},
    ]

    normalized = proxy._normalize_models(models)

    ids = [m["id"] for m in normalized]
    assert "chat_123" not in ids
    assert "Gemini Lite [gemini-lite-a]" in ids
    assert "Gemini Lite [gemini-lite-b]" in ids
    image = next(m for m in normalized if m["antigravity_model"] == "gemini-image")
    assert image["capabilities"]["image_generation"] is True
    assert image["capabilities"]["tools"] is False


def test_usage_parser_accepts_multiple_upstream_shapes():
    usage = proxy._usage_from_response(
        {"response": {"usage": {"input_tokens": 7, "output_tokens": 11}}},
        messages=[proxy.ChatMessage(role="user", content="hello")],
        completion="world",
    )

    assert usage == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}


def test_tool_choice_validation_rejects_unknown_function():
    try:
        proxy._tool_choice_to_gemini(
            {"type": "function", "function": {"name": "missing"}},
            [{"type": "function", "function": {"name": "known"}}],
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "missing" in str(exc.detail)
    else:
        raise AssertionError("Expected HTTPException")


def test_streaming_tool_call_uses_delta_chunks():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": "{\"location\":\"Seoul\"}"},
            }
        ],
    }
    response = proxy._fc_sse("chatcmpl_test", 1, "model", msg, "tool_calls")

    async def collect():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return chunks

    body = "".join(asyncio.run(collect()))
    assert '"role": "assistant"' in body
    assert '"name": "weather"' in body
    assert '"arguments": "{\\"location\\":\\"Seoul\\"}"' in body
    assert '"finish_reason": "tool_calls"' in body
    assert "data: [DONE]" in body


def test_openai_error_response_shape():
    response = proxy._openai_error_response(
        "Nope",
        status_code=404,
        code="not_found",
    )
    payload = json.loads(response.body)

    assert response.status_code == 404
    assert payload["error"]["message"] == "Nope"
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "not_found"


def test_chat_raw_path_maps_structured_output_thinking_and_grounding(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            seen["model"] = model
            return {
                "response": {
                    "candidates": [{
                        "content": {"parts": [{"text": "{\"answer\":\"ok\"}"}]}
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 8,
                    },
                }
            }

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1/chat/completions", json={
        "model": "Gemini 3.5 Flash (High)",
        "messages": [{"role": "user", "content": "answer as json"}],
        "tools": [{"type": "web_search_preview"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
        "reasoning": {"effort": "low"},
    })

    assert response.status_code == 200
    gen = seen["request"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseSchema"]["properties"]["answer"]["type"] == "string"
    assert gen["thinkingConfig"] == {"thinkingLevel": "low"}
    assert seen["request"]["tools"] == [{"google_search": {}}]


def test_chat_rejects_unsupported_hosted_tool():
    client = TestClient(proxy.app)

    response = client.post("/v1/chat/completions", json={
        "model": "Gemini 3.5 Flash (High)",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "file_search"}],
    })

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Unsupported tool type: file_search"


def test_gemini_models_and_count_tokens():
    client = TestClient(proxy.app)

    models = client.get("/v1beta/models")
    assert models.status_code == 200
    first = models.json()["models"][0]
    assert first["name"].startswith("models/")
    assert "generateContent" in first["supportedGenerationMethods"]

    one = client.get("/v1beta/models/gemini-3-flash-agent")
    assert one.status_code == 200
    assert one.json()["name"] == "models/gemini-3-flash-agent"

    counted = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "hello world"}]}]
    })
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0


def test_gemini_generate_content_passes_through_and_normalizes(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            seen["model"] = model
            return {
                "response": {
                    "candidates": [{
                        "content": {"role": "model", "parts": [{"text": "hello"}]},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
                }
            }

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generation_config": {"response_mime_type": "application/json", "max_output_tokens": 32},
        "tools": [{"googleSearch": {}}],
    })

    assert response.status_code == 200
    assert response.json()["candidates"][0]["content"]["parts"][0]["text"] == "hello"
    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["request"]["generationConfig"]["maxOutputTokens"] == 32
    assert seen["request"]["tools"] == [{"google_search": {}}]


def test_gemini_stream_generate_content_sse(monkeypatch):
    class FakeClient:
        async def generate_raw_stream_async(self, *, request, model=""):
            yield {"response": {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    with client.stream("POST", "/v1beta/models/gemini-3-flash-agent:streamGenerateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert 'data: {"candidates":' in body
    assert "data: [DONE]" in body


def test_admin_refresh_requires_configured_api_key(monkeypatch):
    monkeypatch.delenv("ANTIGRAVITY_PROXY_API_KEY", raising=False)
    client = TestClient(proxy.app)

    response = client.post("/admin/models/refresh")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "admin_api_key_not_configured"


def test_admin_refresh_requires_valid_api_key(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_PROXY_API_KEY", "secret")
    client = TestClient(proxy.app)

    response = client.post("/admin/models/refresh", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_admin_refresh_calls_fetch_with_valid_key(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_PROXY_API_KEY", "secret")

    async def fake_fetch():
        return {"ok": True, "updated_count": 2, "model_count": 3, "map_entries": 9}

    monkeypatch.setattr(proxy, "_fetch_and_update_models", fake_fetch)
    client = TestClient(proxy.app)

    response = client.post("/admin/models/refresh", headers={"X-API-Key": "secret"})

    assert response.status_code == 200
    assert response.json()["updated_count"] == 2
