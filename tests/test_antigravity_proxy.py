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
    assert "embedContent" in first["supportedGenerationMethods"]

    one = client.get("/v1beta/models/gemini-3-flash-agent")
    assert one.status_code == 200
    assert one.json()["name"] == "models/gemini-3-flash-agent"

    counted = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "hello world"}]}]
    })
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0


def test_gemini_embeddings_are_deterministic():
    client = TestClient(proxy.app)

    payload = {
        "content": {"parts": [{"text": "embed me"}]},
        "outputDimensionality": 32,
    }
    first = client.post("/v1beta/models/gemini-3-flash-agent:embedContent", json=payload)
    second = client.post("/v1beta/models/gemini-3-flash-agent:embedContent", json=payload)
    batch = client.post("/v1beta/models/gemini-3-flash-agent:batchEmbedContents", json={
        "requests": [
            {"content": {"parts": [{"text": "one"}]}, "outputDimensionality": 16},
            {"content": {"parts": [{"text": "two"}]}, "outputDimensionality": 16},
        ]
    })

    assert first.status_code == 200
    values = first.json()["embedding"]["values"]
    assert len(values) == 32
    assert values == second.json()["embedding"]["values"]
    assert batch.status_code == 200
    assert len(batch.json()["embeddings"]) == 2
    assert len(batch.json()["embeddings"][0]["values"]) == 16


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


def test_gemini_batch_generate_content_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {"response": {"candidates": [{"content": {"parts": [{"text": request["contents"][0]["parts"][0]["text"]}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1beta/models/gemini-3-flash-agent:batchGenerateContent", json={
        "requests": [
            {"contents": [{"role": "user", "parts": [{"text": "first"}]}]},
            {"contents": [{"role": "user", "parts": [{"text": "second"}]}]},
        ]
    })

    assert created.status_code == 200
    operation = created.json()
    assert operation["done"] is True
    assert len(operation["response"]["responses"]) == 2

    fetched = client.get(f"/v1beta/{operation['name']}")
    listed = client.get("/v1beta/operations")
    deleted = client.delete(f"/v1beta/{operation['name']}")

    assert fetched.status_code == 200
    assert fetched.json()["name"] == operation["name"]
    assert listed.status_code == 200
    assert listed.json()["operations"][0]["name"] == operation["name"]
    assert deleted.status_code == 200


def test_gemini_files_upload_and_file_data_inline_conversion(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "gemini_files"))
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "file ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    uploaded = client.post(
        "/upload/v1beta/files?uploadType=media&displayName=note.txt",
        content=b"hello file",
        headers={"Content-Type": "text/plain"},
    )
    assert uploaded.status_code == 200
    file_resource = uploaded.json()["file"]
    assert file_resource["name"].startswith("files/file_")
    assert file_resource["mimeType"] == "text/plain"

    listed = client.get("/v1beta/files")
    assert listed.status_code == 200
    assert listed.json()["files"][0]["name"] == file_resource["name"]

    fetched = client.get(f"/v1beta/{file_resource['name']}")
    assert fetched.status_code == 200
    assert fetched.json()["displayName"] == "note.txt"

    generated = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "read this"},
                {"fileData": {"mimeType": "text/plain", "fileUri": file_resource["uri"]}},
            ],
        }]
    })

    assert generated.status_code == 200
    parts = seen["request"]["contents"][0]["parts"]
    assert parts[1]["inlineData"]["mimeType"] == "text/plain"
    assert parts[1]["inlineData"]["data"]

    deleted = client.delete(f"/v1beta/{file_resource['name']}")
    assert deleted.status_code == 200
    missing = client.get(f"/v1beta/{file_resource['name']}")
    assert missing.status_code == 404


def test_gemini_resumable_file_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "gemini_files"))
    client = TestClient(proxy.app)

    started = client.post(
        "/upload/v1beta/files",
        json={"file": {"displayName": "resumable.txt", "mimeType": "text/plain"}},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Type": "text/plain",
        },
    )
    assert started.status_code == 200
    upload_url = started.headers["x-goog-upload-url"]
    session_path = "/" + upload_url.split("/", 3)[3]

    finished = client.post(
        session_path,
        content=b"resumable body",
        headers={"X-Goog-Upload-Command": "upload, finalize"},
    )

    assert finished.status_code == 200
    file_resource = finished.json()["file"]
    assert file_resource["displayName"] == "resumable.txt"
    assert file_resource["mimeType"] == "text/plain"


def test_gemini_cached_contents_merge_into_generate_request(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_CACHED_CONTENTS_DIR", str(tmp_path / "gemini_cached"))
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "cached ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1beta/cachedContents", json={
        "model": "models/gemini-3-flash-agent",
        "contents": [{"role": "user", "parts": [{"text": "cached context"}]}],
        "systemInstruction": {"parts": [{"text": "cached system"}]},
    })
    assert created.status_code == 200
    cache_name = created.json()["name"]

    listed = client.get("/v1beta/cachedContents")
    assert listed.status_code == 200
    assert listed.json()["cachedContents"][0]["name"] == cache_name

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "cachedContent": cache_name,
        "contents": [{"role": "user", "parts": [{"text": "new prompt"}]}],
    })

    assert response.status_code == 200
    assert seen["request"]["systemInstruction"]["parts"][0]["text"] == "cached system"
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "cached context"
    assert seen["request"]["contents"][1]["parts"][0]["text"] == "new prompt"
    assert "cachedContent" not in seen["request"]

    deleted = client.delete(f"/v1beta/{cache_name}")
    assert deleted.status_code == 200
    missing = client.get(f"/v1beta/{cache_name}")
    assert missing.status_code == 404


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
