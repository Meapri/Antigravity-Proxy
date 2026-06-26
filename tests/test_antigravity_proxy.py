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
    assert "generateAnswer" in first["supportedGenerationMethods"]
    assert "asyncBatchEmbedContent" in first["supportedGenerationMethods"]
    assert first["inputTokenLimit"] > 0
    assert "capabilities" in first

    one = client.get("/v1beta/models/gemini-3-flash-agent")
    assert one.status_code == 200
    assert one.json()["name"] == "models/gemini-3-flash-agent"

    image = client.get("/v1beta/models/gemini-3.1-flash-image")
    assert image.status_code == 200
    assert "predict" in image.json()["supportedGenerationMethods"]
    assert "generateContent" in image.json()["supportedGenerationMethods"]
    assert "generateImages" in image.json()["supportedGenerationMethods"]
    assert image.json()["capabilities"]["imageGeneration"] is True

    counted = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "hello world"}]}]
    })
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0
    assert counted.json()["promptTokensDetails"][0]["modality"] == "TEXT"
    assert counted.json()["cacheTokensDetails"] == []

    counted_media = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "describe"},
                {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
            ],
        }]
    })
    assert counted_media.status_code == 200
    assert {item["modality"] for item in counted_media.json()["promptTokensDetails"]} >= {"TEXT", "IMAGE"}


def test_gemini_count_tokens_applies_generate_content_request_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_CACHED_CONTENTS_DIR", str(tmp_path / "gemini_cached"))
    client = TestClient(proxy.app)

    created = client.post("/v1beta/cachedContents", json={
        "model": "models/gemini-3-flash-agent",
        "contents": [{"role": "user", "parts": [{"text": "cached context with several extra words"}]}],
    })
    assert created.status_code == 200
    cache_name = created.json()["name"]

    uncached = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "new prompt"}]}],
    })
    counted = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "generateContentRequest": {
            "cachedContent": cache_name,
            "contents": [{"role": "user", "parts": [{"text": "new prompt"}]}],
        }
    })

    assert uncached.status_code == 200
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > uncached.json()["totalTokens"]


def test_gemini_video_model_and_generate_videos_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    client = TestClient(proxy.app)

    model = client.get("/v1beta/models/veo-3.1-generate-preview")
    generated = client.post("/v1beta/models/veo-3.1-generate-preview:generateVideos", json={
        "prompt": "make a short clip",
    })
    predicted = client.post("/v1beta/models/veo-3.1-generate-preview:predictLongRunning", json={
        "instances": [{"prompt": "make another short clip"}],
    })

    assert model.status_code == 200
    assert model.json()["name"] == "models/veo-3.1-generate-preview"
    assert "generateVideos" in model.json()["supportedGenerationMethods"]
    assert model.json()["capabilities"]["videoGeneration"] is True

    assert generated.status_code == 200
    generated_body = generated.json()
    assert generated_body["done"] is True
    assert generated_body["error"]["status"] == "UNIMPLEMENTED"
    assert generated_body["metadata"]["model"] == "models/veo-3.1-generate-preview"

    op_id = generated_body["name"].split("/", 1)[1]
    fetched = client.get(f"/v1beta/operations/{op_id}")
    model_operation = client.get(f"/v1beta/models/veo-3.1-generate-preview/operations/{op_id}")
    model_operations = client.get("/v1beta/models/veo-3.1-generate-preview/operations")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == generated_body["name"]
    assert model_operation.status_code == 200
    assert model_operations.status_code == 200
    assert generated_body["name"] in {item["name"] for item in model_operations.json()["operations"]}

    assert predicted.status_code == 200
    assert predicted.json()["error"]["status"] == "UNIMPLEMENTED"


def test_gemini_model_aliases_resolve_for_public_style_names(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["model"] = model
            return {"response": {"candidates": [{"content": {"parts": [{"text": "alias ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    fetched = client.get("/v1beta/models/gemini-flash-latest")
    generated = client.post("/v1beta/models/gemini-flash-latest:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    })
    image = client.get("/v1beta/models/gemini-image-latest")

    assert fetched.status_code == 200
    assert fetched.json()["name"] == "models/gemini-3-flash-agent"
    assert generated.status_code == 200
    assert seen["model"] == "gemini-3-flash-agent"
    assert image.status_code == 200
    assert image.json()["name"] == "models/gemini-3.1-flash-image"


def test_gemini_auth_accepts_google_api_key_styles(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_PROXY_API_KEY", "secret")
    client = TestClient(proxy.app)

    header_response = client.get("/v1beta/models", headers={"x-goog-api-key": "secret"})
    query_response = client.get("/v1beta/models?key=secret")
    rejected = client.get("/v1beta/models?key=wrong")

    assert header_response.status_code == 200
    assert query_response.status_code == 200
    assert rejected.status_code == 401


def test_gemini_v1_stable_aliases_do_not_break_openai_models(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "files"))
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "stable ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    openai_models = client.get("/v1/models")
    generated = client.post("/v1/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    })
    registered = client.post("/v1/files:register", json={
        "file": {"displayName": "stable.txt", "uri": "gs://bucket/stable.txt"}
    })
    uploaded = client.post(
        "/upload/v1/files?uploadType=media&displayName=stable-upload.txt",
        content=b"stable upload",
        headers={"Content-Type": "text/plain"},
    )

    assert openai_models.status_code == 200
    assert openai_models.json()["object"] == "list"
    assert generated.status_code == 200
    assert generated.json()["candidates"][0]["content"]["parts"][0]["text"] == "stable ok"
    assert registered.status_code == 200
    assert registered.json()["file"]["uri"] == "gs://bucket/stable.txt"
    assert uploaded.status_code == 200
    assert uploaded.json()["file"]["displayName"] == "stable-upload.txt"


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


def test_gemini_embeddings_respect_task_type_title_and_snake_case():
    client = TestClient(proxy.app)

    base = client.post("/v1beta/models/gemini-3-flash-agent:embedContent", json={
        "content": {"parts": [{"text": "same document"}]},
        "output_dimensionality": 16,
    })
    retrieval = client.post("/v1beta/models/gemini-3-flash-agent:embedContent", json={
        "content": {"parts": [{"text": "same document"}]},
        "output_dimensionality": 16,
        "task_type": "RETRIEVAL_DOCUMENT",
        "title": "Atlas Plan",
    })
    batch = client.post("/v1beta/models/gemini-3-flash-agent:batchEmbedContents", json={
        "requests": [{
            "content": {"parts": [{"text": "same document"}]},
            "output_dimensionality": 16,
            "task_type": "RETRIEVAL_QUERY",
        }]
    })

    assert base.status_code == 200
    assert retrieval.status_code == 200
    assert batch.status_code == 200
    assert len(base.json()["embedding"]["values"]) == 16
    assert len(batch.json()["embeddings"][0]["values"]) == 16
    assert base.json()["embedding"]["values"] != retrieval.json()["embedding"]["values"]


def test_gemini_legacy_embed_text_methods():
    client = TestClient(proxy.app)

    single = client.post("/v1beta/models/gemini-3-flash-agent:embedText", json={
        "text": "legacy embed",
        "outputDimensionality": 12,
    })
    batch = client.post("/v1/models/gemini-3-flash-agent:batchEmbedText", json={
        "texts": ["one", "two"],
        "outputDimensionality": 10,
    })

    assert single.status_code == 200
    assert len(single.json()["embedding"]["value"]) == 12
    assert batch.status_code == 200
    assert len(batch.json()["embeddings"]) == 2
    assert len(batch.json()["embeddings"][0]["value"]) == 10


def test_gemini_async_batch_embed_and_batch_update(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", str(tmp_path / "batches"))
    client = TestClient(proxy.app)

    created = client.post("/v1beta/models/gemini-3-flash-agent:asyncBatchEmbedContent", json={
        "displayName": "embed job",
        "requests": [
            {"content": {"parts": [{"text": "alpha"}]}, "outputDimensionality": 8},
            {"content": {"parts": [{"text": "beta"}]}, "outputDimensionality": 8},
        ],
    })

    assert created.status_code == 200
    operation = created.json()
    assert operation["done"] is True
    assert operation["response"]["embeddings"][0]["values"]

    batch_name = operation["metadata"]["batch"]
    updated = client.patch(f"/v1beta/{batch_name}:updateEmbedContentBatch", json={"displayName": "renamed"})
    fetched = client.get(f"/v1beta/{batch_name}")

    assert updated.status_code == 200
    assert updated.json()["name"] == batch_name
    assert updated.json()["metadata"]["batchResource"]["displayName"] == "renamed"
    assert fetched.json()["name"] == batch_name
    assert fetched.json()["metadata"]["operation"] == operation["name"]
    assert fetched.json()["metadata"]["batchResource"]["displayName"] == "renamed"
    assert fetched.json()["metadata"]["batchStats"]["requestCount"] == "2"


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
        "tool_config": {"function_calling_config": {"mode": "any", "allowed_function_names": "lookup"}},
    })

    assert response.status_code == 200
    assert response.json()["candidates"][0]["content"]["parts"][0]["text"] == "hello"
    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["request"]["generationConfig"]["maxOutputTokens"] == 32
    assert seen["request"]["tools"] == [{"google_search": {}}]
    assert seen["request"]["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["lookup"],
    }


def test_gemini_generate_content_rejects_unsupported_builtin_tools():
    client = TestClient(proxy.app)

    url_context = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "read url"}]}],
        "tools": [{"urlContext": {}}],
    })
    code_execution = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "run code"}]}],
        "tools": [{"code_execution": {}}],
    })

    assert url_context.status_code == 501
    assert url_context.json()["error"]["status"] == "UNIMPLEMENTED"
    assert "url_context" in url_context.json()["error"]["message"]
    assert code_execution.status_code == 501
    assert code_execution.json()["error"]["status"] == "UNIMPLEMENTED"
    assert "code_execution" in code_execution.json()["error"]["message"]


def test_gemini_generate_content_alt_sse(monkeypatch):
    class FakeClient:
        async def generate_raw_stream_async(self, *, request, model=""):
            yield {"response": {"candidates": [{"content": {"parts": [{"text": "alt stream"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    with client.stream("POST", "/v1beta/models/gemini-3-flash-agent:generateContent?alt=sse", json={
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "alt stream" in body
    assert "data: [DONE]" in body


def test_gemini_generate_content_alt_sse_falls_back_on_empty_stream(monkeypatch):
    class FakeClient:
        async def generate_raw_stream_async(self, *, request, model=""):
            if False:
                yield {}

        def generate_raw(self, *, request, model=""):
            return {"response": {"candidates": [{"content": {"parts": [{"text": "fallback stream"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    with client.stream("POST", "/v1beta/models/gemini-3-flash-agent:generateContent?alt=sse", json={
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "fallback stream" in body
    assert "data: [DONE]" in body


def test_gemini_predict_and_predict_long_running(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {"response": {"candidates": [{"content": {"parts": [{"text": request["contents"][0]["parts"][0]["text"]}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    predicted = client.post("/v1beta/models/gemini-3-flash-agent:predict", json={
        "instances": [{"text": "predict me"}],
    })
    long_running = client.post("/v1beta/models/gemini-3-flash-agent:predictLongRunning", json={
        "instances": [{"text": "later"}],
    })

    assert predicted.status_code == 200
    assert predicted.json()["predictions"][0]["candidates"][0]["content"]["parts"][0]["text"] == "predict me"
    assert long_running.status_code == 200
    assert long_running.json()["done"] is True
    assert long_running.json()["response"]["predictions"]

    op_id = long_running.json()["name"].split("/", 1)[1]
    model_operation = client.get(f"/v1beta/models/gemini-3-flash-agent/operations/{op_id}")
    model_operations = client.get("/v1beta/models/gemini-3-flash-agent/operations")
    waited_operation = client.post(f"/v1beta/models/gemini-3-flash-agent/operations/{op_id}:wait")
    assert model_operation.status_code == 200
    assert model_operation.json()["name"] == long_running.json()["name"]
    assert model_operations.status_code == 200
    assert model_operations.json()["operations"][0]["name"] == long_running.json()["name"]
    assert waited_operation.status_code == 200
    assert waited_operation.json()["name"] == long_running.json()["name"]


def test_gemini_legacy_text_message_answer_and_token_methods(monkeypatch):
    seen = []

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen.append(request)
            text = request["contents"][0]["parts"][0]["text"]
            return {"response": {"candidates": [{"content": {"parts": [{"text": f"legacy:{text}"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    text = client.post("/v1beta/models/gemini-3-flash-agent:generateText", json={"prompt": {"text": "hello text"}})
    message = client.post("/v1beta/models/gemini-3-flash-agent:generateMessage", json={"prompt": {"messages": [{"content": "hello msg"}]}})
    answer = client.post("/v1/models/gemini-3-flash-agent:generateAnswer", json={"text": "hello answer"})
    counted_text = client.post("/v1beta/models/gemini-3-flash-agent:countTextTokens", json={"prompt": {"text": "count these"}})
    counted_message = client.post("/v1beta/models/gemini-3-flash-agent:countMessageTokens", json={"message": {"content": "count msg"}})

    assert text.status_code == 200
    assert text.json()["candidates"][0]["output"] == "legacy:hello text"
    assert message.status_code == 200
    assert message.json()["candidates"][0]["content"] == "legacy:hello msg"
    assert answer.status_code == 200
    assert answer.json()["answer"]["content"] == "legacy:hello answer"
    assert counted_text.status_code == 200
    assert counted_text.json()["tokenCount"] > 0
    assert counted_message.status_code == 200
    assert counted_message.json()["tokenCount"] > 0


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
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", str(tmp_path / "batches"))

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
    waited = client.post(f"/v1beta/{operation['name']}:wait")
    batch_name = operation["metadata"]["batch"]
    batch = client.get(f"/v1beta/{batch_name}")
    batches = client.get("/v1beta/batches")
    deleted = client.delete(f"/v1beta/{operation['name']}")

    assert fetched.status_code == 200
    assert fetched.json()["name"] == operation["name"]
    assert listed.status_code == 200
    assert listed.json()["operations"][0]["name"] == operation["name"]
    assert waited.status_code == 200
    assert waited.json()["name"] == operation["name"]
    assert batch.status_code == 200
    assert batch.json()["name"] == batch_name
    assert batch.json()["done"] is True
    assert batch.json()["metadata"]["state"] == "BATCH_STATE_SUCCEEDED"
    assert batch.json()["metadata"]["stats"]["requestCount"] == "2"
    assert batch.json()["metadata"]["batchResource"]["name"] == batch_name
    assert operation["metadata"]["stats"]["successfulRequestCount"] == "2"
    assert batches.status_code == 200
    assert batches.json()["operations"][0]["name"] == batch_name
    assert batches.json()["batches"][0]["operation"] == operation["name"]
    assert deleted.status_code == 200


def test_gemini_batch_wrapped_request_bodies(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", str(tmp_path / "batches"))

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {"response": {"candidates": [{"content": {"parts": [{"text": request["contents"][0]["parts"][0]["text"]}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1beta/batches", json={
        "batch": {
            "model": "models/gemini-3-flash-agent",
            "displayName": "wrapped batch",
            "requests": [{"contents": [{"role": "user", "parts": [{"text": "wrapped"}]}]}],
        }
    })
    generated = client.post("/v1beta/models/gemini-3-flash-agent:batchGenerateContent", json={
        "generateContentBatch": {
            "displayName": "wrapped method",
            "requests": [{"contents": [{"role": "user", "parts": [{"text": "method"}]}]}],
        }
    })
    embedded = client.post("/v1beta/batches", json={
        "embedContentBatch": {
            "model": "models/gemini-3-flash-agent",
            "displayName": "wrapped embed",
            "requests": [{
                "content": {"parts": [{"text": "embed wrapped"}]},
                "outputDimensionality": 8,
            }],
        }
    })
    wrong_method = client.post("/v1beta/models/gemini-3-flash-agent:batchGenerateContent", json={
        "embedContentBatch": {
            "requests": [{"content": {"parts": [{"text": "wrong"}]}}],
        }
    })

    assert created.status_code == 200
    assert created.json()["done"] is True
    assert created.json()["metadata"]["batchResource"]["displayName"] == "wrapped batch"
    assert created.json()["metadata"]["stats"]["requestCount"] == "1"
    assert generated.status_code == 200
    assert generated.json()["metadata"]["requestCount"] == 1
    assert generated.json()["metadata"]["stats"]["successfulRequestCount"] == "1"
    assert generated.json()["response"]["responses"][0]["candidates"][0]["content"]["parts"][0]["text"] == "method"
    assert embedded.status_code == 200
    assert embedded.json()["metadata"]["batchResource"]["displayName"] == "wrapped embed"
    assert embedded.json()["metadata"]["state"] == "BATCH_STATE_SUCCEEDED"
    assert embedded.json()["metadata"]["stats"]["successfulRequestCount"] == "1"
    assert embedded.json()["response"]["embeddings"][0]["values"]
    assert len(embedded.json()["response"]["embeddings"][0]["values"]) == 8
    assert wrong_method.status_code == 400


def test_gemini_batches_create_get_cancel_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", str(tmp_path / "batches"))

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {"response": {"candidates": [{"content": {"parts": [{"text": "batch ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1beta/batches", json={
        "model": "models/gemini-3-flash-agent",
        "displayName": "docs batch",
        "requests": [
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        ],
    })

    assert created.status_code == 200
    batch_operation = created.json()
    batch_resource = batch_operation["metadata"]["batchResource"]
    assert batch_operation["name"].startswith("batches/")
    assert batch_operation["done"] is True
    assert batch_resource["displayName"] == "docs batch"
    assert batch_operation["metadata"]["state"] == "BATCH_STATE_SUCCEEDED"
    assert batch_resource["requestCount"] == 1
    assert batch_operation["metadata"]["stats"]["requestCount"] == "1"

    fetched = client.get(f"/v1beta/{batch_operation['name']}")
    operation = client.get(f"/v1beta/{batch_resource['operation']}")
    cancelled = client.post(f"/v1beta/{batch_operation['name']}:cancel")
    deleted = client.delete(f"/v1beta/{batch_operation['name']}")
    missing = client.get(f"/v1beta/{batch_operation['name']}")

    assert fetched.status_code == 200
    assert fetched.json()["name"] == batch_operation["name"]
    assert operation.status_code == 200
    assert operation.json()["metadata"]["batch"] == batch_operation["name"]
    assert cancelled.status_code == 200
    assert deleted.status_code == 200
    assert missing.status_code == 404


def test_gemini_snake_case_query_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", str(tmp_path / "batches"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    first = client.post("/v1beta/batches", json={
        "model": "models/gemini-3-flash-agent",
        "displayName": "first",
        "requests": [{"contents": [{"role": "user", "parts": [{"text": "a"}]}]}],
    }).json()
    client.post("/v1beta/batches", json={
        "model": "models/gemini-3-flash-agent",
        "displayName": "second",
        "requests": [{"contents": [{"role": "user", "parts": [{"text": "b"}]}]}],
    })

    listed = client.get("/v1beta/batches?page_size=1&page_token=0")
    patched = client.patch(f"/v1beta/{first['name']}:updateGenerateContentBatch?update_mask=displayName", json={
        "generateContentBatch": {"displayName": "patched"},
    })
    priority = client.patch(f"/v1beta/{first['name']}:updateGenerateContentBatch?updateMask=priority", json={
        "priority": "7",
    })
    bad_patch = client.patch(f"/v1beta/{first['name']}:updateGenerateContentBatch?updateMask=state", json={
        "state": "BATCH_STATE_CANCELLED",
    })

    assert listed.status_code == 200
    assert len(listed.json()["batches"]) == 1
    assert listed.json()["nextPageToken"] == "1"
    assert patched.status_code == 200
    assert patched.json()["metadata"]["batchResource"]["displayName"] == "patched"
    assert priority.status_code == 200
    assert priority.json()["metadata"]["batchResource"]["priority"] == "7"
    assert bad_patch.status_code == 400


def test_gemini_interactions_create_previous_store_and_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_INTERACTIONS_DIR", str(tmp_path / "interactions"))
    seen = []

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen.append(request)
            text = request["contents"][-1]["parts"][0]["text"]
            return {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": f"echo:{text}"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    first = client.post("/v1beta/interactions", json={
        "model": "models/gemini-3-flash-agent",
        "input": "first",
    })
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["name"].startswith("interactions/")
    assert first_body["outputText"] == "echo:first"

    second = client.post("/v1beta/interactions", json={
        "model": "models/gemini-3-flash-agent",
        "previous_interaction_id": first_body["name"],
        "input": "second",
    })
    assert second.status_code == 200
    assert len(seen[-1]["contents"]) == 3
    assert seen[-1]["contents"][0]["parts"][0]["text"] == "first"
    assert seen[-1]["contents"][1]["parts"][0]["text"] == "echo:first"

    fetched = client.get(f"/v1beta/{first_body['name']}")
    no_store = client.post("/v1beta/interactions", json={"input": "transient", "store": False})
    missing = client.get(f"/v1beta/{no_store.json()['name']}")
    cancelled = client.post(f"/v1beta/{first_body['name']}:cancel")
    deleted = client.delete(f"/v1beta/{first_body['name']}")

    assert fetched.status_code == 200
    assert missing.status_code == 404
    assert cancelled.status_code == 200
    assert deleted.status_code == 200

    with client.stream("POST", "/v1beta/interactions", json={"input": "stream", "stream": True}) as streamed:
        body = streamed.read().decode()

    assert streamed.status_code == 200
    assert "interaction.created" in body
    assert "interaction.output_text.delta" in body
    assert "data: [DONE]" in body


def test_gemini_interactions_cancel_accepts_rest_and_colon_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_INTERACTIONS_DIR", str(tmp_path / "interactions"))
    client = TestClient(proxy.app)

    proxy._gemini_store_interaction({
        "name": "interactions/int_cancel_rest",
        "id": "int_cancel_rest",
        "status": "in_progress",
        "createTime": proxy._gemini_now_iso(),
        "updateTime": proxy._gemini_now_iso(),
    })
    proxy._gemini_store_interaction({
        "name": "interactions/int_cancel_colon",
        "id": "int_cancel_colon",
        "status": "in_progress",
        "createTime": proxy._gemini_now_iso(),
        "updateTime": proxy._gemini_now_iso(),
    })

    rest = client.post("/v1beta/interactions/int_cancel_rest/cancel")
    colon = client.post("/v1beta/interactions/int_cancel_colon:cancel")
    missing = client.post("/v1beta/interactions/missing/cancel")

    assert rest.status_code == 200
    assert colon.status_code == 200
    assert missing.status_code == 404
    assert rest.json()["name"] == "interactions/int_cancel_rest"
    assert rest.json()["status"] == "cancelled"
    assert colon.json()["name"] == "interactions/int_cancel_colon"
    assert colon.json()["status"] == "cancelled"
    assert client.get("/v1beta/interactions/int_cancel_rest").json()["status"] == "cancelled"
    assert client.get("/v1beta/interactions/int_cancel_colon").json()["status"] == "cancelled"


def test_gemini_interactions_accept_content_item_aliases_and_image_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_INTERACTIONS_DIR", str(tmp_path / "interactions"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_GENERATED_FILES_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    monkeypatch.setattr(proxy, "_gemini_remote_file_uri_to_inline", lambda uri, mime_type=None: {
        "inlineData": {"mimeType": mime_type or "image/jpeg", "data": "aW1hZ2U="}
    })
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            seen["model"] = model
            return {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "aliases ok"}]}}]}}

        def generate_image(self, *, prompt, output_dir, aspect_ratio="", image_size=""):
            seen["image_prompt"] = prompt
            output = output_dir / "interaction-image.jpg"
            output.write_bytes(b"interaction-image")
            return output

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    text_interaction = client.post("/v1beta/interactions", json={
        "input": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image", "image_url": {"url": "https://example.test/cat.jpg"}, "mime_type": "image/jpeg"},
            ],
        }],
        "generation_config": {
            "response_modalities": ["TEXT"],
            "media_resolution": "MEDIA_RESOLUTION_LOW",
        },
    })

    assert text_interaction.status_code == 200
    parts = seen["request"]["contents"][0]["parts"]
    assert parts[0] == {"text": "describe"}
    assert parts[1]["inlineData"] == {"mimeType": "image/jpeg", "data": "aW1hZ2U="}
    assert seen["request"]["generationConfig"]["responseModalities"] == ["TEXT"]
    assert seen["request"]["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_LOW"

    image_interaction = client.post("/v1beta/interactions", json={
        "model": "models/gemini-image-latest",
        "input": [{"type": "text", "text": "draw an icon"}],
        "store": False,
    })

    assert image_interaction.status_code == 200
    body = image_interaction.json()
    assert seen["image_prompt"] == "draw an icon"
    assert body["generatedFile"]["name"].startswith("generatedFiles/")
    assert body["output"]["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    assert client.get(f"/v1beta/{body['name']}").status_code == 404


def test_gemini_webhooks_crud_and_v1_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_WEBHOOKS_DIR", str(tmp_path / "webhooks"))
    client = TestClient(proxy.app)

    created = client.post("/v1beta/webhooks", json={
        "webhook": {
            "display_name": "Batch updates",
            "target_uri": "https://example.test/hook",
            "event_types": ["batch.succeeded"],
        }
    })
    assert created.status_code == 200
    webhook = created.json()
    assert webhook["name"].startswith("webhooks/")
    assert webhook["displayName"] == "Batch updates"
    assert webhook["uri"] == "https://example.test/hook"
    assert webhook["targetUri"] == "https://example.test/hook"
    assert webhook["subscribedEvents"] == ["batch.succeeded"]
    assert webhook["eventTypes"] == ["batch.succeeded"]
    assert webhook["state"] == "enabled"
    assert webhook["newSigningSecret"]["secret"]
    assert "secret" not in webhook["signingSecrets"][0]

    fetched = client.get(f"/v1/{webhook['name']}")
    listed = client.get("/v1/webhooks?page_size=1&page_token=0")
    patched = client.patch(f"/v1/{webhook['name']}?update_mask=displayName", json={
        "display_name": "Renamed",
        "target_uri": "https://example.test/ignored",
    })

    assert fetched.status_code == 200
    assert fetched.json()["name"] == webhook["name"]
    assert listed.status_code == 200
    assert listed.json()["webhooks"][0]["name"] == webhook["name"]
    assert "newSigningSecret" not in fetched.json()
    assert patched.status_code == 200
    assert patched.json()["displayName"] == "Renamed"
    assert patched.json()["targetUri"] == "https://example.test/hook"

    malformed = client.patch(
        f"/v1/{webhook['name']}",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert malformed.status_code == 400
    assert malformed.json()["error"]["status"] == "INVALID_ARGUMENT"

    rotated = client.post(f"/v1beta/{webhook['name']}:rotateSigningSecret")
    assert rotated.status_code == 200
    assert rotated.json()["newSigningSecret"]["secret"]
    assert rotated.json()["newSigningSecret"]["secret"] != webhook["newSigningSecret"]["secret"]
    assert "secret" not in rotated.json()["signingSecrets"][0]

    deleted = client.delete(f"/v1beta/{webhook['name']}")
    missing = client.get(f"/v1beta/{webhook['name']}")
    assert deleted.status_code == 200
    assert missing.status_code == 404


def test_gemini_webhooks_ping_and_batch_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_WEBHOOKS_DIR", str(tmp_path / "webhooks"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", str(tmp_path / "batches"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "operations"))
    deliveries = []

    async def fake_post(uri, payload, headers):
        deliveries.append({"uri": uri, "payload": payload, "headers": headers})
        return 204, "ok"

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {"response": {"candidates": [{"content": {"parts": [{"text": "batch ok"}]}}]}}

    monkeypatch.setattr(proxy, "_gemini_post_webhook_json", fake_post)
    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1beta/webhooks", json={
        "uri": "https://example.test/hook",
        "subscribed_events": ["webhooks.ping", "batches.*"],
    })
    assert created.status_code == 200
    name = created.json()["name"]

    pinged = client.post(f"/v1beta/{name}:ping")
    assert pinged.status_code == 200
    assert deliveries[-1]["payload"]["eventType"] == "webhooks.ping"
    assert deliveries[-1]["headers"]["X-Goog-Webhook-Signature"].startswith("sha256=")

    batch = client.post("/v1beta/batches", json={
        "model": "models/gemini-3-flash-agent",
        "requests": [{"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}],
    })
    assert batch.status_code == 200
    assert deliveries[-1]["payload"]["eventType"] == "batch.succeeded"
    assert deliveries[-1]["payload"]["resource"]["name"].startswith("batches/")

    fetched = client.get(f"/v1beta/{name}")
    attempts = fetched.json()["deliveryAttempts"]
    assert [item["eventType"] for item in attempts] == ["webhooks.ping", "batch.succeeded"]
    assert all(item["status"] == "delivered" for item in attempts)


def test_gemini_live_websocket_text_turn(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            seen["model"] = model
            return {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "live ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    with client.websocket_connect("/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent") as ws:
        ws.send_json({"setup": {"model": "models/gemini-3-flash-agent"}})
        assert ws.receive_json() == {"setupComplete": {}}
        ws.send_json({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": "hello live"}]}],
                "turnComplete": True,
            }
        })
        response = ws.receive_json()

    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "hello live"
    assert response["serverContent"]["turnComplete"] is True
    assert response["serverContent"]["modelTurn"]["parts"][0]["text"] == "live ok"


def test_gemini_live_websocket_accepts_query_api_key(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_PROXY_API_KEY", "secret")

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "auth live"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    with client.websocket_connect("/v1/live?key=secret") as ws:
        ws.send_json({"setup": {"model": "models/gemini-3-flash-agent"}})
        assert ws.receive_json() == {"setupComplete": {}}
        ws.send_json({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": "hello"}]}],
                "turnComplete": True,
            }
        })
        response = ws.receive_json()

    assert response["serverContent"]["modelTurn"]["parts"][0]["text"] == "auth live"


def test_gemini_live_websocket_rejects_realtime_media(monkeypatch):
    client = TestClient(proxy.app)

    with client.websocket_connect("/v1beta/live") as ws:
        ws.send_json({"setup": {"model": "models/gemini-3-flash-agent"}})
        assert ws.receive_json() == {"setupComplete": {}}
        ws.send_json({"realtimeInput": {"mediaChunks": [{"mimeType": "audio/pcm", "data": "AAAA"}]}})
        response = ws.receive_json()

    assert response["error"]["status"] == "UNIMPLEMENTED"


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
    assert file_resource["sha256Hash"] == "84d+ij2Y+AnZ+EQGD76ihkpLZpgKIv8iKXAU0MFo2y4="
    assert file_resource["downloadUri"].endswith(":download")
    assert file_resource["source"] == "UPLOADED"

    listed = client.get("/v1beta/files")
    assert listed.status_code == 200
    assert listed.json()["files"][0]["name"] == file_resource["name"]

    fetched = client.get(f"/v1beta/{file_resource['name']}")
    assert fetched.status_code == 200
    assert fetched.json()["displayName"] == "note.txt"

    downloaded = client.get(f"/v1beta/{file_resource['name']}:download")
    assert downloaded.status_code == 200
    assert downloaded.content == b"hello file"

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


def test_gemini_files_register_metadata_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "gemini_files"))
    client = TestClient(proxy.app)

    registered = client.post("/v1beta/files:register", json={
        "file": {
            "displayName": "external.txt",
            "mimeType": "text/plain",
            "uri": "gs://bucket/external.txt",
            "downloadUri": "https://storage.example/external.txt",
            "sizeBytes": "12",
        }
    })

    assert registered.status_code == 200
    file_resource = registered.json()["file"]
    assert file_resource["displayName"] == "external.txt"
    assert file_resource["uri"] == "gs://bucket/external.txt"
    assert file_resource["downloadUri"] == "https://storage.example/external.txt"
    assert file_resource["source"] == "REGISTERED"

    video = client.post("/v1beta/files:register", json={
        "file": {
            "displayName": "clip.mp4",
            "mimeType": "video/mp4",
            "uri": "gs://bucket/clip.mp4",
            "videoMetadata": {"videoDuration": "3s"},
        }
    })
    assert video.status_code == 200
    assert video.json()["file"]["videoMetadata"]["videoDuration"] == "3s"

    fetched = client.get(f"/v1beta/{file_resource['name']}")
    downloaded = client.get(f"/v1beta/{file_resource['name']}:download")

    assert fetched.status_code == 200
    assert downloaded.status_code == 404


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

    wrapped_created = client.post("/v1beta/cachedContents", json={
        "cachedContent": {
            "model": "models/gemini-3-flash-agent",
            "contents": [{"role": "user", "parts": [{"text": "wrapped cached context"}]}],
            "ttl": "60s",
        }
    })
    assert wrapped_created.status_code == 200
    assert wrapped_created.json()["model"] == "models/gemini-3-flash-agent"
    assert wrapped_created.json()["expireTime"]

    patched = client.patch(f"/v1beta/{cache_name}?update_mask=ttl", json={
        "cachedContent": {"ttl": "120s"}
    })
    assert patched.status_code == 200
    assert patched.json()["ttl"] == "120s"
    assert patched.json()["expireTime"]

    patched_expire = client.patch(f"/v1beta/{cache_name}?updateMask=expireTime", json={
        "expireTime": "2099-01-01T00:00:00Z"
    })
    assert patched_expire.status_code == 200
    assert patched_expire.json()["expireTime"] == "2099-01-01T00:00:00Z"
    assert "ttl" not in patched_expire.json()

    bad_patch = client.patch(f"/v1beta/{cache_name}?updateMask=contents", json={
        "contents": [{"role": "user", "parts": [{"text": "no"}]}],
    })
    assert bad_patch.status_code == 400

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


def test_gemini_corpora_documents_chunks_permissions_and_query(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_CORPORA_DIR", str(tmp_path / "corpora"))
    client = TestClient(proxy.app)

    created = client.post("/v1beta/corpora", json={"displayName": "Knowledge"})
    assert created.status_code == 200
    corpus_name = created.json()["name"]
    corpus_id = corpus_name.split("/", 1)[1]

    document = client.post(f"/v1beta/corpora/{corpus_id}/documents", json={"displayName": "Launch notes"})
    assert document.status_code == 200
    doc_name = document.json()["name"]
    doc_id = doc_name.rsplit("/", 1)[-1]

    chunk = client.post(f"/v1beta/corpora/{corpus_id}/documents/{doc_id}/chunks", json={
        "data": {"stringValue": "Project Atlas launch window is October."},
    })
    assert chunk.status_code == 200
    chunk_id = chunk.json()["name"].rsplit("/", 1)[-1]

    queried = client.post(f"/v1beta/corpora/{corpus_id}:query", json={"query": "Atlas October"})
    doc_queried = client.post(f"/v1beta/corpora/{corpus_id}/documents/{doc_id}:query", json={"query": "launch"})
    listed_chunks = client.get(f"/v1beta/corpora/{corpus_id}/documents/{doc_id}/chunks")
    fetched_chunk = client.get(f"/v1beta/corpora/{corpus_id}/documents/{doc_id}/chunks/{chunk_id}")
    patched_chunk = client.patch(f"/v1beta/corpora/{corpus_id}/documents/{doc_id}/chunks/{chunk_id}", json={
        "data": {"stringValue": "Project Atlas moved to November."},
    })

    assert queried.status_code == 200
    assert queried.json()["relevantChunks"][0]["chunk"]["name"] == chunk.json()["name"]
    assert doc_queried.status_code == 200
    assert listed_chunks.json()["chunks"][0]["name"] == chunk.json()["name"]
    assert fetched_chunk.json()["data"]["stringValue"].endswith("October.")
    assert patched_chunk.json()["data"]["stringValue"].endswith("November.")

    perm = client.post(f"/v1beta/corpora/{corpus_id}/permissions", json={
        "emailAddress": "reader@example.com",
        "role": "READER",
    })
    perm_id = perm.json()["name"].rsplit("/", 1)[-1]
    fetched_perm = client.get(f"/v1beta/corpora/{corpus_id}/permissions/{perm_id}")
    patched_perm = client.patch(f"/v1beta/corpora/{corpus_id}/permissions/{perm_id}", json={"role": "WRITER"})

    assert perm.status_code == 200
    assert fetched_perm.json()["role"] == "READER"
    assert patched_perm.json()["role"] == "WRITER"

    assert client.delete(f"/v1beta/corpora/{corpus_id}/documents/{doc_id}/chunks/{chunk_id}").status_code == 200
    assert client.delete(f"/v1beta/corpora/{corpus_id}/permissions/{perm_id}").status_code == 200
    assert client.delete(f"/v1beta/corpora/{corpus_id}/documents/{doc_id}").status_code == 200
    assert client.delete(f"/v1beta/corpora/{corpus_id}").status_code == 200


def test_gemini_file_search_store_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILE_SEARCH_STORES_DIR", str(tmp_path / "fss"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    client = TestClient(proxy.app)

    created = client.post("/v1beta/fileSearchStores", json={"displayName": "notes"})
    assert created.status_code == 200
    store_name = created.json()["name"]
    store_id = store_name.split("/", 1)[1]

    uploaded_file = client.post(
        "/upload/v1beta/files?uploadType=media&displayName=source.txt",
        content=b"source document",
        headers={"Content-Type": "text/plain"},
    ).json()["file"]

    imported = client.post(f"/v1beta/fileSearchStores/{store_id}:importFile", json={"fileName": uploaded_file["name"]})
    assert imported.status_code == 200
    imported_doc = imported.json()["response"]["document"]

    uploaded_doc = client.post(
        f"/upload/v1beta/fileSearchStores/{store_id}:uploadToFileSearchStore?displayName=direct.txt",
        content=b"direct document",
        headers={"Content-Type": "text/plain"},
    )
    assert uploaded_doc.status_code == 200

    listed = client.get(f"/v1beta/fileSearchStores/{store_id}/documents")
    assert listed.status_code == 200
    assert len(listed.json()["documents"]) == 2

    fetched = client.get(f"/v1beta/{imported_doc['name']}")
    assert fetched.status_code == 200
    assert fetched.json()["displayName"] == "source.txt"

    operation = client.get(f"/v1beta/{imported.json()['name']}")
    assert operation.status_code == 200
    assert operation.json()["done"] is True

    uploaded_op_id = uploaded_doc.json()["name"].split("/", 1)[1]
    nested_operation = client.get(f"/v1beta/fileSearchStores/{store_id}/{imported.json()['name']}")
    nested_operations = client.get(f"/v1beta/fileSearchStores/{store_id}/operations")
    waited_operation = client.post(f"/v1beta/fileSearchStores/{store_id}/{imported.json()['name']}:wait")
    upload_operation = client.get(f"/v1beta/fileSearchStores/{store_id}/upload/operations/{uploaded_op_id}")
    waited_upload_operation = client.post(f"/v1beta/fileSearchStores/{store_id}/upload/operations/{uploaded_op_id}:wait")
    media = client.get(f"/v1beta/fileSearchStores/{store_id}/media/{imported_doc['name'].rsplit('/', 1)[-1]}")
    assert nested_operation.status_code == 200
    assert nested_operation.json()["name"] == imported.json()["name"]
    assert nested_operations.status_code == 200
    assert nested_operations.json()["operations"][0]["name"] == imported.json()["name"]
    assert waited_operation.status_code == 200
    assert waited_operation.json()["name"] == imported.json()["name"]
    assert upload_operation.status_code == 200
    assert upload_operation.json()["name"] == uploaded_doc.json()["name"]
    assert waited_upload_operation.status_code == 200
    assert waited_upload_operation.json()["name"] == uploaded_doc.json()["name"]
    assert media.status_code == 200
    assert media.content == b"source document"

    deleted_doc = client.delete(f"/v1beta/{imported_doc['name']}")
    deleted_store = client.delete(f"/v1beta/{store_name}")

    assert deleted_doc.status_code == 200
    assert deleted_store.status_code == 200


def test_gemini_file_search_tool_injects_local_context(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILE_SEARCH_STORES_DIR", str(tmp_path / "fss"))
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "searched"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    store = client.post("/v1beta/fileSearchStores", json={"displayName": "knowledge"}).json()
    store_id = store["name"].split("/", 1)[1]
    uploaded = client.post(
        f"/upload/v1beta/fileSearchStores/{store_id}:uploadToFileSearchStore?displayName=plan.txt",
        content=b"Project Atlas launch window is October.",
        headers={"Content-Type": "text/plain"},
    )
    assert uploaded.status_code == 200

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "When is Project Atlas launch?"}]}],
        "tools": [{"file_search": {"file_search_store_names": [store["name"]]}}],
    })

    assert response.status_code == 200
    injected = seen["request"]["contents"][0]["parts"][0]["text"]
    assert "Local Gemini file_search results" in injected
    assert "Project Atlas launch window is October" in injected
    assert seen["request"]["contents"][1]["parts"][0]["text"] == "When is Project Atlas launch?"
    assert "tools" not in seen["request"]


def test_gemini_tuned_models_permissions_and_generate(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_TUNED_MODELS_DIR", str(tmp_path / "tuned"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            seen["model"] = model
            return {"response": {"candidates": [{"content": {"parts": [{"text": "tuned ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1beta/tunedModels", json={
        "tunedModelId": "my_tuned",
        "tunedModel": {
            "displayName": "My tuned",
            "baseModel": "models/gemini-3-flash-agent",
        },
    })
    assert created.status_code == 200
    tuned = created.json()["response"]
    assert tuned["name"] == "tunedModels/my_tuned"
    created_op_id = created.json()["name"].split("/", 1)[1]

    listed = client.get("/v1beta/tunedModels")
    fetched = client.get("/v1beta/tunedModels/my_tuned")
    listed_operations = client.get("/v1beta/tunedModels/my_tuned/operations")
    fetched_operation = client.get(f"/v1beta/tunedModels/my_tuned/operations/{created_op_id}")
    waited_operation = client.post(f"/v1beta/tunedModels/my_tuned/operations/{created_op_id}:wait")
    patched = client.patch("/v1beta/tunedModels/my_tuned", json={"description": "updated"})
    assert listed.status_code == 200
    assert fetched.json()["displayName"] == "My tuned"
    assert listed_operations.status_code == 200
    assert listed_operations.json()["operations"][0]["name"] == created.json()["name"]
    assert fetched_operation.status_code == 200
    assert fetched_operation.json()["name"] == created.json()["name"]
    assert waited_operation.status_code == 200
    assert waited_operation.json()["name"] == created.json()["name"]
    assert patched.json()["description"] == "updated"

    perm = client.post("/v1beta/tunedModels/my_tuned/permissions", json={
        "emailAddress": "user@example.com",
        "role": "READER",
    })
    assert perm.status_code == 200
    perm_id = perm.json()["name"].rsplit("/", 1)[-1]
    promoted = client.post(f"/v1beta/tunedModels/my_tuned/permissions/{perm_id}:transferOwnership")
    fetched_perm = client.get(f"/v1beta/tunedModels/my_tuned/permissions/{perm_id}")
    assert promoted.status_code == 200
    assert fetched_perm.json()["role"] == "OWNER"

    generated = client.post("/v1beta/tunedModels/my_tuned:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "hello tuned"}]}]
    })
    assert generated.status_code == 200
    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "hello tuned"

    counted = client.post("/v1beta/tunedModels/my_tuned:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "hello tuned"}]}]
    })
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0
    assert counted.json()["promptTokensDetails"][0]["modality"] == "TEXT"
    assert counted.json()["cacheTokensDetails"] == []

    deleted_perm = client.delete(f"/v1beta/tunedModels/my_tuned/permissions/{perm_id}")
    deleted_model = client.delete("/v1beta/tunedModels/my_tuned")
    assert deleted_perm.status_code == 200
    assert deleted_model.status_code == 200


def test_openai_image_generation_registers_gemini_generated_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_GENERATED_FILES_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))

    class FakeClient:
        def generate_image(self, *, prompt, output_dir, aspect_ratio="", image_size=""):
            output = output_dir / "image.png"
            output.write_bytes(b"fake-png")
            return output

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1/images/generations", json={"prompt": "draw a square"})

    assert created.status_code == 200
    generated_name = created.json()["data"][0]["generated_file"]
    listed = client.get("/v1beta/generatedFiles")
    operations = client.get("/v1beta/generatedFiles/operations")
    fetched = client.get(f"/v1beta/{generated_name}")
    downloaded = client.get(f"/v1beta/{generated_name}:download")

    assert listed.status_code == 200
    assert listed.json()["generatedFiles"][0]["name"] == generated_name
    assert operations.status_code == 200
    assert operations.json()["operations"][0]["metadata"]["generatedFile"] == generated_name
    assert fetched.status_code == 200
    assert fetched.json()["mimeType"] == "image/png"
    assert downloaded.status_code == 200
    assert downloaded.content == b"fake-png"


def test_gemini_image_model_generate_content_predict_and_generate_images(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_GENERATED_FILES_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))

    class FakeClient:
        def generate_image(self, *, prompt, output_dir, aspect_ratio="", image_size=""):
            output = output_dir / "gemini-image.png"
            output.write_bytes(b"gemini-image")
            return output

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    content = client.post("/v1beta/models/gemini-image-latest:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "draw image"}]}],
    })
    predict = client.post("/v1beta/models/gemini-image-latest:predict", json={
        "instances": [{"prompt": "draw predict"}],
    })
    generated = client.post("/v1beta/models/gemini-image-latest:generateImages", json={
        "prompt": "draw generated",
    })

    assert content.status_code == 200
    inline = content.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
    assert inline["mimeType"] == "image/png"
    assert inline["data"]
    assert content.json()["generatedFile"].startswith("generatedFiles/")

    assert predict.status_code == 200
    assert predict.json()["predictions"][0]["bytesBase64Encoded"]
    assert predict.json()["predictions"][0]["generatedFile"].startswith("generatedFiles/")

    assert generated.status_code == 200
    generated_image = generated.json()["generatedImages"][0]
    assert generated_image["image"]["imageBytes"]
    assert generated_image["generatedFile"]["name"].startswith("generatedFiles/")

    listed = client.get("/v1beta/generatedFiles")
    assert listed.status_code == 200
    assert len(listed.json()["generatedFiles"]) == 3


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
