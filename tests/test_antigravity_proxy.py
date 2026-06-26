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


def test_gemini_error_status_mapping_for_quota_and_server_errors():
    quota = proxy._gemini_error_payload("Too many requests.", status_code=429)["error"]
    timeout = proxy._gemini_error_payload("Timed out.", status_code=504)["error"]
    internal = proxy._gemini_error_payload("Server exploded.", status_code=500)["error"]

    assert quota["status"] == "RESOURCE_EXHAUSTED"
    assert quota["details"][0]["reason"] == "RESOURCE_EXHAUSTED"
    assert timeout["status"] == "DEADLINE_EXCEEDED"
    assert timeout["details"][0]["reason"] == "DEADLINE_EXCEEDED"
    assert internal["status"] == "INTERNAL"
    assert internal["details"][0]["reason"] == "INTERNAL"


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
    paged = client.get("/v1beta/models?pageSize=1")
    next_page = client.get(f"/v1beta/models?pageSize=1&pageToken={paged.json()['nextPageToken']}")
    too_many = client.get("/v1beta/models?pageSize=1001")

    assert paged.status_code == 200
    assert len(paged.json()["models"]) == 1
    assert paged.json()["nextPageToken"] == "1"
    assert next_page.status_code == 200
    assert len(next_page.json()["models"]) == 1
    assert too_many.status_code == 400
    assert too_many.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert too_many.json()["error"]["details"][0]["@type"] == "type.googleapis.com/google.rpc.BadRequest"
    assert too_many.json()["error"]["details"][0]["fieldViolations"][0]["field"] == "pageSize"

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
    counted_string = client.post("/v1/models/gemini-3-flash-agent:countTokens", json={
        "contents": "hello from sdk string"
    })
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0
    assert counted.json()["promptTokensDetails"][0]["modality"] == "TEXT"
    assert counted.json()["cachedContentTokenCount"] == 0
    assert counted.json()["cacheTokensDetails"] == []
    assert counted_string.status_code == 200
    assert counted_string.json()["totalTokens"] > 0
    assert counted_string.json()["promptTokensDetails"][0]["modality"] == "TEXT"

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
    assert counted.json()["cachedContentTokenCount"] > 0
    assert counted.json()["cacheTokensDetails"][0]["tokenCount"] == counted.json()["cachedContentTokenCount"]


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
    assert rejected.json()["error"]["status"] == "UNAUTHENTICATED"
    assert rejected.json()["error"]["details"][0]["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"


def test_gemini_unmatched_routes_use_gemini_error_shape():
    client = TestClient(proxy.app)

    gemini_missing = client.get("/v1beta/not-a-route")
    openai_missing = client.get("/v1/chat/completions/not-a-route")

    assert gemini_missing.status_code == 404
    assert gemini_missing.json()["error"]["status"] == "NOT_FOUND"
    assert gemini_missing.json()["error"]["code"] == 404
    assert openai_missing.status_code == 404
    assert openai_missing.json()["error"]["type"] == "invalid_request_error"


def test_gemini_v1_model_routes_do_not_break_openai_models(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "files"))
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "stable ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    openai_models = client.get("/v1/models")
    fetched_model = client.get("/v1/models/gemini-3-flash-agent")
    generated = client.post("/v1/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    })
    counted = client.post("/v1/models/gemini-3-flash-agent:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    })
    embedded = client.post("/v1/models/gemini-3-flash-agent:embedContent", json={
        "content": {"parts": [{"text": "embed me"}]},
        "outputDimensionality": 8,
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
    assert fetched_model.status_code == 200
    assert fetched_model.json()["name"] == "models/gemini-3-flash-agent"
    assert generated.status_code == 200
    assert generated.json()["candidates"][0]["content"]["parts"][0]["text"] == "stable ok"
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0
    assert embedded.status_code == 200
    assert len(embedded.json()["embedding"]["values"]) == 8
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


def test_gemini_embeddings_accept_latest_config_and_contents_forms():
    client = TestClient(proxy.app)

    configured = client.post("/v1beta/models/gemini-3-flash-agent:embedContent", json={
        "contents": ["alpha", "beta"],
        "config": {
            "outputDimensionality": 14,
            "taskType": "RETRIEVAL_DOCUMENT",
            "title": "SDK Config",
        },
    })
    wrapped = client.post("/v1/models/gemini-3-flash-agent:batchEmbedContents", json={
        "requests": [{
            "contents": [{"parts": [{"text": "gamma"}]}, {"parts": [{"text": "delta"}]}],
            "embed_content_config": {"output_dimensionality": 11},
        }]
    })

    assert configured.status_code == 200
    assert len(configured.json()["embeddings"]) == 2
    assert len(configured.json()["embeddings"][0]["values"]) == 14
    assert configured.json()["embedding"] == configured.json()["embeddings"][0]
    assert configured.json()["embeddings"][0]["values"] != configured.json()["embeddings"][1]["values"]
    assert wrapped.status_code == 200
    assert len(wrapped.json()["embeddings"]) == 2
    assert len(wrapped.json()["embeddings"][0]["values"]) == 11


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
        "response_format": {
            "type": "json_schema",
            "schema": {
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "property_ordering": ["answer"],
                "any_of": [
                    {"properties": {"answer": {"type": "string"}}},
                    {"properties": {"answer": {"type": "string", "min_length": 1}}},
                ],
            },
        },
        "tools": [{"googleSearch": {}}],
        "tool_config": {"function_calling_config": {"mode": "any", "allowed_function_names": "lookup"}},
    })

    assert response.status_code == 200
    assert response.json()["candidates"][0]["content"]["parts"][0]["text"] == "hello"
    assert response.json()["candidates"][0]["index"] == 0
    assert response.json()["candidates"][0]["finishReason"] == "STOP"
    assert response.json()["modelVersion"] == "gemini-3-flash-agent"
    assert response.json()["responseId"].startswith("resp_")
    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["request"]["generationConfig"]["responseSchema"] == {
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "propertyOrdering": ["answer"],
        "anyOf": [
            {"properties": {"answer": {"type": "string"}}, "type": "object"},
            {"properties": {"answer": {"type": "string", "minLength": 1}}, "type": "object"},
        ],
        "type": "object",
    }
    assert seen["request"]["generationConfig"]["maxOutputTokens"] == 32
    assert seen["request"]["tools"] == [{"google_search": {}}]
    assert seen["request"]["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["lookup"],
    }


def test_gemini_generate_content_normalizes_response_usage_and_content(monkeypatch):
    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {
                "response": {
                    "candidates": [{
                        "content": {"parts": "hello"},
                        "finish_reason": "MAX_TOKENS",
                        "safety_ratings": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "LOW"}],
                        "grounding_metadata": {"searchEntryPoint": {"renderedContent": "x"}},
                        "avg_logprobs": -0.2,
                    }],
                    "usage_metadata": {
                        "prompt_tokens": 4,
                        "output_tokens": 2,
                        "prompt_tokens_details": [{"modality": "TEXT", "tokenCount": 4}],
                    },
                }
            }

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={"contents": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["candidates"][0]["content"] == {"role": "model", "parts": [{"text": "hello"}]}
    assert body["candidates"][0]["finishReason"] == "MAX_TOKENS"
    assert "finish_reason" not in body["candidates"][0]
    assert body["candidates"][0]["safetyRatings"][0]["probability"] == "LOW"
    assert body["candidates"][0]["groundingMetadata"]["searchEntryPoint"]["renderedContent"] == "x"
    assert body["candidates"][0]["avgLogprobs"] == -0.2
    assert body["usageMetadata"]["promptTokenCount"] == 4
    assert body["usageMetadata"]["candidatesTokenCount"] == 2
    assert body["usageMetadata"]["promptTokensDetails"][0]["modality"] == "TEXT"
    assert body["usageMetadata"]["totalTokenCount"] >= body["usageMetadata"]["promptTokenCount"]
    assert "usage_metadata" not in body


def test_gemini_generate_content_accepts_sdk_content_unions(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    generated = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": ["hello", {"text": "world"}, [{"text": "part one"}, "part two"]],
        "system_instruction": "be brief",
    })
    counted = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "generate_content_request": {
            "contents": "count me",
            "system_instruction": "count system",
        }
    })

    assert generated.status_code == 200
    assert seen["request"]["contents"] == [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "user", "parts": [{"text": "world"}]},
        {"role": "user", "parts": [{"text": "part one"}, {"text": "part two"}]},
    ]
    assert seen["request"]["systemInstruction"] == {"role": "system", "parts": [{"text": "be brief"}]}
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0

    bytes_part = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{
            "role": "user",
            "parts": [
                {"data": "aW1hZ2U=", "mime_type": "image/png"},
                {"text": "describe bytes"},
            ],
        }]
    })
    assert bytes_part.status_code == 200
    assert seen["request"]["contents"][0]["parts"][0]["inlineData"] == {
        "mimeType": "image/png",
        "data": "aW1hZ2U=",
    }
    assert seen["request"]["contents"][0]["parts"][1]["text"] == "describe bytes"


def test_gemini_generate_content_accepts_sdk_config(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "request_options": {"timeout": 1},
        "config": {
            "system_instruction": "answer tersely",
            "max_output_tokens": 17,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop_sequences": "END",
            "response_modalities": "TEXT",
            "response_mime_type": "application/json",
            "response_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            "tool_config": {"function_calling_config": {"mode": "none"}},
            "labels": {"source": "sdk"},
            "http_options": {"api_version": "v1"},
            "api_version": "v1",
        },
    })

    assert response.status_code == 200
    assert "config" not in seen["request"]
    assert "requestOptions" not in seen["request"]
    assert "httpOptions" not in seen["request"]
    assert "apiVersion" not in seen["request"]
    assert seen["request"]["systemInstruction"] == {"role": "system", "parts": [{"text": "answer tersely"}]}
    assert seen["request"]["labels"] == {"source": "sdk"}
    assert seen["request"]["toolConfig"]["functionCallingConfig"] == {"mode": "NONE"}
    assert seen["request"]["generationConfig"]["maxOutputTokens"] == 17
    assert seen["request"]["generationConfig"]["temperature"] == 0.2
    assert seen["request"]["generationConfig"]["topP"] == 0.9
    assert seen["request"]["generationConfig"]["stopSequences"] == ["END"]
    assert seen["request"]["generationConfig"]["responseModalities"] == ["TEXT"]
    assert seen["request"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["request"]["generationConfig"]["responseSchema"]["type"] == "object"


def test_gemini_generate_content_normalizes_tools_unions(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    single_tool = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "tools": {"googleSearch": {}},
    })
    assert single_tool.status_code == 200
    assert seen["request"]["tools"] == [{"google_search": {}}]

    function_declarations = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "function_declarations": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    })

    assert function_declarations.status_code == 200
    assert seen["request"]["tools"] == [{
        "functionDeclarations": [{
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }]
    }]

    schema_aliases = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "tools": {
            "functionDeclarations": {
                "name": "typed_lookup",
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "response_json_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            }
        },
    })

    assert schema_aliases.status_code == 200
    declaration = seen["request"]["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == "typed_lookup"
    assert declaration["parameters"]["properties"]["query"]["type"] == "string"
    assert declaration["response"]["properties"]["answer"]["type"] == "string"
    assert "parametersJsonSchema" not in declaration
    assert "responseJsonSchema" not in declaration


def test_gemini_generate_content_normalizes_safety_and_tool_config_shortcuts(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "safety_settings": {
            "harm_category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "harm_block_threshold": "BLOCK_ONLY_HIGH",
        },
        "tool_config": {
            "mode": "any",
            "allowed_function_names": "lookup",
        },
    })

    assert response.status_code == 200
    assert seen["request"]["safetySettings"] == [{
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    }]
    assert seen["request"]["toolConfig"] == {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": ["lookup"],
        }
    }

    shortcut = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "safety_settings": {
            "harassment": "only_high",
            "dangerous": "none",
        },
    })
    assert shortcut.status_code == 200
    assert seen["request"]["safetySettings"] == [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    with_method = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "safety_settings": {
            "harm_category": "hate",
            "harm_block_threshold": "medium_and_above",
            "harm_block_method": "severity",
        },
    })
    assert with_method.status_code == 200
    assert seen["request"]["safetySettings"] == [{
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        "method": "SEVERITY",
    }]


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
    assert url_context.json()["error"]["details"] == [{
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "UNIMPLEMENTED",
        "domain": "generativelanguage.googleapis.com",
    }]
    assert code_execution.status_code == 501
    assert code_execution.json()["error"]["status"] == "UNIMPLEMENTED"
    assert "code_execution" in code_execution.json()["error"]["message"]

    single_url_context = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "read url",
        "tools": {"urlContext": {}},
    })
    assert single_url_context.status_code == 501
    assert single_url_context.json()["error"]["status"] == "UNIMPLEMENTED"


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


def test_gemini_generate_content_alt_sse_error_uses_gemini_error_details(monkeypatch):
    class FakeClient:
        async def generate_raw_stream_async(self, *, request, model=""):
            raise RuntimeError("stream down")
            yield {}

        def generate_raw(self, *, request, model=""):
            raise RuntimeError("fallback down")

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    with client.stream("POST", "/v1beta/models/gemini-3-flash-agent:generateContent?alt=sse", json={
        "contents": "hi",
    }) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert '"status": "UNAVAILABLE"' in body
    assert '"@type": "type.googleapis.com/google.rpc.ErrorInfo"' in body
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


def test_gemini_operations_v1_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    client = TestClient(proxy.app)
    operation = proxy._gemini_store_operation({
        "name": "operations/op_v1_alias",
        "metadata": {"model": "models/gemini-3-flash-agent"},
        "done": False,
    })

    listed = client.get("/v1/operations")
    fetched = client.get(f"/v1/{operation['name']}")
    waited = client.post(f"/v1/{operation['name']}:wait")
    cancelled = client.post(f"/v1/{operation['name']}:cancel")
    cancelled_fetched = client.get(f"/v1/{operation['name']}")
    deleted = client.delete(f"/v1/{operation['name']}")
    missing = client.get(f"/v1/{operation['name']}")

    assert listed.status_code == 200
    assert listed.json()["operations"][0]["name"] == operation["name"]
    assert fetched.status_code == 200
    assert fetched.json()["done"] is False
    assert waited.status_code == 200
    assert waited.json()["name"] == operation["name"]
    assert cancelled.status_code == 200
    assert cancelled.json() == {}
    assert cancelled_fetched.status_code == 200
    assert cancelled_fetched.json()["done"] is True
    assert cancelled_fetched.json()["error"]["status"] == "CANCELLED"
    assert deleted.status_code == 200
    assert missing.status_code == 404


def test_gemini_operations_delete_accepts_scoped_resource_name(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    client = TestClient(proxy.app)
    operation = proxy._gemini_store_operation({
        "name": "operations/scoped_delete",
        "metadata": {"model": "models/gemini-3-flash-agent"},
        "done": True,
    })

    scoped_name = f"models/gemini-3-flash-agent/{operation['name']}"
    fetched = client.get(f"/v1beta/operations/{scoped_name}")
    deleted = client.delete(f"/v1beta/operations/{scoped_name}")
    missing = client.get(f"/v1beta/{operation['name']}")

    assert fetched.status_code == 200
    assert fetched.json()["name"] == operation["name"]
    assert deleted.status_code == 200
    assert missing.status_code == 404


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
    generated_request_wrapped = client.post("/v1beta/models/gemini-3-flash-agent:batchGenerateContent", json={
        "requests": [{
            "metadata": {"key": "row-1"},
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "request wrapped"}]}],
            },
        }]
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
    embedded_request_wrapped = client.post("/v1beta/batches", json={
        "embedContentBatch": {
            "model": "models/gemini-3-flash-agent",
            "displayName": "wrapped embed request",
            "requests": [{
                "embedContentRequest": {
                    "content": {"parts": [{"text": "embed request wrapped"}]},
                    "config": {"output_dimensionality": 6},
                },
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
    assert generated_request_wrapped.status_code == 200
    assert generated_request_wrapped.json()["response"]["responses"][0]["candidates"][0]["content"]["parts"][0]["text"] == "request wrapped"
    assert embedded.status_code == 200
    assert embedded.json()["metadata"]["batchResource"]["displayName"] == "wrapped embed"
    assert embedded.json()["metadata"]["state"] == "BATCH_STATE_SUCCEEDED"
    assert embedded.json()["metadata"]["stats"]["successfulRequestCount"] == "1"
    assert embedded.json()["response"]["embeddings"][0]["values"]
    assert len(embedded.json()["response"]["embeddings"][0]["values"]) == 8
    assert embedded_request_wrapped.status_code == 200
    assert len(embedded_request_wrapped.json()["response"]["embeddings"][0]["values"]) == 6
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


def test_gemini_batches_v1_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", str(tmp_path / "batches"))

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {"response": {"candidates": [{"content": {"parts": [{"text": "v1 batch ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1/batches", json={
        "model": "models/gemini-3-flash-agent",
        "displayName": "v1 docs batch",
        "requests": [
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        ],
    })
    assert created.status_code == 200
    batch_operation = created.json()
    batch_name = batch_operation["name"]

    listed = client.get("/v1/batches")
    fetched = client.get(f"/v1/{batch_name}")
    patched = client.patch(f"/v1/{batch_name}:updateGenerateContentBatch?updateMask=displayName", json={
        "generateContentBatch": {"displayName": "v1 renamed batch"},
    })
    embed_patched = client.post(f"/v1/{batch_name}:updateEmbedContentBatch?updateMask=priority", json={
        "embedContentBatch": {"priority": "HIGH"},
    })
    cancelled = client.post(f"/v1/{batch_name}:cancel")
    deleted = client.delete(f"/v1/{batch_name}")
    missing = client.get(f"/v1/{batch_name}")

    assert listed.status_code == 200
    assert listed.json()["operations"][0]["name"] == batch_name
    assert listed.json()["batches"][0]["operation"] == batch_operation["metadata"]["batchResource"]["operation"]
    assert fetched.status_code == 200
    assert fetched.json()["metadata"]["batchResource"]["displayName"] == "v1 docs batch"
    assert patched.status_code == 200
    assert patched.json()["metadata"]["batchResource"]["displayName"] == "v1 renamed batch"
    assert embed_patched.status_code == 200
    assert embed_patched.json()["metadata"]["batchResource"]["priority"] == "HIGH"
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
    body_masked = client.patch(f"/v1beta/{first['name']}:updateGenerateContentBatch", json={
        "update_mask": "display_name",
        "generateContentBatch": {"displayName": "body masked"},
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
    assert body_masked.status_code == 200
    assert body_masked.json()["metadata"]["batchResource"]["displayName"] == "body masked"
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
    assert first_body["output"]["modelVersion"] == "gemini-3-flash-agent"
    assert first_body["output"]["responseId"].startswith("resp_")
    assert first_body["usageMetadata"]["totalTokenCount"] > 0
    assert first_body["steps"][0]["type"] == "model_output"
    assert first_body["steps"][0]["content"][0]["type"] == "text"
    assert first_body["steps"][0]["content"][0]["text"] == "echo:first"
    assert first_body["steps"][0]["content"][0]["annotations"] == []

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
    assert "interaction.step.completed" in body
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


def test_gemini_interactions_v1_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_INTERACTIONS_DIR", str(tmp_path / "interactions"))

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            text = request["contents"][-1]["parts"][0]["text"]
            return {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": f"v1:{text}"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1/interactions", json={"input": "hello"})
    assert created.status_code == 200
    body = created.json()
    assert body["outputText"] == "v1:hello"

    fetched = client.get(f"/v1/{body['name']}")
    colon_cancelled = client.post(f"/v1/{body['name']}:cancel")
    rest_cancelled = client.post(f"/v1/{body['name']}/cancel")
    deleted = client.delete(f"/v1/{body['name']}")
    missing = client.get(f"/v1/{body['name']}")

    assert fetched.status_code == 200
    assert colon_cancelled.status_code == 200
    assert rest_cancelled.status_code == 200
    assert deleted.status_code == 200
    assert missing.status_code == 404

    with client.stream("POST", "/v1/interactions", json={"input": "stream", "stream": True}) as streamed:
        stream_body = streamed.read().decode()

    assert streamed.status_code == 200
    assert "interaction.created" in stream_body
    assert "interaction.step.completed" in stream_body
    assert "interaction.completed" in stream_body
    assert "data: [DONE]" in stream_body


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

    bytes_interaction = client.post("/v1beta/interactions", json={
        "input": [{
            "role": "user",
            "content": [
                {"data": "ZmlsZQ==", "mime_type": "application/pdf"},
                {"type": "text", "text": "read bytes"},
            ],
        }],
        "store": False,
    })
    assert bytes_interaction.status_code == 200
    parts = seen["request"]["contents"][0]["parts"]
    assert parts[0]["inlineData"] == {"mimeType": "application/pdf", "data": "ZmlsZQ=="}
    assert parts[1] == {"text": "read bytes"}

    structured_interaction = client.post("/v1/interactions", json={
        "input": "return json",
        "response_format": {
            "mime_type": "application/json",
            "schema": {"properties": {"ok": {"type": "boolean"}}},
        },
        "store": False,
    })

    assert structured_interaction.status_code == 200
    assert seen["request"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["request"]["generationConfig"]["responseSchema"] == {
        "properties": {"ok": {"type": "boolean"}},
        "type": "object",
    }

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

    created = client.post("/v1/webhooks", json={
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
    body_masked = client.patch(f"/v1/{webhook['name']}", json={
        "update_mask": "targetUri",
        "display_name": "Ignored by mask",
        "target_uri": "https://example.test/body-mask",
    })

    assert fetched.status_code == 200
    assert fetched.json()["name"] == webhook["name"]
    assert listed.status_code == 200
    assert listed.json()["webhooks"][0]["name"] == webhook["name"]
    assert "newSigningSecret" not in fetched.json()
    assert patched.status_code == 200
    assert patched.json()["displayName"] == "Renamed"
    assert patched.json()["targetUri"] == "https://example.test/hook"
    assert body_masked.status_code == 200
    assert body_masked.json()["displayName"] == "Renamed"
    assert body_masked.json()["targetUri"] == "https://example.test/body-mask"

    malformed = client.patch(
        f"/v1/{webhook['name']}",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert malformed.status_code == 400
    assert malformed.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert malformed.json()["error"]["details"][0]["@type"] == "type.googleapis.com/google.rpc.BadRequest"
    assert malformed.json()["error"]["details"][0]["fieldViolations"][0]["description"]

    pinged = client.post(f"/v1/{webhook['name']}:ping")
    assert pinged.status_code == 200
    assert pinged.json()["deliveryAttempt"]["eventType"] == "webhooks.ping"

    rotated = client.post(f"/v1/{webhook['name']}:rotateSigningSecret")
    assert rotated.status_code == 200
    assert rotated.json()["newSigningSecret"]["secret"]
    assert rotated.json()["newSigningSecret"]["secret"] != webhook["newSigningSecret"]["secret"]
    assert "secret" not in rotated.json()["signingSecrets"][0]

    deleted = client.delete(f"/v1/{webhook['name']}")
    missing = client.get(f"/v1/{webhook['name']}")
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
        ws.send_json({
            "setup": {
                "model": "models/gemini-3-flash-agent",
                "system_instruction": {"parts": [{"text": "be terse"}]},
                "generation_config": {"max_output_tokens": 8},
                "response_format": {
                    "mime_type": "application/json",
                    "schema": {"properties": {"ok": {"type": "boolean"}}},
                },
            }
        })
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
    assert seen["request"]["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert seen["request"]["generationConfig"]["maxOutputTokens"] == 8
    assert seen["request"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["request"]["generationConfig"]["responseSchema"] == {
        "properties": {"ok": {"type": "boolean"}},
        "type": "object",
    }
    assert response["serverContent"]["turnComplete"] is True
    assert response["serverContent"]["modelTurn"]["parts"][0]["text"] == "live ok"
    assert response["serverContent"]["usageMetadata"]["totalTokenCount"] > 0


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
    assert response["error"]["code"] == 501
    assert response["error"]["details"][0]["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"


def test_gemini_live_websocket_errors_use_gemini_error_details(monkeypatch):
    class FakeClient:
        def generate_raw(self, *, request, model=""):
            raise RuntimeError("live down")

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    with client.websocket_connect("/v1beta/live") as ws:
        ws.send_text("{not json")
        malformed = ws.receive_json()
        ws.send_json({"clientContent": {"turns": [{"role": "user", "parts": [{"text": "hello"}]}], "turnComplete": True}})
        failed = ws.receive_json()

    assert malformed["error"]["status"] == "INVALID_ARGUMENT"
    assert malformed["error"]["details"][0]["@type"] == "type.googleapis.com/google.rpc.BadRequest"
    assert failed["error"]["status"] == "UNAVAILABLE"
    assert failed["error"]["details"][0]["@type"] == "type.googleapis.com/google.rpc.ErrorInfo"


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


def test_gemini_file_resource_content_part_inline_conversion(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "gemini_files"))
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "file resource ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    uploaded = client.post(
        "/upload/v1beta/files?uploadType=media&displayName=file-object.txt",
        content=b"file object body",
        headers={"Content-Type": "text/plain"},
    )
    assert uploaded.status_code == 200
    file_resource = uploaded.json()["file"]

    generated = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{
            "role": "user",
            "parts": [
                file_resource,
                {"text": "summarize this uploaded file"},
            ],
        }]
    })

    assert generated.status_code == 200
    parts = seen["request"]["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "text/plain"
    assert parts[0]["inlineData"]["data"]
    assert parts[1]["text"] == "summarize this uploaded file"


def test_gemini_upload_query_snake_case_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "gemini_files"))
    client = TestClient(proxy.app)

    uploaded = client.post(
        "/upload/v1beta/files?upload_type=media&display_name=snake-note.txt",
        content=b"snake upload",
        headers={"Content-Type": "text/plain"},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["file"]["displayName"] == "snake-note.txt"

    started = client.post(
        "/upload/v1beta/files?upload_type=resumable&display_name=snake-resumable.txt",
        json={"config": {"mime_type": "text/plain"}},
        headers={"X-Goog-Upload-Command": "start"},
    )
    assert started.status_code == 200
    session_path = "/" + started.headers["x-goog-upload-url"].split("/", 3)[3]

    finished = client.post(
        session_path,
        content=b"snake resumable",
        headers={"X-Goog-Upload-Command": "upload, finalize"},
    )
    assert finished.status_code == 200
    assert finished.json()["file"]["displayName"] == "snake-resumable.txt"
    assert finished.json()["file"]["mimeType"] == "text/plain"


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

    official_created = client.post("/v1beta/files", json={
        "file": {
            "displayName": "official.txt",
            "mimeType": "text/plain",
            "uri": "gs://bucket/official.txt",
            "sizeBytes": "5",
        }
    })
    assert official_created.status_code == 200
    official_file = official_created.json()["file"]
    assert official_file["displayName"] == "official.txt"
    assert official_file["uri"] == "gs://bucket/official.txt"
    assert official_file["source"] == "REGISTERED"
    official_download = client.get(f"/v1beta/{official_file['name']}:download")
    assert official_download.status_code == 404

    config_created = client.post("/v1beta/files", json={
        "file": {
            "displayName": "config-file.txt",
            "uri": "gs://bucket/config-file.txt",
        },
        "config": {"mime_type": "text/markdown", "sizeBytes": "9"},
    })
    assert config_created.status_code == 200
    assert config_created.json()["file"]["mimeType"] == "text/markdown"
    assert config_created.json()["file"]["sizeBytes"] == "9"

    official_registered = client.post("/v1beta/files:register", json={
        "uris": ["gs://bucket/one.txt", "gs://bucket/two.txt"]
    })
    assert official_registered.status_code == 200
    official_registered_files = official_registered.json()["files"]
    assert [item["uri"] for item in official_registered_files] == [
        "gs://bucket/one.txt",
        "gs://bucket/two.txt",
    ]
    assert all(item["source"] == "REGISTERED" for item in official_registered_files)

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

    for idx in range(12):
        created = client.post("/v1beta/files:register", json={
            "file": {"displayName": f"page-{idx}.txt", "uri": f"gs://bucket/page-{idx}.txt"}
        })
        assert created.status_code == 200

    default_page = client.get("/v1beta/files")
    second_page = client.get(f"/v1beta/files?page_token={default_page.json()['nextPageToken']}")
    too_large_page = client.get("/v1beta/files?pageSize=101")

    assert len(default_page.json()["files"]) == 10
    assert default_page.json()["nextPageToken"]
    assert second_page.status_code == 200
    assert second_page.json()["files"]
    assert too_large_page.status_code == 400
    assert too_large_page.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert too_large_page.json()["error"]["details"][0]["fieldViolations"][0]["field"] == "pageSize"

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

    started_with_config = client.post(
        "/upload/v1beta/files",
        json={
            "file": {"displayName": "sdk-config.txt"},
            "config": {"mime_type": "text/markdown"},
        },
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
        },
    )
    assert started_with_config.status_code == 200
    config_session_path = "/" + started_with_config.headers["x-goog-upload-url"].split("/", 3)[3]

    config_finished = client.post(
        config_session_path,
        content=b"# hello",
        headers={"X-Goog-Upload-Command": "upload, finalize"},
    )
    assert config_finished.status_code == 200
    assert config_finished.json()["file"]["displayName"] == "sdk-config.txt"
    assert config_finished.json()["file"]["mimeType"] == "text/markdown"

    started_by_query = client.post(
        "/upload/v1beta/files?uploadType=resumable",
        json={
            "file": {"displayName": "query-resumable.txt"},
            "config": {"mime_type": "text/plain"},
        },
        headers={"X-Goog-Upload-Command": "start"},
    )
    assert started_by_query.status_code == 200
    query_session_path = "/" + started_by_query.headers["x-goog-upload-url"].split("/", 3)[3]

    query_finished = client.post(
        query_session_path,
        content=b"query resumable",
        headers={"X-Goog-Upload-Command": "upload, finalize"},
    )
    assert query_finished.status_code == 200
    assert query_finished.json()["file"]["displayName"] == "query-resumable.txt"
    assert query_finished.json()["file"]["mimeType"] == "text/plain"


def test_gemini_resumable_file_upload_query_offset_and_finalize(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "gemini_files"))
    client = TestClient(proxy.app)

    started = client.post(
        "/upload/v1/files",
        json={"file": {"displayName": "chunked.txt", "mimeType": "text/plain"}},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Type": "text/plain",
        },
    )
    session_path = "/" + started.headers["x-goog-upload-url"].split("/", 3)[3]

    initial_query = client.post(session_path, headers={"X-Goog-Upload-Command": "query"})
    first_chunk = client.post(
        session_path,
        content=b"hello ",
        headers={"X-Goog-Upload-Command": "upload", "X-Goog-Upload-Offset": "0"},
    )
    second_query = client.post(session_path, headers={"X-Goog-Upload-Command": "query"})
    wrong_offset = client.post(
        session_path,
        content=b"bad",
        headers={"X-Goog-Upload-Command": "upload", "X-Goog-Upload-Offset": "0"},
    )
    finalized = client.post(
        session_path,
        content=b"world",
        headers={"X-Goog-Upload-Command": "upload, finalize", "X-Goog-Upload-Offset": "6"},
    )

    assert initial_query.headers["x-goog-upload-size-received"] == "0"
    assert first_chunk.status_code == 200
    assert first_chunk.headers["x-goog-upload-size-received"] == "6"
    assert second_query.headers["x-goog-upload-size-received"] == "6"
    assert wrong_offset.status_code == 400
    assert finalized.status_code == 200
    file_resource = finalized.json()["file"]
    downloaded = client.get(f"/v1/{file_resource['name']}:download")
    assert downloaded.content == b"hello world"


def test_gemini_files_v1_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "gemini_files"))
    client = TestClient(proxy.app)

    uploaded = client.post(
        "/upload/v1/files?uploadType=media&displayName=v1-note.txt",
        content=b"v1 file",
        headers={"Content-Type": "text/plain"},
    )
    assert uploaded.status_code == 200
    file_resource = uploaded.json()["file"]
    assert file_resource["displayName"] == "v1-note.txt"

    assert client.get("/v1/files").json()["files"][0]["name"] == file_resource["name"]
    assert client.get(f"/v1/{file_resource['name']}").json()["displayName"] == "v1-note.txt"
    assert client.get(f"/v1/{file_resource['name']}:download").content == b"v1 file"

    registered = client.post("/v1/files", json={
        "file": {"displayName": "v1-external.txt", "uri": "gs://bucket/v1-external.txt"}
    })
    assert registered.status_code == 200
    assert registered.json()["file"]["source"] == "REGISTERED"

    registered_many = client.post("/v1/files:register", json={"uris": ["gs://bucket/v1-one.txt"]})
    assert registered_many.status_code == 200
    assert registered_many.json()["files"][0]["uri"] == "gs://bucket/v1-one.txt"

    started = client.post(
        "/upload/v1/files",
        json={"file": {"displayName": "v1-resumable.txt", "mimeType": "text/plain"}},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Type": "text/plain",
        },
    )
    assert started.status_code == 200
    assert "/upload/v1/files/" in started.headers["x-goog-upload-url"]

    deleted = client.delete(f"/v1/{file_resource['name']}")
    assert deleted.status_code == 200
    assert client.get(f"/v1/{file_resource['name']}").status_code == 404


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
            "contents": "wrapped cached context",
            "config": {
                "system_instruction": "wrapped cached system",
                "tool_config": {"mode": "none"},
                "safety_settings": {
                    "harm_category": "HARM_CATEGORY_HARASSMENT",
                    "harm_block_threshold": "BLOCK_ONLY_HIGH",
                },
            },
            "ttl": "60s",
        }
    })
    assert wrapped_created.status_code == 200
    assert wrapped_created.json()["model"] == "models/gemini-3-flash-agent"
    assert wrapped_created.json()["contents"] == [{"role": "user", "parts": [{"text": "wrapped cached context"}]}]
    assert wrapped_created.json()["systemInstruction"] == {
        "role": "system",
        "parts": [{"text": "wrapped cached system"}],
    }
    assert wrapped_created.json()["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}
    assert wrapped_created.json()["safetySettings"] == [{
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH",
    }]
    assert wrapped_created.json()["expireTime"]

    paged = client.get("/v1beta/cachedContents?pageSize=1")
    next_page = client.get(f"/v1beta/cachedContents?pageSize=1&pageToken={paged.json()['nextPageToken']}")
    coerced = client.get("/v1beta/cachedContents?pageSize=1001")

    assert paged.status_code == 200
    assert len(paged.json()["cachedContents"]) == 1
    assert paged.json()["nextPageToken"] == "1"
    assert next_page.status_code == 200
    assert next_page.json()["cachedContents"]
    assert coerced.status_code == 200

    patched = client.patch(f"/v1beta/{cache_name}?update_mask=ttl", json={
        "cachedContent": {"ttl": "120s"}
    })
    assert patched.status_code == 200
    assert patched.json()["ttl"] == "120s"
    assert patched.json()["expireTime"]

    patched_body_mask = client.patch(f"/v1beta/{cache_name}", json={
        "update_mask": "ttl",
        "cachedContent": {"ttl": "150s"},
    })
    assert patched_body_mask.status_code == 200
    assert patched_body_mask.json()["ttl"] == "150s"

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


def test_gemini_cached_contents_v1_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_CACHED_CONTENTS_DIR", str(tmp_path / "gemini_cached"))
    client = TestClient(proxy.app)

    created = client.post("/v1/cachedContents", json={
        "cachedContent": {
            "model": "models/gemini-3-flash-agent",
            "contents": [{"role": "user", "parts": [{"text": "v1 cached context"}]}],
            "ttl": "60s",
        }
    })
    assert created.status_code == 200
    cache_name = created.json()["name"]

    listed = client.get("/v1/cachedContents")
    fetched = client.get(f"/v1/{cache_name}")
    patched = client.patch(f"/v1/{cache_name}?updateMask=ttl", json={"ttl": "90s"})
    deleted = client.delete(f"/v1/{cache_name}")
    missing = client.get(f"/v1/{cache_name}")

    assert listed.status_code == 200
    assert listed.json()["cachedContents"][0]["name"] == cache_name
    assert fetched.status_code == 200
    assert fetched.json()["contents"][0]["parts"][0]["text"] == "v1 cached context"
    assert patched.status_code == 200
    assert patched.json()["ttl"] == "90s"
    assert deleted.status_code == 200
    assert missing.status_code == 404


def test_gemini_corpora_documents_chunks_permissions_and_query(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_CORPORA_DIR", str(tmp_path / "corpora"))
    client = TestClient(proxy.app)

    created = client.post("/v1/corpora", json={"displayName": "Knowledge"})
    assert created.status_code == 200
    corpus_name = created.json()["name"]
    corpus_id = corpus_name.split("/", 1)[1]

    listed_corpora = client.get("/v1/corpora")
    fetched_corpus = client.get(f"/v1/{corpus_name}")
    patched_corpus = client.patch(f"/v1/{corpus_name}", json={"displayName": "Knowledge updated"})

    assert listed_corpora.status_code == 200
    assert listed_corpora.json()["corpora"][0]["name"] == corpus_name
    assert fetched_corpus.status_code == 200
    assert fetched_corpus.json()["displayName"] == "Knowledge"
    assert patched_corpus.status_code == 200
    assert patched_corpus.json()["displayName"] == "Knowledge updated"

    document = client.post(f"/v1/corpora/{corpus_id}/documents", json={"displayName": "Launch notes"})
    assert document.status_code == 200
    doc_name = document.json()["name"]
    doc_id = doc_name.rsplit("/", 1)[-1]

    listed_docs = client.get(f"/v1/corpora/{corpus_id}/documents")
    fetched_doc = client.get(f"/v1/{doc_name}")
    patched_doc = client.patch(f"/v1/{doc_name}", json={"displayName": "Launch notes updated"})

    assert listed_docs.status_code == 200
    assert listed_docs.json()["documents"][0]["name"] == doc_name
    assert fetched_doc.status_code == 200
    assert fetched_doc.json()["displayName"] == "Launch notes"
    assert patched_doc.status_code == 200
    assert patched_doc.json()["displayName"] == "Launch notes updated"

    chunk = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks", json={
        "data": {"stringValue": "Project Atlas launch window is October."},
    })
    assert chunk.status_code == 200
    chunk_id = chunk.json()["name"].rsplit("/", 1)[-1]

    batch_created = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks:batchCreate", json={
        "requests": [{"chunk": {"chunkId": "batch_one", "data": {"stringValue": "Batch chunk text"}}}],
    })
    batch_updated = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks:batchUpdate", json={
        "requests": [{"chunk": {"name": "batch_one", "data": {"stringValue": "Batch chunk updated"}}}],
    })

    queried = client.post(f"/v1/corpora/{corpus_id}:query", json={"query": "Atlas October"})
    doc_queried = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}:query", json={"query": "launch"})
    listed_chunks = client.get(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks")
    fetched_chunk = client.get(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks/{chunk_id}")
    patched_chunk = client.patch(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks/{chunk_id}", json={
        "data": {"stringValue": "Project Atlas moved to November."},
    })

    assert batch_created.status_code == 200
    assert batch_created.json()["chunks"][0]["name"].endswith("/chunks/batch_one")
    assert batch_updated.status_code == 200
    assert batch_updated.json()["chunks"][0]["data"]["stringValue"] == "Batch chunk updated"
    assert queried.status_code == 200
    assert queried.json()["relevantChunks"][0]["chunk"]["name"] == chunk.json()["name"]
    assert doc_queried.status_code == 200
    assert chunk.json()["name"] in {item["name"] for item in listed_chunks.json()["chunks"]}
    assert fetched_chunk.json()["data"]["stringValue"].endswith("October.")
    assert patched_chunk.json()["data"]["stringValue"].endswith("November.")

    batch_deleted = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks:batchDelete", json={
        "names": [f"{doc_name}/chunks/batch_one"],
    })
    assert batch_deleted.status_code == 200

    perm = client.post(f"/v1/corpora/{corpus_id}/permissions", json={
        "emailAddress": "reader@example.com",
        "role": "READER",
    })
    perm_id = perm.json()["name"].rsplit("/", 1)[-1]
    listed_perms = client.get(f"/v1/corpora/{corpus_id}/permissions")
    fetched_perm = client.get(f"/v1/corpora/{corpus_id}/permissions/{perm_id}")
    patched_perm = client.patch(f"/v1/corpora/{corpus_id}/permissions/{perm_id}", json={"role": "WRITER"})

    assert perm.status_code == 200
    assert listed_perms.status_code == 200
    assert listed_perms.json()["permissions"][0]["role"] == "READER"
    assert fetched_perm.json()["role"] == "READER"
    assert patched_perm.json()["role"] == "WRITER"

    assert client.delete(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks/{chunk_id}").status_code == 200
    assert client.delete(f"/v1/corpora/{corpus_id}/permissions/{perm_id}").status_code == 200
    assert client.delete(f"/v1/corpora/{corpus_id}/documents/{doc_id}").status_code == 200
    assert client.delete(f"/v1/{corpus_name}").status_code == 200


def test_gemini_file_search_store_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILE_SEARCH_STORES_DIR", str(tmp_path / "fss"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    client = TestClient(proxy.app)

    created = client.post("/v1/fileSearchStores", json={
        "config": {
            "displayName": "notes",
            "embeddingModel": "models/text-embedding-004",
            "chunkingConfig": {"whiteSpaceConfig": {}},
            "customMetadata": [{"key": "team", "stringValue": "research"}],
        }
    })
    assert created.status_code == 200
    store_name = created.json()["name"]
    store_id = store_name.split("/", 1)[1]
    assert created.json()["displayName"] == "notes"
    assert created.json()["embeddingModel"] == "models/text-embedding-004"
    assert created.json()["chunkingConfig"] == {"whiteSpaceConfig": {}}
    assert created.json()["customMetadata"][0]["key"] == "team"

    uploaded_file = client.post(
        "/upload/v1beta/files?uploadType=media&displayName=source.txt",
        content=b"source document",
        headers={"Content-Type": "text/plain"},
    ).json()["file"]

    imported = client.post(f"/v1/fileSearchStores/{store_id}:importFile", json={
        "config": {
            "fileName": uploaded_file["name"],
            "displayName": "source import",
            "customMetadata": [{"key": "source", "stringValue": "files-api"}],
        }
    })
    assert imported.status_code == 200
    imported_doc = imported.json()["response"]["document"]
    assert imported_doc["displayName"] == "source import"
    assert imported_doc["customMetadata"][0]["stringValue"] == "files-api"

    uploaded_doc = client.post(
        f"/upload/v1/fileSearchStores/{store_id}:uploadToFileSearchStore?displayName=direct.txt",
        content=b"direct document",
        headers={"Content-Type": "text/plain"},
    )
    assert uploaded_doc.status_code == 200

    listed_stores = client.get("/v1/fileSearchStores")
    fetched_store = client.get(f"/v1/{store_name}")
    listed = client.get(f"/v1/fileSearchStores/{store_id}/documents")
    assert listed_stores.status_code == 200
    assert listed_stores.json()["fileSearchStores"][0]["name"] == store_name
    assert fetched_store.status_code == 200
    assert fetched_store.json()["name"] == store_name
    assert listed.status_code == 200
    assert len(listed.json()["documents"]) == 2

    fetched = client.get(f"/v1/{imported_doc['name']}")
    assert fetched.status_code == 200
    assert fetched.json()["displayName"] == "source import"
    assert fetched.json()["customMetadata"][0]["key"] == "source"

    operation = client.get(f"/v1/{imported.json()['name']}")
    assert operation.status_code == 200
    assert operation.json()["done"] is True

    uploaded_op_id = uploaded_doc.json()["name"].split("/", 1)[1]
    nested_operation = client.get(f"/v1/fileSearchStores/{store_id}/{imported.json()['name']}")
    nested_operations = client.get(f"/v1/fileSearchStores/{store_id}/operations")
    waited_operation = client.post(f"/v1/fileSearchStores/{store_id}/{imported.json()['name']}:wait")
    cancelled_operation = client.post(f"/v1/fileSearchStores/{store_id}/{imported.json()['name']}:cancel")
    upload_operation = client.get(f"/v1/fileSearchStores/{store_id}/upload/operations/{uploaded_op_id}")
    waited_upload_operation = client.post(f"/v1/fileSearchStores/{store_id}/upload/operations/{uploaded_op_id}:wait")
    cancelled_upload_operation = client.post(f"/v1/fileSearchStores/{store_id}/upload/operations/{uploaded_op_id}:cancel")
    media = client.get(f"/v1/fileSearchStores/{store_id}/media/{imported_doc['name'].rsplit('/', 1)[-1]}")
    assert nested_operation.status_code == 200
    assert nested_operation.json()["name"] == imported.json()["name"]
    assert nested_operations.status_code == 200
    assert nested_operations.json()["operations"][0]["name"] == imported.json()["name"]
    assert waited_operation.status_code == 200
    assert waited_operation.json()["name"] == imported.json()["name"]
    assert cancelled_operation.status_code == 200
    assert cancelled_operation.json() == {}
    assert upload_operation.status_code == 200
    assert upload_operation.json()["name"] == uploaded_doc.json()["name"]
    assert waited_upload_operation.status_code == 200
    assert waited_upload_operation.json()["name"] == uploaded_doc.json()["name"]
    assert cancelled_upload_operation.status_code == 200
    assert cancelled_upload_operation.json() == {}
    assert media.status_code == 200
    assert media.content == b"source document"

    deleted_upload_operation = client.delete(f"/v1/fileSearchStores/{store_id}/upload/operations/{uploaded_op_id}")
    deleted_doc = client.delete(f"/v1/{imported_doc['name']}")
    deleted_store = client.delete(f"/v1/{store_name}")

    assert deleted_upload_operation.status_code == 200
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

    created = client.post("/v1/tunedModels", json={
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

    listed = client.get("/v1/tunedModels")
    fetched = client.get("/v1/tunedModels/my_tuned")
    listed_operations = client.get("/v1/tunedModels/my_tuned/operations")
    fetched_operation = client.get(f"/v1/tunedModels/my_tuned/operations/{created_op_id}")
    waited_operation = client.post(f"/v1/tunedModels/my_tuned/operations/{created_op_id}:wait")
    cancelled_operation = client.post(f"/v1/tunedModels/my_tuned/operations/{created_op_id}:cancel")
    patched = client.patch("/v1/tunedModels/my_tuned", json={"description": "updated"})
    assert listed.status_code == 200
    assert fetched.json()["displayName"] == "My tuned"
    assert listed_operations.status_code == 200
    assert listed_operations.json()["operations"][0]["name"] == created.json()["name"]
    assert fetched_operation.status_code == 200
    assert fetched_operation.json()["name"] == created.json()["name"]
    assert waited_operation.status_code == 200
    assert waited_operation.json()["name"] == created.json()["name"]
    assert cancelled_operation.status_code == 200
    assert cancelled_operation.json() == {}
    assert patched.json()["description"] == "updated"

    perm = client.post("/v1/tunedModels/my_tuned/permissions", json={
        "emailAddress": "user@example.com",
        "role": "READER",
    })
    assert perm.status_code == 200
    perm_id = perm.json()["name"].rsplit("/", 1)[-1]
    listed_perms = client.get("/v1/tunedModels/my_tuned/permissions")
    patched_perm = client.patch(f"/v1/tunedModels/my_tuned/permissions/{perm_id}", json={"role": "WRITER"})
    promoted = client.post(f"/v1/tunedModels/my_tuned/permissions/{perm_id}:transferOwnership")
    fetched_perm = client.get(f"/v1/tunedModels/my_tuned/permissions/{perm_id}")
    assert listed_perms.status_code == 200
    assert listed_perms.json()["permissions"][0]["role"] == "READER"
    assert patched_perm.status_code == 200
    assert patched_perm.json()["role"] == "WRITER"
    assert promoted.status_code == 200
    assert fetched_perm.json()["role"] == "OWNER"

    generated = client.post("/v1/tunedModels/my_tuned:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "hello tuned"}]}]
    })
    assert generated.status_code == 200
    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "hello tuned"

    counted = client.post("/v1/tunedModels/my_tuned:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "hello tuned"}]}]
    })
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0
    assert counted.json()["promptTokensDetails"][0]["modality"] == "TEXT"
    assert counted.json()["cacheTokensDetails"] == []

    deleted_perm = client.delete(f"/v1/tunedModels/my_tuned/permissions/{perm_id}")
    deleted_operation = client.delete(f"/v1/tunedModels/my_tuned/operations/{created_op_id}")
    deleted_model = client.delete("/v1/tunedModels/my_tuned")
    assert deleted_perm.status_code == 200
    assert deleted_operation.status_code == 200
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
    v1_listed = client.get("/v1/generatedFiles")
    v1_operations = client.get("/v1/generatedFiles/operations")
    operation_id = operations.json()["operations"][0]["name"].rsplit("/", 1)[-1]
    v1_operation = client.get(f"/v1/generatedFiles/operations/{operation_id}")
    v1_waited = client.post(f"/v1/generatedFiles/operations/{operation_id}:wait")
    v1_cancelled = client.post(f"/v1/generatedFiles/operations/{operation_id}:cancel")
    v1_fetched = client.get(f"/v1/{generated_name}")
    v1_downloaded = client.get(f"/v1/{generated_name}:download")
    v1_deleted = client.delete(f"/v1/{generated_name}")
    v1_missing = client.get(f"/v1/{generated_name}")

    assert listed.status_code == 200
    assert listed.json()["generatedFiles"][0]["name"] == generated_name
    assert operations.status_code == 200
    assert operations.json()["operations"][0]["metadata"]["generatedFile"] == generated_name
    assert fetched.status_code == 200
    assert fetched.json()["mimeType"] == "image/png"
    assert downloaded.status_code == 200
    assert downloaded.content == b"fake-png"
    assert v1_listed.status_code == 200
    assert v1_listed.json()["generatedFiles"][0]["name"] == generated_name
    assert v1_operations.status_code == 200
    assert v1_operations.json()["operations"][0]["metadata"]["generatedFile"] == generated_name
    assert v1_operation.status_code == 200
    assert v1_operation.json()["metadata"]["generatedFile"] == generated_name
    assert v1_waited.status_code == 200
    assert v1_waited.json()["name"] == v1_operation.json()["name"]
    assert v1_cancelled.status_code == 200
    assert v1_cancelled.json() == {}
    assert v1_fetched.status_code == 200
    assert v1_fetched.json()["mimeType"] == "image/png"
    assert v1_downloaded.status_code == 200
    assert v1_downloaded.content == b"fake-png"
    assert v1_deleted.status_code == 200
    assert v1_missing.status_code == 404


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
