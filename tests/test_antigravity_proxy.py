import asyncio
import base64
import json
import io

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from google import genai
from google.genai import types
from starlette.routing import Match

import antigravity_proxy as proxy


GEMINI_V1BETA_DISCOVERY_REVISION = "20260626"

GEMINI_V1BETA_DISCOVERY_ROUTES_20260626 = (
    ("DELETE", "v1beta/batches/{batchesId}"),
    ("DELETE", "v1beta/cachedContents/{cachedContentsId}"),
    ("DELETE", "v1beta/corpora/{corporaId}"),
    ("DELETE", "v1beta/corpora/{corporaId}/permissions/{permissionsId}"),
    ("DELETE", "v1beta/fileSearchStores/{fileSearchStoresId}"),
    ("DELETE", "v1beta/fileSearchStores/{fileSearchStoresId}/documents/{documentsId}"),
    ("DELETE", "v1beta/files/{filesId}"),
    ("DELETE", "v1beta/tunedModels/{tunedModelsId}"),
    ("DELETE", "v1beta/tunedModels/{tunedModelsId}/permissions/{permissionsId}"),
    ("GET", "v1beta/batches"),
    ("GET", "v1beta/batches/{batchesId}"),
    ("GET", "v1beta/cachedContents"),
    ("GET", "v1beta/cachedContents/{cachedContentsId}"),
    ("GET", "v1beta/corpora"),
    ("GET", "v1beta/corpora/{corporaId}"),
    ("GET", "v1beta/corpora/{corporaId}/operations/{operationsId}"),
    ("GET", "v1beta/corpora/{corporaId}/permissions"),
    ("GET", "v1beta/corpora/{corporaId}/permissions/{permissionsId}"),
    ("GET", "v1beta/fileSearchStores"),
    ("GET", "v1beta/fileSearchStores/{fileSearchStoresId}"),
    ("GET", "v1beta/fileSearchStores/{fileSearchStoresId}/documents"),
    ("GET", "v1beta/fileSearchStores/{fileSearchStoresId}/documents/{documentsId}"),
    ("GET", "v1beta/fileSearchStores/{fileSearchStoresId}/operations/{operationsId}"),
    ("GET", "v1beta/fileSearchStores/{fileSearchStoresId}/upload/operations/{operationsId}"),
    ("GET", "v1beta/files"),
    ("GET", "v1beta/files/{filesId}"),
    ("GET", "v1beta/generatedFiles"),
    ("GET", "v1beta/generatedFiles/{generatedFilesId}/operations/{operationsId}"),
    ("GET", "v1beta/models"),
    ("GET", "v1beta/models/{modelsId}"),
    ("GET", "v1beta/models/{modelsId}/operations"),
    ("GET", "v1beta/models/{modelsId}/operations/{operationsId}"),
    ("GET", "v1beta/tunedModels"),
    ("GET", "v1beta/tunedModels/{tunedModelsId}"),
    ("GET", "v1beta/tunedModels/{tunedModelsId}/operations"),
    ("GET", "v1beta/tunedModels/{tunedModelsId}/operations/{operationsId}"),
    ("GET", "v1beta/tunedModels/{tunedModelsId}/permissions"),
    ("GET", "v1beta/tunedModels/{tunedModelsId}/permissions/{permissionsId}"),
    ("PATCH", "v1beta/batches/{batchesId}:updateEmbedContentBatch"),
    ("PATCH", "v1beta/batches/{batchesId}:updateGenerateContentBatch"),
    ("PATCH", "v1beta/cachedContents/{cachedContentsId}"),
    ("PATCH", "v1beta/corpora/{corporaId}/permissions/{permissionsId}"),
    ("PATCH", "v1beta/tunedModels/{tunedModelsId}"),
    ("PATCH", "v1beta/tunedModels/{tunedModelsId}/permissions/{permissionsId}"),
    ("POST", "v1beta/batches/{batchesId}:cancel"),
    ("POST", "v1beta/cachedContents"),
    ("POST", "v1beta/corpora"),
    ("POST", "v1beta/corpora/{corporaId}/permissions"),
    ("POST", "v1beta/dynamic/{dynamicId}:generateContent"),
    ("POST", "v1beta/dynamic/{dynamicId}:streamGenerateContent"),
    ("POST", "v1beta/fileSearchStores"),
    ("POST", "v1beta/fileSearchStores/{fileSearchStoresId}:importFile"),
    ("POST", "v1beta/fileSearchStores/{fileSearchStoresId}:uploadToFileSearchStore"),
    ("POST", "v1beta/files"),
    ("POST", "v1beta/files:register"),
    ("POST", "v1beta/models/{modelsId}:asyncBatchEmbedContent"),
    ("POST", "v1beta/models/{modelsId}:batchEmbedContents"),
    ("POST", "v1beta/models/{modelsId}:batchEmbedText"),
    ("POST", "v1beta/models/{modelsId}:batchGenerateContent"),
    ("POST", "v1beta/models/{modelsId}:countMessageTokens"),
    ("POST", "v1beta/models/{modelsId}:countTextTokens"),
    ("POST", "v1beta/models/{modelsId}:countTokens"),
    ("POST", "v1beta/models/{modelsId}:embedContent"),
    ("POST", "v1beta/models/{modelsId}:embedText"),
    ("POST", "v1beta/models/{modelsId}:generateAnswer"),
    ("POST", "v1beta/models/{modelsId}:generateContent"),
    ("POST", "v1beta/models/{modelsId}:generateMessage"),
    ("POST", "v1beta/models/{modelsId}:generateText"),
    ("POST", "v1beta/models/{modelsId}:predict"),
    ("POST", "v1beta/models/{modelsId}:predictLongRunning"),
    ("POST", "v1beta/models/{modelsId}:streamGenerateContent"),
    ("POST", "v1beta/tunedModels"),
    ("POST", "v1beta/tunedModels/{tunedModelsId}/permissions"),
    ("POST", "v1beta/tunedModels/{tunedModelsId}:asyncBatchEmbedContent"),
    ("POST", "v1beta/tunedModels/{tunedModelsId}:batchGenerateContent"),
    ("POST", "v1beta/tunedModels/{tunedModelsId}:generateContent"),
    ("POST", "v1beta/tunedModels/{tunedModelsId}:generateText"),
    ("POST", "v1beta/tunedModels/{tunedModelsId}:streamGenerateContent"),
    ("POST", "v1beta/tunedModels/{tunedModelsId}:transferOwnership"),
)


def _sample_discovery_path(flat_path):
    path = "/" + flat_path
    samples = {
        "batchesId": "batch-1",
        "cachedContentsId": "cache-1",
        "corporaId": "corpus-1",
        "documentsId": "documents/doc-1",
        "dynamicId": "gemini-3-flash-agent",
        "fileSearchStoresId": "store-1",
        "filesId": "file-1",
        "generatedFilesId": "gen-1",
        "modelsId": "gemini-3-flash-agent",
        "operationsId": "op-1",
        "permissionsId": "perm-1",
        "tunedModelsId": "tuned-1",
    }
    for key, value in samples.items():
        path = path.replace("{" + key + "}", value)
    return path


def _matched_route_paths(method, path):
    scope = {"type": "http", "method": method, "path": path, "root_path": "", "path_params": {}}
    return [
        route.path
        for route in proxy.app.routes
        if hasattr(route, "matches") and route.matches(scope)[0] is Match.FULL
    ]


def test_gemini_v1beta_discovery_20260626_flatpaths_match_fastapi_routes():
    assert GEMINI_V1BETA_DISCOVERY_REVISION == "20260626"
    missing = []
    for method, flat_path in GEMINI_V1BETA_DISCOVERY_ROUTES_20260626:
        path = _sample_discovery_path(flat_path)
        if not _matched_route_paths(method, path):
            missing.append(f"{method} {flat_path} -> {path}")

    assert missing == []


def test_vertex_model_aliases_cover_all_v1beta_model_post_methods():
    route_paths = {
        route.path
        for route in proxy.app.routes
        if "POST" in getattr(route, "methods", set())
    }
    model_suffixes = sorted(
        path.split("{model_name:path}", 1)[1]
        for path in route_paths
        if path.startswith("/v1beta/models/{model_name:path}:")
    )

    assert model_suffixes
    for suffix in model_suffixes:
        assert f"/v1beta/publishers/google/models/{{model_name:path}}{suffix}" in route_paths
        assert (
            f"/v1beta/projects/{{project}}/locations/{{location}}/publishers/google/models/{{model_name:path}}{suffix}"
            in route_paths
        )


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


def test_openai_compat_endpoints_are_removed():
    client = TestClient(proxy.app)

    chat = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    responses = client.post("/v1/responses", json={"model": "x", "input": "hi"})
    image = client.post("/v1/images/generations", json={"prompt": "draw"})

    for response in (chat, responses, image):
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["status"] == "NOT_FOUND"
        assert "OpenAI-compatible endpoints have been removed" in body["error"]["message"]


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
    paged_snake = client.get("/v1beta/models?page_size=1")
    stable_snake = client.get("/v1/models?page_size=1")
    too_many = client.get("/v1beta/models?pageSize=1001")

    assert paged.status_code == 200
    assert len(paged.json()["models"]) == 1
    assert paged.json()["nextPageToken"] == "1"
    assert next_page.status_code == 200
    assert len(next_page.json()["models"]) == 1
    assert paged_snake.status_code == 200
    assert len(paged_snake.json()["models"]) == 1
    assert paged_snake.json()["nextPageToken"] == "1"
    assert stable_snake.status_code == 200
    assert len(stable_snake.json()["models"]) == 1
    assert too_many.status_code == 400
    assert too_many.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert too_many.json()["error"]["details"][0]["@type"] == "type.googleapis.com/google.rpc.BadRequest"
    assert too_many.json()["error"]["details"][0]["fieldViolations"][0]["field"] == "pageSize"

    one = client.get("/v1beta/models/gemini-3-flash-agent")
    assert one.status_code == 200
    assert one.json()["name"] == "models/gemini-3-flash-agent"
    for resource_name in (
        "models/gemini-3-flash-agent",
        "publishers/google/models/gemini-3-flash-agent",
        "projects/proj/locations/global/publishers/google/models/gemini-3-flash-agent",
        "google/gemini-3-flash-agent",
    ):
        assert proxy._gemini_resource_model_id(resource_name) == "gemini-3-flash-agent"
        assert proxy._resolve_gemini_model(resource_name)["antigravity_model"] == "gemini-3-flash-agent"

    vertex_models = client.get("/v1beta/projects/proj/locations/global/publishers/google/models?pageSize=1")
    vertex_model = client.get(
        "/v1beta/projects/proj/locations/global/publishers/google/models/gemini-3-flash-agent"
    )
    assert vertex_models.status_code == 200
    assert len(vertex_models.json()["models"]) == 1
    assert vertex_model.status_code == 200
    assert vertex_model.json()["name"] == "models/gemini-3-flash-agent"

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
    counted_vertex = client.post(
        "/v1beta/projects/proj/locations/global/publishers/google/models/gemini-3-flash-agent:countTokens",
        json={"contents": "hello from vertex sdk"},
    )
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0
    assert counted.json()["promptTokensDetails"][0]["modality"] == "TEXT"
    assert counted.json()["cachedContentTokenCount"] == 0
    assert counted.json()["cacheTokensDetails"] == []
    assert counted_string.status_code == 200
    assert counted_string.json()["totalTokens"] > 0
    assert counted_string.json()["promptTokensDetails"][0]["modality"] == "TEXT"
    assert counted_vertex.status_code == 200
    assert counted_vertex.json()["totalTokens"] > 0

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

    counted_tools = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "contents": "use a tool",
        "tools": [{
            "function_declarations": [{
                "name": "lookup",
                "description": "Look up a record by query",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }]
        }],
        "tool_config": {"function_calling_config": {"mode": "auto"}},
    })
    assert counted_tools.status_code == 200
    assert counted_tools.json()["totalTokens"] > counted_string.json()["totalTokens"]


def test_gemini_compute_tokens_accepts_wrappers_and_media():
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:computeTokens", json={
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": "hello token world"}]},
                {"role": "model", "parts": [{"text": "reply"}, {"inline_data": {"mime_type": "image/png", "data": "aW1hZ2U="}}]},
            ],
        }
    })
    v1_response = client.post("/v1/models/gemini-3-flash-agent:computeTokens", json={
        "contents": "sdk string input",
    })
    vertex_response = client.post(
        "/v1beta/projects/proj/locations/global/publishers/google/models/gemini-3-flash-agent:computeTokens",
        json={"contents": "vertex token input"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["tokensInfo"]) == 2
    assert body["tokensInfo"][0]["role"] == "user"
    assert body["tokensInfo"][1]["role"] == "model"
    assert len(body["tokensInfo"][0]["tokenIds"]) == len(body["tokensInfo"][0]["tokens"])
    assert base64.b64decode(body["tokensInfo"][0]["tokens"][0]).decode("utf-8") == "hello"
    assert base64.b64decode(body["tokensInfo"][1]["tokens"][-1]).decode("utf-8") == "<inline:image/png>"
    assert v1_response.status_code == 200
    assert v1_response.json()["tokensInfo"][0]["role"] == "user"
    assert vertex_response.status_code == 200
    assert vertex_response.json()["tokensInfo"][0]["role"] == "user"
    assert "computeTokens" in client.get("/v1beta/models/gemini-3-flash-agent").json()["supportedGenerationMethods"]

    tools_response = client.post("/v1beta/models/gemini-3-flash-agent:computeTokens", json={
        "contents": "tool tokenization",
        "tools": [{"function_declarations": [{"name": "lookup", "description": "Lookup by query"}]}],
    })
    assert tools_response.status_code == 200
    assert tools_response.json()["tokensInfo"][0]["role"] == "system"


def test_gemini_count_tokens_applies_generate_content_request_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_CACHED_CONTENTS_DIR", str(tmp_path / "gemini_cached"))
    client = TestClient(proxy.app)

    created = client.post("/v1beta/cachedContents", json={
        "model": "models/gemini-3-flash-agent",
        "contents": [{"role": "user", "parts": [{"text": "cached context with several extra words"}]}],
    })
    assert created.status_code == 200
    cache_name = created.json()["name"]
    assert created.json()["usageMetadata"]["totalTokenCount"] > 0
    assert created.json()["usageMetadata"]["promptTokensDetails"][0]["modality"] == "TEXT"

    uncached = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "new prompt"}]}],
    })
    counted = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "generateContentRequest": {
            "cachedContent": cache_name,
            "contents": [{"role": "user", "parts": [{"text": "new prompt"}]}],
        }
    })
    counted_request_wrapper = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "request": {
            "cached_content": cache_name,
            "contents": "new prompt",
            "processing_options": {"media_resolution": "MEDIA_RESOLUTION_LOW"},
        }
    })
    counted_resource_object = client.post("/v1beta/models/gemini-3-flash-agent:countTokens", json={
        "generateContentRequest": {
            "cachedContent": {"name": cache_name},
            "contents": "new prompt",
        }
    })

    assert uncached.status_code == 200
    assert counted.status_code == 200
    assert counted_request_wrapper.status_code == 200
    assert counted_resource_object.status_code == 200
    assert counted.json()["totalTokens"] > uncached.json()["totalTokens"]
    assert counted.json()["cachedContentTokenCount"] > 0
    assert counted.json()["cacheTokensDetails"][0]["tokenCount"] == counted.json()["cachedContentTokenCount"]
    assert counted_request_wrapper.json()["cachedContentTokenCount"] == counted.json()["cachedContentTokenCount"]
    assert counted_request_wrapper.json()["totalTokens"] > uncached.json()["totalTokens"]
    assert counted_resource_object.json()["cachedContentTokenCount"] == counted.json()["cachedContentTokenCount"]


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
    vertex_predicted = client.post(
        "/v1beta/projects/proj/locations/us-central1/publishers/google/models/veo-3.1-generate-preview:predictLongRunning",
        json={"instances": [{"prompt": "make a vertex short clip"}]},
    )

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
    assert vertex_predicted.status_code == 200
    assert vertex_predicted.json()["metadata"]["model"] == "models/veo-3.1-generate-preview"
    assert vertex_predicted.json()["error"]["status"] == "UNIMPLEMENTED"


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
    flash_preview = client.get("/v1beta/models/gemini-3-flash-preview")
    flash_menu = client.get("/v1beta/models/gemini-3.5-flash")
    pro_preview = client.get("/v1beta/models/gemini-3.1-pro-preview")
    lite_latest = client.get("/v1beta/models/gemini-flash-lite-latest")

    assert fetched.status_code == 200
    assert fetched.json()["name"] == "models/gemini-3-flash-agent"
    assert generated.status_code == 200
    assert seen["model"] == "gemini-3-flash-agent"
    assert image.status_code == 200
    assert image.json()["name"] == "models/gemini-3.1-flash-image"
    assert flash_preview.status_code == 200
    assert flash_preview.json()["name"] == "models/gemini-3-flash-agent"
    assert flash_menu.status_code == 200
    assert flash_menu.json()["name"] == "models/gemini-3-flash-agent"
    assert pro_preview.status_code == 200
    assert pro_preview.json()["name"] == "models/gemini-pro-agent"
    assert lite_latest.status_code == 200
    assert lite_latest.json()["name"] == "models/gemini-3.1-flash-lite"


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


def test_gemini_google_host_prefixed_gateway_paths(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_PROXY_API_KEY", "secret")
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            seen["model"] = model
            return {"response": {"candidates": [{"content": {"parts": [{"text": "prefixed ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    listed = client.get(
        "/generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": "secret"},
    )
    fetched = client.get(
        "/generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-agent",
        headers={"x-goog-api-key": "secret"},
    )
    generated = client.post(
        "/generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-agent:generateContent",
        headers={"x-goog-api-key": "secret"},
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    alpha_generated = client.post(
        "/generativelanguage.googleapis.com/v1alpha/models/gemini-3-flash-agent:generateContent",
        headers={"x-goog-api-key": "secret"},
        json={"contents": [{"role": "user", "parts": [{"text": "preview hi"}]}]},
    )

    assert listed.status_code == 200
    assert "models" in listed.json()
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "models/gemini-3-flash-agent"
    assert generated.status_code == 200
    assert generated.json()["candidates"][0]["content"]["parts"][0]["text"] == "prefixed ok"
    assert alpha_generated.status_code == 200
    assert alpha_generated.json()["candidates"][0]["content"]["parts"][0]["text"] == "prefixed ok"
    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "preview hi"


def test_gemini_unmatched_routes_use_gemini_error_shape():
    client = TestClient(proxy.app)

    gemini_missing = client.get("/v1beta/not-a-route")
    removed_openai = client.post("/v1/chat/completions", json={"model": "x", "messages": []})

    assert gemini_missing.status_code == 404
    assert gemini_missing.json()["error"]["status"] == "NOT_FOUND"
    assert gemini_missing.json()["error"]["code"] == 404
    assert removed_openai.status_code == 404
    assert removed_openai.json()["error"]["status"] == "NOT_FOUND"


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
    gemini_models = client.get("/v1/models?pageSize=1")
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
    assert "models" in openai_models.json()
    assert "data" not in openai_models.json()
    assert gemini_models.status_code == 200
    assert len(gemini_models.json()["models"]) == 1
    assert gemini_models.json()["models"][0]["name"].startswith("models/")
    assert gemini_models.json()["nextPageToken"] == "1"
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


def test_gemini_v1alpha_gateway_aliases_to_v1beta(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "files"))
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            seen["model"] = model
            return {"response": {"candidates": [{"content": {"parts": [{"text": "alpha ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    listed = client.get("/v1alpha/models?page_size=1")
    generated = client.post("/v1alpha/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "alpha direct"}]}],
    })
    counted = client.post("/v1alpha/models/gemini-3-flash-agent:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "alpha count"}]}],
    })
    uploaded = client.post(
        "/upload/v1alpha/files?upload_type=media&display_name=alpha-note.txt",
        content=b"alpha upload",
        headers={"Content-Type": "text/plain"},
    )

    assert listed.status_code == 200
    assert len(listed.json()["models"]) == 1
    assert generated.status_code == 200
    assert generated.json()["candidates"][0]["content"]["parts"][0]["text"] == "alpha ok"
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0
    assert uploaded.status_code == 200
    assert uploaded.json()["file"]["displayName"] == "alpha-note.txt"
    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "alpha direct"


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
    vertex = client.post(
        "/v1beta/projects/proj/locations/global/publishers/google/models/gemini-3-flash-agent:embedContent",
        json=payload,
    )
    vertex_batch = client.post(
        "/v1beta/projects/proj/locations/global/publishers/google/models/gemini-3-flash-agent:batchEmbedContents",
        json={
            "requests": [
                {"content": {"parts": [{"text": "vertex one"}]}, "outputDimensionality": 12},
                {"content": {"parts": [{"text": "vertex two"}]}, "outputDimensionality": 12},
            ]
        },
    )

    assert first.status_code == 200
    values = first.json()["embedding"]["values"]
    assert len(values) == 32
    assert values == second.json()["embedding"]["values"]
    assert first.json()["usageMetadata"]["promptTokenCount"] > 0
    assert first.json()["usageMetadata"]["promptTokenDetails"][0]["modality"] == "TEXT"
    assert batch.status_code == 200
    assert len(batch.json()["embeddings"]) == 2
    assert len(batch.json()["embeddings"][0]["values"]) == 16
    assert batch.json()["usageMetadata"]["promptTokenCount"] > first.json()["usageMetadata"]["promptTokenCount"]
    assert batch.json()["usageMetadata"]["promptTokenDetails"][0]["tokenCount"] == batch.json()["usageMetadata"]["promptTokenCount"]
    assert vertex.status_code == 200
    assert len(vertex.json()["embedding"]["values"]) == 32
    assert vertex_batch.status_code == 200
    assert len(vertex_batch.json()["embeddings"]) == 2
    assert len(vertex_batch.json()["embeddings"][0]["values"]) == 12


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
            "output_dimensionality": "14",
            "task_type": "retrieval document",
            "title": "SDK Config",
            "auto_truncate": "false",
            "document_ocr": "true",
            "audio_track_extraction": "0",
        },
    })
    wrapped = client.post("/v1/models/gemini-3-flash-agent:batchEmbedContents", json={
        "requests": [{
            "contents": [{"parts": [{"text": "gamma"}]}, {"parts": [{"text": "delta"}]}],
            "embed_content_config": {"output_dimensionality": "11", "task_type": "code retrieval query"},
        }]
    })
    request_wrapped = client.post("/v1beta/models/gemini-3-flash-agent:embedContent", json={
        "request": {
            "content": {"parts": [{"text": "provider wrapped"}]},
        },
        "provider_options": {
            "google": {
                "output_dimensionality": "9",
                "task_type": "retrieval query",
            }
        },
    })
    embedding_configured = client.post("/v1beta/models/gemini-3-flash-agent:embedContent", json={
        "content": {"parts": [{"text": "embedding config alias"}]},
        "embedding_config": {
            "output_dimensionality": "13",
            "task_type": "semantic similarity",
        },
    })
    embed_content_wrapped = client.post("/v1beta/models/gemini-3-flash-agent:batchEmbedContents", json={
        "requests": [{
            "embed_content_request": {
                "content": {"parts": [{"text": "batch provider wrapped"}]},
            },
            "provider_options": {
                "google": {
                    "output_dimensionality": "7",
                    "task_type": "classification",
                }
            },
        }]
    })
    batch_embedding_configured = client.post("/v1beta/models/gemini-3-flash-agent:batchEmbedContents", json={
        "embedding_config": {"output_dimensionality": "6"},
        "requests": [{"content": {"parts": [{"text": "shared embedding config"}]}}],
    })

    assert configured.status_code == 200
    assert len(configured.json()["embeddings"]) == 2
    assert len(configured.json()["embeddings"][0]["values"]) == 14
    assert configured.json()["embedding"] == configured.json()["embeddings"][0]
    assert configured.json()["embeddings"][0]["values"] != configured.json()["embeddings"][1]["values"]
    assert wrapped.status_code == 200
    assert len(wrapped.json()["embeddings"]) == 2
    assert len(wrapped.json()["embeddings"][0]["values"]) == 11
    assert request_wrapped.status_code == 200
    assert len(request_wrapped.json()["embedding"]["values"]) == 9
    assert embedding_configured.status_code == 200
    assert len(embedding_configured.json()["embedding"]["values"]) == 13
    assert embed_content_wrapped.status_code == 200
    assert len(embed_content_wrapped.json()["embeddings"][0]["values"]) == 7
    assert batch_embedding_configured.status_code == 200
    assert len(batch_embedding_configured.json()["embeddings"][0]["values"]) == 6


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
    wrapped_created = client.post("/v1/models/gemini-3-flash-agent:asyncBatchEmbedContent", json={
        "embed_content_batch": {
            "display_name": "wrapped embed job",
            "requests": [
                {"embed_content_request": {"content": {"parts": [{"text": "wrapped alpha"}]}}},
                {"embed_content_request": {"content": {"parts": [{"text": "wrapped beta"}]}}},
            ],
        },
        "config": {"output_dimensionality": "6"},
        "priority": "HIGH",
    })

    assert created.status_code == 200
    operation = created.json()
    assert operation["done"] is True
    assert operation["response"]["embeddings"][0]["values"]
    assert operation["metadata"]["batchResource"]["displayName"] == "embed job"
    assert operation["metadata"]["state"] == "BATCH_STATE_SUCCEEDED"
    assert operation["metadata"]["operation"] == operation["name"]
    assert wrapped_created.status_code == 200
    wrapped_operation = wrapped_created.json()
    assert wrapped_operation["metadata"]["batchResource"]["displayName"] == "wrapped embed job"
    assert wrapped_operation["metadata"]["batchResource"]["priority"] == "HIGH"
    assert len(wrapped_operation["response"]["embeddings"]) == 2
    assert len(wrapped_operation["response"]["embeddings"][0]["values"]) == 6

    batch_name = operation["metadata"]["batch"]
    updated = client.patch(f"/v1beta/{batch_name}:updateEmbedContentBatch", json={"displayName": "renamed"})
    fetched = client.get(f"/v1beta/{batch_name}")
    filtered_operations = client.get(
        "/v1beta/operations",
        params={"filter": 'metadata.state="BATCH_STATE_SUCCEEDED" AND metadata.display_name="embed job"'},
    )

    assert updated.status_code == 200
    assert updated.json()["name"] == batch_name
    assert updated.json()["metadata"]["batchResource"]["displayName"] == "renamed"
    assert fetched.json()["name"] == batch_name
    assert fetched.json()["metadata"]["operation"] == operation["name"]
    assert fetched.json()["metadata"]["batchResource"]["displayName"] == "renamed"
    assert fetched.json()["metadata"]["batchStats"]["requestCount"] == "2"
    assert filtered_operations.status_code == 200
    assert operation["name"] in {item["name"] for item in filtered_operations.json()["operations"]}


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

    vertex_response = client.post(
        "/v1beta/projects/proj/locations/global/publishers/google/models/gemini-3-flash-agent:generateContent",
        json={"contents": "hello vertex"},
    )
    assert vertex_response.status_code == 200
    assert vertex_response.json()["candidates"][0]["content"]["parts"][0]["text"] == "hello"


def test_gemini_generate_content_normalizes_response_usage_and_content(monkeypatch):
    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {
                "response": {
                    "model_version": "gemini-upstream-version",
                    "response_id": "upstream-response-id",
                    "model_status": {"model_stage": "preview"},
                    "prompt_feedback": {
                        "block_reason": "SAFETY",
                        "block_reason_message": "blocked by policy",
                        "safety_ratings": [{"category": "dangerous", "probability": "medium", "blocked": "true"}],
                    },
                    "candidates": [{
                        "content": {"parts": "hello"},
                        "finish_reason": "max tokens",
                        "safety_ratings": [{"category": "harassment", "probability": "low", "blocked": "false"}],
                        "citation_metadata": {
                            "citation_sources": [{"start_index": 0, "end_index": 5, "uri": "https://example.test"}],
                        },
                        "grounding_metadata": {
                            "search_entry_point": {"rendered_content": "x", "sdk_blob": "blob"},
                            "grounding_chunks": [
                                {"retrieved_context": {"media_id": "media-1", "page_number": 3, "text": "retrieved"}},
                                {"image": {"image_uri": "https://example.test/image.png", "source_uri": "https://example.test"}},
                                {"maps": {"place_id": "place-1", "place_answer_sources": {"review_snippets": ["good"]}}},
                            ],
                            "grounding_supports": [{
                                "segment": {"part_index": 0, "start_index": 0, "end_index": 5, "text": "hello"},
                                "grounding_chunk_indices": [0, 1],
                                "confidence_scores": [0.9, 0.8],
                            }],
                            "web_search_queries": ["atlas"],
                            "image_search_queries": ["atlas image"],
                            "google_maps_widget_context_token": "maps-token",
                            "retrieval_metadata": {"google_search_dynamic_retrieval_score": 0.42},
                        },
                        "grounding_attributions": [{
                            "source_id": {
                                "grounding_passage": {"passage_id": "passage-1", "part_index": 0},
                                "semantic_retriever_chunk": {"source": "corpora/demo", "chunk": "chunks/1"},
                            },
                            "content": {"role": "model", "parts": [{"text": "attributed"}]},
                        }],
                        "url_context_metadata": {
                            "url_metadata": [{
                                "retrieved_url": "https://example.test",
                                "url_retrieval_status": "success",
                            }],
                        },
                        "logprobs_result": {
                            "log_probability_sum": -1.5,
                            "top_candidates": [{"candidates": [{"token": "hello", "token_id": 42, "log_probability": -0.1}]}],
                            "chosen_candidates": [{"token": "hello", "token_id": 42, "log_probability": -0.1}],
                        },
                        "avg_logprobs": -0.2,
                    }],
                    "usage_metadata": {
                        "prompt_token_count": 4,
                        "candidates_token_count": 2,
                        "total_token_count": 14,
                        "tool_use_prompt_tokens": 3,
                        "thoughts_tokens": 5,
                        "prompt_tokens_details": [{"modality": "TEXT", "token_count": 4}],
                        "tool_use_prompt_tokens_details": [{"modality": "TEXT", "token_count": 3}],
                        "thoughts_tokens_details": [{"modality": "TEXT", "token_count": 5}],
                        "cache_tokens_details": [{"modality": "document", "token_count": 1}],
                        "service_tier": "SERVICE_TIER_PRIORITY",
                        "traffic_type": "ON_DEMAND",
                    },
                }
            }

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={"contents": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["modelVersion"] == "gemini-upstream-version"
    assert body["responseId"] == "upstream-response-id"
    assert body["modelStatus"]["modelStage"] == "PREVIEW"
    assert body["promptFeedback"]["blockReason"] == "SAFETY"
    assert body["promptFeedback"]["blockReasonMessage"] == "blocked by policy"
    assert body["promptFeedback"]["safetyRatings"][0]["category"] == "HARM_CATEGORY_DANGEROUS_CONTENT"
    assert body["promptFeedback"]["safetyRatings"][0]["probability"] == "MEDIUM"
    assert body["promptFeedback"]["safetyRatings"][0]["blocked"] is True
    assert "prompt_feedback" not in body
    assert body["candidates"][0]["content"] == {"role": "model", "parts": [{"text": "hello"}]}
    assert body["candidates"][0]["finishReason"] == "MAX_TOKENS"
    assert "finish_reason" not in body["candidates"][0]
    assert body["candidates"][0]["safetyRatings"][0]["category"] == "HARM_CATEGORY_HARASSMENT"
    assert body["candidates"][0]["safetyRatings"][0]["probability"] == "LOW"
    assert body["candidates"][0]["safetyRatings"][0]["blocked"] is False
    assert body["candidates"][0]["groundingMetadata"]["searchEntryPoint"]["renderedContent"] == "x"
    assert body["candidates"][0]["groundingMetadata"]["searchEntryPoint"]["sdkBlob"] == "blob"
    assert body["candidates"][0]["groundingMetadata"]["webSearchQueries"] == ["atlas"]
    assert body["candidates"][0]["groundingMetadata"]["imageSearchQueries"] == ["atlas image"]
    assert body["candidates"][0]["groundingMetadata"]["googleMapsWidgetContextToken"] == "maps-token"
    assert body["candidates"][0]["groundingMetadata"]["retrievalMetadata"]["googleSearchDynamicRetrievalScore"] == 0.42
    assert body["candidates"][0]["groundingMetadata"]["groundingChunks"][0]["retrievedContext"]["mediaId"] == "media-1"
    assert body["candidates"][0]["groundingMetadata"]["groundingChunks"][0]["retrievedContext"]["pageNumber"] == 3
    assert body["candidates"][0]["groundingMetadata"]["groundingChunks"][1]["image"]["imageUri"] == "https://example.test/image.png"
    assert body["candidates"][0]["groundingMetadata"]["groundingChunks"][1]["image"]["sourceUri"] == "https://example.test"
    assert body["candidates"][0]["groundingMetadata"]["groundingChunks"][2]["maps"]["placeId"] == "place-1"
    assert body["candidates"][0]["groundingMetadata"]["groundingChunks"][2]["maps"]["placeAnswerSources"]["reviewSnippets"] == ["good"]
    assert body["candidates"][0]["groundingMetadata"]["groundingSupports"][0]["segment"]["partIndex"] == 0
    assert body["candidates"][0]["groundingMetadata"]["groundingSupports"][0]["groundingChunkIndices"] == [0, 1]
    assert body["candidates"][0]["groundingMetadata"]["groundingSupports"][0]["confidenceScores"] == [0.9, 0.8]
    assert body["candidates"][0]["groundingAttributions"][0]["sourceId"]["groundingPassage"]["passageId"] == "passage-1"
    assert body["candidates"][0]["groundingAttributions"][0]["sourceId"]["groundingPassage"]["partIndex"] == 0
    assert body["candidates"][0]["groundingAttributions"][0]["sourceId"]["semanticRetrieverChunk"]["chunk"] == "chunks/1"
    assert body["candidates"][0]["citationMetadata"]["citationSources"][0]["startIndex"] == 0
    assert body["candidates"][0]["citationMetadata"]["citationSources"][0]["endIndex"] == 5
    assert body["candidates"][0]["urlContextMetadata"]["urlMetadata"][0]["retrievedUrl"] == "https://example.test"
    assert body["candidates"][0]["urlContextMetadata"]["urlMetadata"][0]["urlRetrievalStatus"] == "URL_RETRIEVAL_STATUS_SUCCESS"
    assert body["candidates"][0]["logprobsResult"]["logProbabilitySum"] == -1.5
    assert body["candidates"][0]["logprobsResult"]["topCandidates"][0]["candidates"][0]["logProbability"] == -0.1
    assert body["candidates"][0]["logprobsResult"]["topCandidates"][0]["candidates"][0]["tokenId"] == 42
    assert body["candidates"][0]["logprobsResult"]["chosenCandidates"][0]["tokenId"] == 42
    assert body["candidates"][0]["avgLogprobs"] == -0.2
    assert body["usageMetadata"]["promptTokenCount"] == 4
    assert body["usageMetadata"]["candidatesTokenCount"] == 2
    assert body["usageMetadata"]["toolUsePromptTokenCount"] == 3
    assert body["usageMetadata"]["thoughtsTokenCount"] == 5
    assert body["usageMetadata"]["promptTokensDetails"][0]["modality"] == "TEXT"
    assert body["usageMetadata"]["promptTokensDetails"][0]["tokenCount"] == 4
    assert body["usageMetadata"]["toolUsePromptTokensDetails"][0]["tokenCount"] == 3
    assert body["usageMetadata"]["thoughtsTokensDetails"][0]["tokenCount"] == 5
    assert body["usageMetadata"]["cacheTokensDetails"][0]["modality"] == "DOCUMENT"
    assert body["usageMetadata"]["cacheTokensDetails"][0]["tokenCount"] == 1
    assert body["usageMetadata"]["serviceTier"] == "priority"
    assert body["usageMetadata"]["trafficType"] == "ON_DEMAND"
    assert body["usageMetadata"]["totalTokenCount"] == 14
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
                {
                    "file_data": {"mime_type": "video/mp4", "file_uri": "files/video-1"},
                    "video_metadata": {"start_offset": "1s", "end_offset": "3s", "fps": "24"},
                },
                {"text": "describe bytes"},
            ],
        }]
    })
    assert bytes_part.status_code == 200
    assert seen["request"]["contents"][0]["parts"][0]["inlineData"] == {
        "mimeType": "image/png",
        "data": "aW1hZ2U=",
    }
    assert seen["request"]["contents"][0]["parts"][1]["fileData"] == {
        "mimeType": "video/mp4",
        "fileUri": "files/video-1",
    }
    assert seen["request"]["contents"][0]["parts"][1]["videoMetadata"] == {
        "startOffset": "1s",
        "endOffset": "3s",
        "fps": 24.0,
    }
    assert seen["request"]["contents"][0]["parts"][2]["text"] == "describe bytes"

    part_aliases = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{
            "role": "model",
            "parts": [
                {
                    "function_call": {"id": "fn-1", "name": "lookup", "args": {"q": "atlas"}},
                    "thought_signature": "sig-123",
                    "thought": True,
                    "part_metadata": {"source": "test"},
                },
                {
                    "function_response": {
                        "id": "fn-1",
                        "name": "lookup",
                        "response": {"answer": "October"},
                        "will_continue": "false",
                        "scheduling": "when idle",
                    }
                },
                {"tool_call": {"id": "tool-1", "tool_type": "web", "args": {"query": "atlas"}}},
                {"tool_response": {"id": "tool-1", "tool_type": "image_search", "response": {"ok": True}}},
                {"executable_code": {"id": "code-1", "language": "python", "code": "print(1)"}},
                {"code_execution_result": {"id": "code-1", "outcome": "ok", "output": "1"}},
                {"inline_data": {"mime_type": "image/png", "data": "AA=="}, "media_resolution": "high"},
            ],
        }]
    })
    assert part_aliases.status_code == 200
    alias_parts = seen["request"]["contents"][0]["parts"]
    assert alias_parts[0] == {
        "functionCall": {"id": "fn-1", "name": "lookup", "args": {"q": "atlas"}},
        "thoughtSignature": "sig-123",
        "thought": True,
        "partMetadata": {"source": "test"},
    }
    assert alias_parts[1] == {
        "functionResponse": {
            "id": "fn-1",
            "name": "lookup",
            "response": {"answer": "October"},
            "willContinue": False,
            "scheduling": "WHEN_IDLE",
        }
    }
    assert alias_parts[2] == {"toolCall": {"id": "tool-1", "toolType": "GOOGLE_SEARCH_WEB", "args": {"query": "atlas"}}}
    assert alias_parts[3] == {"toolResponse": {"id": "tool-1", "toolType": "GOOGLE_SEARCH_IMAGE", "response": {"ok": True}}}
    assert alias_parts[4] == {"executableCode": {"id": "code-1", "language": "PYTHON", "code": "print(1)"}}
    assert alias_parts[5] == {"codeExecutionResult": {"id": "code-1", "outcome": "OUTCOME_OK", "output": "1"}}
    assert alias_parts[6] == {
        "inlineData": {"mimeType": "image/png", "data": "AA=="},
        "mediaResolution": "MEDIA_RESOLUTION_HIGH",
    }


def test_gemini_candidate_part_aliases_are_canonicalized(monkeypatch):
    class FakeClient:
        def generate_raw(self, *, request, model=""):
            return {
                "response": {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "function_call": {"name": "lookup", "args": {"q": "atlas"}},
                                "thought_signature": "sig-456",
                            }]
                        }
                    }]
                }
            }

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={"contents": "hi"})

    assert response.status_code == 200
    part = response.json()["candidates"][0]["content"]["parts"][0]
    assert part == {
        "functionCall": {"name": "lookup", "args": {"q": "atlas"}},
        "thoughtSignature": "sig-456",
    }


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
            "max_output_tokens": "17",
            "candidate_count": "1",
            "temperature": "0.2",
            "top_p": "0.9",
            "top_k": "40",
            "response_logprobs": "true",
            "stop_sequences": "END",
            "response_modalities": "text",
            "media_resolution": "low",
            "audio_timestamp": "true",
            "translation_config": {
                "target_language_code": "ko",
                "echo_target_language": "false",
            },
            "speech_config": {
                "language_code": "ko-KR",
                "multi_speaker_voice_config": {
                    "speaker_voice_configs": [{
                        "speaker": "narrator",
                        "voice_config": {
                            "prebuilt_voice_config": {"voice_name": "Puck"},
                        },
                    }],
                },
            },
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string", "min_length": 1}},
                "required": ["answer"],
                "property_ordering": ["answer"],
                "any_of": [{"type": "object", "properties": {"answer": {"type": "string"}}}],
            },
            "response_json_schema": {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 0}},
                "property_ordering": ["score"],
            },
            "response_format": {
                "text": {"mime_type": "application/json"},
                "image": {
                    "mime_type": "image/jpeg",
                    "delivery": "inline",
                    "aspect_ratio": "16:9",
                    "image_size": "1k",
                },
                "audio": {
                    "mime_type": "audio/wav",
                    "delivery": "uri",
                    "sample_rate": "24000",
                    "bit_rate": "64000",
                },
            },
            "image_config": {
                "aspect_ratio": "4:3",
                "image_size": "1k",
            },
            "enable_enhanced_civic_answers": "true",
            "tool_config": {"function_calling_config": {"mode": "none"}},
            "labels": {"source": "sdk"},
            "service_tier": "SERVICE_TIER_PRIORITY",
            "store": "false",
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
    assert seen["request"]["serviceTier"] == "priority"
    assert seen["request"]["store"] is False
    assert seen["request"]["toolConfig"]["functionCallingConfig"] == {"mode": "NONE"}
    assert seen["request"]["generationConfig"]["maxOutputTokens"] == 17
    assert seen["request"]["generationConfig"]["candidateCount"] == 1
    assert seen["request"]["generationConfig"]["temperature"] == 0.2
    assert seen["request"]["generationConfig"]["topP"] == 0.9
    assert seen["request"]["generationConfig"]["topK"] == 40
    assert seen["request"]["generationConfig"]["responseLogprobs"] is True
    assert seen["request"]["generationConfig"]["enableEnhancedCivicAnswers"] is True
    assert seen["request"]["generationConfig"]["stopSequences"] == ["END"]
    assert seen["request"]["generationConfig"]["responseModalities"] == ["TEXT"]
    assert seen["request"]["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_LOW"
    assert seen["request"]["generationConfig"]["audioTimestamp"] is True
    assert seen["request"]["generationConfig"]["responseFormat"] == {
        "text": {"mimeType": "APPLICATION_JSON"},
        "image": {
            "mimeType": "IMAGE_JPEG",
            "delivery": "INLINE",
            "aspectRatio": "ASPECT_RATIO_SIXTEEN_BY_NINE",
            "imageSize": "IMAGE_SIZE_ONE_K",
        },
        "audio": {
            "mimeType": "AUDIO_WAV",
            "delivery": "URI",
            "sampleRate": 24000,
            "bitRate": 64000,
        },
    }
    assert seen["request"]["generationConfig"]["imageConfig"] == {
        "aspectRatio": "4:3",
        "imageSize": "1K",
    }
    assert seen["request"]["generationConfig"]["translationConfig"] == {
        "targetLanguageCode": "ko",
        "echoTargetLanguage": False,
    }
    assert seen["request"]["generationConfig"]["speechConfig"] == {
        "languageCode": "ko-KR",
        "multiSpeakerVoiceConfig": {
            "speakerVoiceConfigs": [{
                "speaker": "narrator",
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": "Puck"},
                },
            }],
        },
    }
    invalid_speech = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "generation_config": {
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}},
                "multi_speaker_voice_config": {
                    "speaker_voice_configs": [{
                        "speaker": "narrator",
                        "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}},
                    }],
                },
            },
        },
    })
    assert invalid_speech.status_code == 400
    assert invalid_speech.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert "mutually exclusive" in invalid_speech.json()["error"]["message"]
    assert seen["request"]["generationConfig"]["responseMimeType"] == "application/json"
    schema = seen["request"]["generationConfig"]["responseSchema"]
    assert schema["type"] == "object"
    assert schema["properties"]["answer"]["minLength"] == 1
    assert schema["propertyOrdering"] == ["answer"]
    assert schema["anyOf"] == [{"type": "object", "properties": {"answer": {"type": "string"}}}]
    json_schema = seen["request"]["generationConfig"]["responseJsonSchema"]
    assert json_schema["properties"]["score"]["minimum"] == 0
    assert json_schema["propertyOrdering"] == ["score"]


def test_gemini_generate_content_maps_tool_choice_to_tool_config(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "weather?",
        "tools": [{
            "function_declaration": {
                "name": "get_weather",
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    })

    assert response.status_code == 200
    assert seen["request"]["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    assert seen["request"]["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["get_weather"],
    }


def test_gemini_dynamic_generate_and_stream_routes(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen.setdefault("requests", []).append(request)
            seen.setdefault("models", []).append(model)
            return {"response": {"candidates": [{"content": {"parts": [{"text": "dynamic ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    generated = client.post("/v1beta/dynamic/gemini-3-flash-agent:generateContent", json={
        "contents": "hello dynamic",
    })
    with client.stream("POST", "/v1/dynamic/gemini-3-flash-agent:streamGenerateContent", json={
        "contents": "hello dynamic stream",
    }) as streamed:
        stream_body = streamed.read().decode()

    assert generated.status_code == 200
    assert generated.json()["candidates"][0]["content"]["parts"][0]["text"] == "dynamic ok"
    assert streamed.status_code == 200
    assert "data:" in stream_body
    assert seen["models"] == ["gemini-3-flash-agent", "gemini-3-flash-agent"]
    assert seen["requests"][0]["contents"][0]["parts"][0]["text"] == "hello dynamic"


def test_gemini_function_calling_config_accepts_required_alias(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "weather?",
        "tool_config": {
            "function_calling_config": {
                "mode": "required",
                "allowed_function_names": "get_weather",
            },
        },
    })

    assert response.status_code == 200
    assert seen["request"]["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["get_weather"],
    }


def test_gemini_generation_config_accepts_nested_response_format(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "generation_config": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string", "min_length": 1}},
                    },
                    "_responseJsonSchema": {
                        "type": "object",
                        "properties": {"score": {"type": "integer", "minimum": 0}},
                    },
                },
            },
        },
    })

    assert response.status_code == 200
    gen = seen["request"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseSchema"]["properties"]["answer"]["minLength"] == 1
    assert gen["responseJsonSchema"]["properties"]["score"]["minimum"] == 0

    official_response_format = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "generation_config": {
            "response_format": {
                "text": {
                    "mime_type": "text/plain",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string", "min_length": 1}},
                    },
                },
                "image": {
                    "mime_type": "image/png",
                    "aspect_ratio": "16:9",
                    "image_size": "1K",
                    "delivery": "inline",
                },
                "audio": {
                    "mime_type": "audio/wav",
                    "sample_rate": "24000",
                    "bit_rate": "128000",
                    "delivery": "stream",
                },
            },
        },
    })

    assert official_response_format.status_code == 200
    official_gen = seen["request"]["generationConfig"]
    assert official_gen["responseMimeType"] == "text/plain"
    assert official_gen["responseSchema"]["properties"]["answer"]["minLength"] == 1
    assert official_gen["responseFormat"]["text"]["schema"]["properties"]["answer"]["minLength"] == 1
    assert official_gen["responseFormat"]["image"] == {
        "mimeType": "image/png",
        "aspectRatio": "ASPECT_RATIO_SIXTEEN_BY_NINE",
        "imageSize": "IMAGE_SIZE_ONE_K",
        "delivery": "INLINE",
    }
    assert official_gen["responseFormat"]["audio"] == {
        "mimeType": "AUDIO_WAV",
        "sampleRate": 24000,
        "bitRate": 128000,
        "delivery": "stream",
    }


def test_gemini_generate_content_accepts_provider_google_options(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "provider_options": {
            "google": {
                "system_instruction": "from provider",
                "max_output_tokens": "12",
                "thinking_config": {"thinking_budget": "64", "include_thoughts": "true"},
                "tool_config": {"function_calling_config": {"mode": "validated"}},
                "service_tier": "FLEX",
            }
        },
        "config": {
            "max_output_tokens": "18",
            "response_mime_type": "text/plain",
        },
    })

    assert response.status_code == 200
    assert "providerOptions" not in seen["request"]
    assert "google" not in seen["request"]
    assert seen["request"]["systemInstruction"] == {"role": "system", "parts": [{"text": "from provider"}]}
    assert seen["request"]["serviceTier"] == "flex"
    assert seen["request"]["toolConfig"]["functionCallingConfig"] == {"mode": "VALIDATED"}
    assert seen["request"]["generationConfig"]["maxOutputTokens"] == 18
    assert seen["request"]["generationConfig"]["responseMimeType"] == "text/plain"
    assert seen["request"]["generationConfig"]["thinkingConfig"] == {
        "thinkingBudget": 64,
        "includeThoughts": True,
    }


def test_gemini_generate_content_accepts_processing_options_without_forwarding(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "summarize video",
        "processing_options": {
            "media_resolution": "MEDIA_RESOLUTION_LOW",
            "start_offset": "1s",
            "end_offset": "3s",
        },
        "provider_options": {
            "google": {
                "processing_options": {
                    "media_resolution": "MEDIA_RESOLUTION_HIGH",
                }
            }
        },
    })

    assert response.status_code == 200
    assert "processingOptions" not in seen["request"]
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "summarize video"


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

    legacy_search = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "tools": [{
            "google_search_retrieval": {
                "dynamic_retrieval_config": {
                    "mode": "dynamic",
                    "dynamic_threshold": "0.35",
                }
            }
        }],
    })
    assert legacy_search.status_code == 200
    assert seen["request"]["tools"] == [{
        "google_search": {
            "dynamicRetrievalConfig": {
                "mode": "MODE_DYNAMIC",
                "dynamicThreshold": 0.35,
            }
        }
    }]

    google_search_options = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "tools": {
            "google_search": {
                "search_types": {"web_search": {}, "image_search": {}},
                "time_range_filter": {
                    "start_time": "2026-01-01T00:00:00Z",
                    "end_time": "2026-01-02T00:00:00Z",
                },
            }
        },
    })
    assert google_search_options.status_code == 200
    assert seen["request"]["tools"] == [{
        "google_search": {
            "searchTypes": {"webSearch": {}, "imageSearch": {}},
            "timeRangeFilter": {
                "startTime": "2026-01-01T00:00:00Z",
                "endTime": "2026-01-02T00:00:00Z",
            },
        }
    }]
    assert proxy._gemini_normalize_tools_value({
        "file_search": {
            "file_search_store_names": ["fileSearchStores/local"],
            "metadata_filter": 'document.custom_metadata.project="atlas"',
            "top_k": "3",
        }
    }) == [{
        "file_search": {
            "fileSearchStoreNames": ["fileSearchStores/local"],
            "metadataFilter": 'document.custom_metadata.project="atlas"',
            "topK": 3,
        }
    }]
    assert proxy._gemini_normalize_tools_value({
        "computer_use": {
            "environment": "browser",
            "disabled_safety_policies": [
                "financial_transactions",
                "sensitive_data_modification",
                "legal_terms_and_agreements",
            ],
            "enable_prompt_injection_detection": "true",
        }
    }) == [{
        "computerUse": {
            "environment": "ENVIRONMENT_BROWSER",
            "disabledSafetyPolicies": [
                "FINANCIAL_TRANSACTIONS",
                "SENSITIVE_DATA_MODIFICATION",
                "LEGAL_TERMS_AND_AGREEMENTS",
            ],
            "enablePromptInjectionDetection": True,
        }
    }]
    assert proxy._gemini_normalize_tools_value({
        "type": "computer_use",
        "environment": "browser",
    }) == [{
        "computerUse": {
            "environment": "ENVIRONMENT_BROWSER",
        }
    }]
    assert proxy._gemini_url_retrieval_status_value("paywall") == "URL_RETRIEVAL_STATUS_PAYWALL"
    assert proxy._gemini_url_retrieval_status_value("unsafe") == "URL_RETRIEVAL_STATUS_UNSAFE"
    assert proxy._gemini_model_stage_value("unstable") == "UNSTABLE_EXPERIMENTAL"
    assert proxy._gemini_model_stage_value("legacy") == "LEGACY"

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

    single_function_declaration = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "function_declaration": {
            "name": "single_lookup",
            "parameters_json_schema": {"type": "object"},
        },
    })

    assert single_function_declaration.status_code == 200
    assert seen["request"]["tools"] == [{
        "functionDeclarations": [{
            "name": "single_lookup",
            "parameters": {"type": "object", "properties": {}},
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
                "behavior": "non_blocking",
            }
        },
    })

    assert schema_aliases.status_code == 200
    declaration = seen["request"]["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == "typed_lookup"
    assert declaration["parameters"]["properties"]["query"]["type"] == "string"
    assert declaration["response"]["properties"]["answer"]["type"] == "string"
    assert declaration["behavior"] == "NON_BLOCKING"
    assert "parametersJsonSchema" not in declaration
    assert "responseJsonSchema" not in declaration

    single_tool_declaration = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "hi",
        "tools": {"function_declaration": {"name": "tool_lookup", "description": "Lookup", "behavior": "blocking"}},
    })

    assert single_tool_declaration.status_code == 200
    assert seen["request"]["tools"] == [{
        "functionDeclarations": [{
            "name": "tool_lookup",
            "description": "Lookup",
            "behavior": "BLOCKING",
        }]
    }]


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
        "generation_config": {
            "thinking_config": {
                "thinking_budget": "128",
                "include_thoughts": "true",
                "thinking_level": "thinking level high",
            },
        },
        "tool_config": {
            "mode": "any",
            "allowed_function_names": "lookup",
            "include_server_side_tool_invocations": "true",
            "retrieval_config": {
                "language_code": "ko-KR",
                "lat_lng": {
                    "latitude": "37.5665",
                    "longitude": "126.9780",
                },
            },
        },
    })

    assert response.status_code == 200
    assert seen["request"]["safetySettings"] == [{
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    }]
    assert seen["request"]["generationConfig"]["thinkingConfig"] == {
        "thinkingBudget": 128,
        "includeThoughts": True,
        "thinkingLevel": "HIGH",
    }
    assert seen["request"]["toolConfig"] == {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": ["lookup"],
        },
        "includeServerSideToolInvocations": True,
        "retrievalConfig": {
            "languageCode": "ko-KR",
            "latLng": {
                "latitude": 37.5665,
                "longitude": 126.978,
            },
        },
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

    code_execution = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "run code"}]}],
        "tools": [{"code_execution": {}}],
    })
    google_maps = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "map it",
        "tools": [{"google_maps": {"enable_widget": "true"}}],
    })
    computer_use = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "use browser",
        "tools": [{
            "computer_use": {
                "enable_prompt_injection_detection": "true",
                "disabled_safety_policies": ["policy"],
                "excluded_predefined_functions": ["navigate"],
            }
        }],
    })
    mcp_servers = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "call mcp",
        "tools": [{"mcp_servers": [{"name": "local", "streamable_http_transport": {"url": "http://example.test"}}]}],
    })

    assert code_execution.status_code == 501
    assert code_execution.json()["error"]["status"] == "UNIMPLEMENTED"
    assert "code_execution" in code_execution.json()["error"]["message"]
    assert google_maps.status_code == 501
    assert "google_maps" in google_maps.json()["error"]["message"]
    assert computer_use.status_code == 501
    assert "computer_use" in computer_use.json()["error"]["message"]
    assert mcp_servers.status_code == 501
    assert "mcp_servers" in mcp_servers.json()["error"]["message"]


def test_gemini_url_context_fetches_and_injects_local_context(monkeypatch):
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            return {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(proxy, "_gemini_fetch_url_context_url", lambda url: "Fetched page text for " + url)
    client = TestClient(proxy.app)

    response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "read https://example.com/page"}]}],
        "tools": [{"urlContext": {}}],
    })

    assert response.status_code == 200
    assert seen["request"]["contents"][0]["parts"][0]["text"].startswith("Local Gemini urlContext results:")
    assert "Fetched page text for https://example.com/page" in seen["request"]["contents"][0]["parts"][0]["text"]
    assert seen["request"]["contents"][1]["parts"][0]["text"] == "read https://example.com/page"
    assert "tools" not in seen["request"]
    assert "_urlContextMetadata" not in seen["request"]
    metadata = response.json()["candidates"][0]["urlContextMetadata"]["urlMetadata"][0]
    assert metadata["retrievedUrl"] == "https://example.com/page"
    assert metadata["urlRetrievalStatus"] == "URL_RETRIEVAL_STATUS_SUCCESS"

    snake = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": "read https://example.com/snake",
        "tools": {"url_context": {}},
    })
    assert snake.status_code == 200
    assert "urlContextMetadata" in snake.json()["candidates"][0]


def test_gemini_generate_content_alt_sse(monkeypatch):
    seen = {}

    class FakeClient:
        async def generate_raw_stream_async(self, *, request, model=""):
            seen["model"] = model
            yield {"response": {"candidates": [{"content": {"parts": [{"text": "alt stream"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    with client.stream("POST", "/v1beta/models/gemini-3-flash-agent:generateContent?alt=sse", json={
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]
    }) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "alt stream" in body
    assert '"modelVersion": "gemini-3-flash-agent"' in body
    assert seen["model"] == "gemini-3-flash-agent"
    assert "data: [DONE]" not in body


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
    assert "data: [DONE]" not in body


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
    assert "data: [DONE]" not in body


def test_gemini_predict_and_predict_long_running(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    seen = []

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen.append(request)
            return {"response": {"candidates": [{"content": {"parts": [{"text": request["contents"][0]["parts"][0]["text"]}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    predicted = client.post("/v1beta/models/gemini-3-flash-agent:predict", json={
        "instances": [{"text": "predict me"}],
        "provider_options": {
            "google": {
                "system_instruction": "predict system",
                "max_output_tokens": "10",
                "tool_config": {"function_calling_config": {"mode": "none"}},
            }
        },
        "processing_options": {"media_resolution": "MEDIA_RESOLUTION_LOW"},
    })
    long_running = client.post("/v1beta/models/gemini-3-flash-agent:predictLongRunning", json={
        "instances": [{"text": "later"}],
        "parameters": {
            "generation_config": {"max_output_tokens": "7"},
            "safety_settings": {"harm_category": "harassment", "harm_block_threshold": "none"},
        },
    })
    parameter_wrapped = client.post("/v1beta/models/gemini-3-flash-agent:predict", json={
        "instances": [{"parts": [{"text": "parameter wrapped"}]}],
        "parameters": {
            "generation_config": {
                "max_output_tokens": "12",
                "response_mime_type": "application/json",
            },
            "safety_settings": {
                "harm_category": "dangerous",
                "harm_block_threshold": "only_high",
            },
            "tool_config": {"mode": "none"},
            "response_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string", "min_length": 1}},
            },
        },
    })

    assert predicted.status_code == 200
    assert predicted.json()["predictions"][0]["candidates"][0]["content"]["parts"][0]["text"] == "predict me"
    assert seen[0]["systemInstruction"]["parts"][0]["text"] == "predict system"
    assert seen[0]["generationConfig"]["maxOutputTokens"] == 10
    assert seen[0]["toolConfig"]["functionCallingConfig"] == {"mode": "NONE"}
    assert "processingOptions" not in seen[0]
    assert parameter_wrapped.status_code == 200
    assert seen[2]["generationConfig"]["maxOutputTokens"] == 12
    assert seen[2]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen[2]["generationConfig"]["responseSchema"]["properties"]["answer"]["minLength"] == 1
    assert seen[2]["safetySettings"] == [{
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    }]
    assert seen[2]["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}
    assert long_running.status_code == 200
    assert long_running.json()["done"] is True
    assert long_running.json()["response"]["predictions"]
    assert long_running.json()["metadata"]["deployedModelId"] == "models/gemini-3-flash-agent"
    assert long_running.json()["metadata"]["createTime"]
    assert long_running.json()["metadata"]["endTime"]
    assert long_running.json()["metadata"]["request"]["contents"][0]["parts"][0]["text"] == "later"
    assert long_running.json()["metadata"]["request"]["generationConfig"]["maxOutputTokens"] == 7
    assert long_running.json()["metadata"]["request"]["safetySettings"] == [{
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    }]

    op_id = long_running.json()["name"].split("/", 1)[1]
    model_operation = client.get(f"/v1beta/models/gemini-3-flash-agent/operations/{op_id}")
    model_operations = client.get("/v1beta/models/gemini-3-flash-agent/operations")
    filtered_model_operations = client.get(
        f"/v1beta/models/gemini-3-flash-agent/operations?filter=operation.name:{op_id}&returnPartialSuccess=true"
    )
    waited_operation = client.post(f"/v1beta/models/gemini-3-flash-agent/operations/{op_id}:wait")
    assert model_operation.status_code == 200
    assert model_operation.json()["name"] == long_running.json()["name"]
    assert model_operations.status_code == 200
    assert model_operations.json()["operations"][0]["name"] == long_running.json()["name"]
    assert filtered_model_operations.status_code == 200
    assert filtered_model_operations.json()["operations"][0]["name"] == long_running.json()["name"]
    assert filtered_model_operations.json()["unreachable"] == []
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

    text = client.post("/v1beta/models/gemini-3-flash-agent:generateText", json={
        "prompt": {"text": "hello text"},
        "provider_options": {
            "google": {
                "system_instruction": "legacy system",
                "max_output_tokens": "12",
                "tool_config": {"function_calling_config": {"mode": "none"}},
            }
        },
        "response_format": "text/plain",
        "processing_options": {"media_resolution": "MEDIA_RESOLUTION_LOW"},
    })
    message = client.post("/v1beta/models/gemini-3-flash-agent:generateMessage", json={"prompt": {"message": {"content": "hello msg"}}})
    answer = client.post("/v1/models/gemini-3-flash-agent:generateAnswer", json={"text": "hello answer"})
    counted_text = client.post("/v1beta/models/gemini-3-flash-agent:countTextTokens", json={"prompt": {"text": "count these"}})
    counted_message = client.post("/v1beta/models/gemini-3-flash-agent:countMessageTokens", json={"prompt": {"message": {"content": "count msg"}}})

    assert text.status_code == 200
    assert text.json()["candidates"][0]["output"] == "legacy:hello text"
    assert seen[0]["systemInstruction"]["parts"][0]["text"] == "legacy system"
    assert seen[0]["generationConfig"]["maxOutputTokens"] == 12
    assert seen[0]["generationConfig"]["responseMimeType"] == "text/plain"
    assert seen[0]["toolConfig"]["functionCallingConfig"] == {"mode": "NONE"}
    assert "processingOptions" not in seen[0]
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
    with client.stream(
        "POST",
        "/v1beta/projects/proj/locations/global/publishers/google/models/gemini-3-flash-agent:streamGenerateContent",
        json={"contents": "hi"},
    ) as vertex_response:
        vertex_body = vertex_response.read().decode()

    assert response.status_code == 200
    assert 'data: {"candidates":' in body
    assert "data: [DONE]" not in body
    assert vertex_response.status_code == 200
    assert 'data: {"candidates":' in vertex_body
    assert "data: [DONE]" not in vertex_body


class _FastApiTransport(httpx.BaseTransport):
    def __init__(self, client: TestClient):
        self.client = client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        target = request.url.path
        if request.url.query:
            target += "?" + request.url.query.decode("ascii")
        response = self.client.request(
            request.method,
            target,
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx.Response(
            response.status_code,
            headers=dict(response.headers),
            content=response.content,
            request=request,
        )


def test_google_genai_sdk_vertex_collection_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_GENERATED_FILES_DIR", str(tmp_path / "generated"))

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            assert model == "gemini-3-flash-agent"
            return {
                "response": {
                    "candidates": [{
                        "content": {"role": "model", "parts": [{"text": "sdk ok"}]},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3},
                }
            }

        def generate_image(self, *, prompt, output_dir, aspect_ratio="", image_size=""):
            output = output_dir / "sdk-image.png"
            output.write_bytes(b"sdk-image")
            return output

        async def generate_raw_stream_async(self, *, request, model=""):
            assert model == "gemini-3-flash-agent"
            yield {
                "response": {
                    "candidates": [{
                        "content": {"role": "model", "parts": [{"text": "sdk stream"}]},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3},
                }
            }

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    app_client = TestClient(proxy.app)
    sdk_http = httpx.Client(transport=_FastApiTransport(app_client))
    sdk = genai.Client(
        vertexai=True,
        http_options=types.HttpOptions(
            base_url="http://testserver/v1beta",
            api_version=None,
            base_url_resource_scope=types.ResourceScope.COLLECTION,
            httpx_client=sdk_http,
        ),
    )

    models = list(sdk.models.list(config={"page_size": 1}))
    generated = sdk.models.generate_content(model="gemini-3-flash-agent", contents="hello")
    counted = sdk.models.count_tokens(model="gemini-3-flash-agent", contents="hello")
    streamed = list(sdk.models.generate_content_stream(model="gemini-3-flash-agent", contents="hello"))
    embedded = app_client.post(
        "/v1beta/publishers/google/models/gemini-3-flash-agent:embedContent",
        json={"content": {"parts": [{"text": "embed me"}]}},
    )
    predicted = app_client.post(
        "/v1beta/publishers/google/models/gemini-image-latest:predict",
        json={"instances": [{"prompt": "draw"}]},
    )

    assert models
    assert models[0].name.startswith("models/")
    assert generated.text == "sdk ok"
    assert counted.total_tokens > 0
    assert "".join(chunk.text or "" for chunk in streamed) == "sdk stream"
    assert embedded.status_code == 200
    assert embedded.json()["embedding"]["values"]
    assert predicted.status_code == 200
    assert predicted.json()["predictions"][0]["bytesBase64Encoded"]


def test_google_genai_sdk_developer_files_upload_and_models_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILES_DIR", str(tmp_path / "files"))
    app_client = TestClient(proxy.app)
    sdk_http = httpx.Client(transport=_FastApiTransport(app_client))
    sdk = genai.Client(
        api_key="test-key",
        http_options=types.HttpOptions(
            base_url="http://testserver/v1beta",
            api_version="",
            httpx_client=sdk_http,
        ),
    )
    upload_source = io.BytesIO(b"sdk file upload")

    filtered = app_client.get('/v1beta/models?filter=displayName:"3.5 Flash"')
    unmatched = app_client.get('/v1beta/models?filter=name:"not-a-model"')
    uploaded = sdk.files.upload(
        file=upload_source,
        config={"mime_type": "text/plain", "display_name": "sdk-upload.txt"},
    )

    assert filtered.status_code == 200
    assert filtered.json()["models"]
    assert all("3.5 Flash" in item["displayName"] for item in filtered.json()["models"])
    assert unmatched.status_code == 200
    assert unmatched.json()["models"] == []
    assert uploaded.name.startswith("files/")
    assert uploaded.display_name == "sdk-upload.txt"
    assert uploaded.mime_type == "text/plain"


def test_google_genai_sdk_vertex_batch_prediction_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", str(tmp_path / "batches"))
    app_client = TestClient(proxy.app)
    sdk_http = httpx.Client(transport=_FastApiTransport(app_client))
    sdk = genai.Client(
        vertexai=True,
        http_options=types.HttpOptions(
            base_url="http://testserver/v1beta",
            api_version=None,
            base_url_resource_scope=types.ResourceScope.COLLECTION,
            httpx_client=sdk_http,
        ),
    )

    job = sdk.batches.create(
        model="gemini-3-flash-agent",
        src="gs://bucket/input.jsonl",
        config={"display_name": "sdk vertex batch"},
    )
    fetched = sdk.batches.get(name=job.name)
    listed = list(sdk.batches.list(config={"page_size": 10}))
    sdk.batches.cancel(name=job.name)
    deleted = sdk.batches.delete(name=job.name)

    assert job.name.startswith("projects/local-project/locations/global/batchPredictionJobs/")
    assert job.display_name == "sdk vertex batch"
    assert job.state == types.JobState.JOB_STATE_SUCCEEDED
    assert job.src.gcs_uri == ["gs://bucket/input.jsonl"]
    assert job.dest.gcs_uri == "gs://bucket/input/dest"
    assert job.output_info.gcs_output_directory == "gs://bucket/input/dest/prediction-results"
    assert fetched.name == job.name
    assert {item.name for item in listed} == {job.name}
    assert deleted.name == job.name
    assert deleted.done is True
    assert app_client.get(f"/v1beta/batchPredictionJobs/{job.name.rsplit('/', 1)[-1]}").status_code == 404


def test_vertex_batch_prediction_jobs_project_scoped_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_BATCHES_DIR", str(tmp_path / "batches"))
    client = TestClient(proxy.app)

    created = client.post(
        "/v1beta/projects/demo/locations/us-central1/batchPredictionJobs",
        json={
            "displayName": "rest vertex batch",
            "model": "publishers/google/models/gemini-3-flash-agent",
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": ["gs://bucket/input.jsonl"]},
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": "gs://bucket/out"},
            },
            "labels": {"client": "test"},
        },
    )

    assert created.status_code == 200
    job = created.json()
    assert job["name"].startswith("projects/demo/locations/us-central1/batchPredictionJobs/")
    assert job["displayName"] == "rest vertex batch"
    assert job["state"] == "JOB_STATE_SUCCEEDED"
    assert job["inputConfig"]["gcsSource"]["uris"] == ["gs://bucket/input.jsonl"]
    assert job["outputInfo"]["gcsOutputDirectory"] == "gs://bucket/out/prediction-results"
    assert job["completionStats"]["successfulCount"] == "0"
    assert job["labels"] == {"client": "test"}

    job_id = job["name"].rsplit("/", 1)[-1]
    scoped_get = client.get(f"/v1beta/projects/demo/locations/us-central1/batchPredictionJobs/{job_id}")
    full_name_get = client.get(f"/v1beta/batchPredictionJobs/{job['name']}")
    listed = client.get("/v1beta/projects/demo/locations/us-central1/batchPredictionJobs", params={"filter": 'displayName="rest vertex batch"'})
    developer_batches = client.get("/v1beta/batches")
    cancelled = client.post(f"/v1beta/projects/demo/locations/us-central1/batchPredictionJobs/{job_id}:cancel")
    deleted = client.delete(f"/v1beta/projects/demo/locations/us-central1/batchPredictionJobs/{job_id}")
    missing = client.get(f"/v1beta/projects/demo/locations/us-central1/batchPredictionJobs/{job_id}")

    assert scoped_get.status_code == 200
    assert scoped_get.json()["name"] == job["name"]
    assert full_name_get.status_code == 200
    assert full_name_get.json()["name"] == job["name"]
    assert listed.status_code == 200
    assert [item["name"] for item in listed.json()["batchPredictionJobs"]] == [job["name"]]
    assert developer_batches.status_code == 200
    assert developer_batches.json()["batches"] == []
    assert cancelled.status_code == 200
    assert cancelled.json() == {}
    assert deleted.status_code == 200
    assert deleted.json()["done"] is True
    assert missing.status_code == 404


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
    vertex_created = client.post(
        "/v1beta/projects/proj/locations/global/publishers/google/models/gemini-3-flash-agent:batchGenerateContent",
        json={
            "requests": [
                {"contents": [{"role": "user", "parts": [{"text": "vertex first"}]}]},
                {"contents": [{"role": "user", "parts": [{"text": "vertex second"}]}]},
            ]
        },
    )

    assert created.status_code == 200
    assert vertex_created.status_code == 200
    operation = created.json()
    assert operation["done"] is True
    assert len(operation["response"]["responses"]) == 2
    assert len(vertex_created.json()["response"]["responses"]) == 2

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
    assert operation["name"] in {item["name"] for item in listed.json()["operations"]}
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
    assert batch_name in {item["name"] for item in batches.json()["operations"]}
    assert operation["name"] in {item["operation"] for item in batches.json()["batches"]}
    assert deleted.status_code == 200


def test_gemini_operations_v1_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    client = TestClient(proxy.app)
    operation = proxy._gemini_store_operation({
        "name": "operations/op_v1_alias",
        "metadata": {"model": "models/gemini-3-flash-agent"},
        "done": False,
    })
    proxy._gemini_store_operation({
        "name": "operations/op_other_model",
        "metadata": {"model": "models/other"},
        "done": True,
    })
    scoped_operation = proxy._gemini_store_operation({
        "name": "operations/op_scoped_cancel",
        "metadata": {"model": "models/gemini-3-flash-agent"},
        "done": False,
    })

    listed = client.get("/v1/operations")
    filtered = client.get(
        "/v1/operations",
        params={
            "filter": 'done=false AND metadata.model="models/gemini-3-flash-agent"',
            "return_partial_success": "true",
        },
    )
    fetched = client.get(f"/v1/{operation['name']}")
    waited = client.post(f"/v1/{operation['name']}:wait")
    cancelled = client.post(f"/v1/{operation['name']}:cancel")
    cancelled_fetched = client.get(f"/v1/{operation['name']}")
    scoped_cancelled = client.post("/v1/models/gemini-3-flash-agent/operations/op_scoped_cancel:cancel")
    scoped_cancelled_fetched = client.get(f"/v1/{scoped_operation['name']}")
    wrong_scoped = client.get("/v1/models/gemini-3-flash-agent/operations/op_other_model")
    deleted = client.delete(f"/v1/{operation['name']}")
    missing = client.get(f"/v1/{operation['name']}")

    assert listed.status_code == 200
    assert operation["name"] in {item["name"] for item in listed.json()["operations"]}
    assert filtered.status_code == 200
    assert {item["name"] for item in filtered.json()["operations"]} == {operation["name"], scoped_operation["name"]}
    assert filtered.json()["unreachable"] == []
    assert fetched.status_code == 200
    assert fetched.json()["done"] is False
    assert waited.status_code == 200
    assert waited.json()["name"] == operation["name"]
    assert cancelled.status_code == 200
    assert cancelled.json() == {}
    assert cancelled_fetched.status_code == 200
    assert cancelled_fetched.json()["done"] is True
    assert cancelled_fetched.json()["error"]["status"] == "CANCELLED"
    assert scoped_cancelled.status_code == 200
    assert scoped_cancelled_fetched.json()["done"] is True
    assert scoped_cancelled_fetched.json()["error"]["status"] == "CANCELLED"
    assert scoped_cancelled_fetched.json()["metadata"]["endTime"]
    assert wrong_scoped.status_code == 404
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
    generated_sdk_wrapped = client.post("/v1beta/models/gemini-3-flash-agent:batchGenerateContent", json={
        "batch": {
            "displayName": "sdk inline batch",
            "inputConfig": {
                "requests": {
                    "requests": [{
                        "request": {
                            "contents": [{"role": "user", "parts": [{"text": "sdk inline"}]}],
                        },
                    }]
                }
            },
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
    snake_wrapped = client.post("/v1beta/batches", json={
        "generate_content_batch": {
            "model": "models/gemini-3-flash-agent",
            "display_name": "snake wrapped batch",
            "requests": [{"contents": [{"role": "user", "parts": [{"text": "snake"}]}]}],
        },
        "input_config": {"instances_format": "jsonl"},
        "output_config": {"predictions_format": "jsonl"},
        "priority": "HIGH",
    })
    snake_embedded = client.post("/v1beta/batches", json={
        "embed_content_batch": {
            "model": "models/gemini-3-flash-agent",
            "display_name": "snake wrapped embed",
            "requests": [{
                "embed_content_request": {
                    "content": {"parts": [{"text": "snake embed"}]},
                },
            }],
            "config": {"output_dimensionality": 5},
        },
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
    assert generated_sdk_wrapped.status_code == 200
    assert generated_sdk_wrapped.json()["metadata"]["batchResource"]["displayName"] == "sdk inline batch"
    assert generated_sdk_wrapped.json()["response"]["responses"][0]["candidates"][0]["content"]["parts"][0]["text"] == "sdk inline"
    assert embedded.status_code == 200
    assert embedded.json()["metadata"]["batchResource"]["displayName"] == "wrapped embed"
    assert embedded.json()["metadata"]["state"] == "BATCH_STATE_SUCCEEDED"
    assert embedded.json()["metadata"]["stats"]["successfulRequestCount"] == "1"
    assert embedded.json()["response"]["embeddings"][0]["values"]
    assert len(embedded.json()["response"]["embeddings"][0]["values"]) == 8
    assert embedded_request_wrapped.status_code == 200
    assert len(embedded_request_wrapped.json()["response"]["embeddings"][0]["values"]) == 6
    assert snake_wrapped.status_code == 200
    assert snake_wrapped.json()["metadata"]["batchResource"]["displayName"] == "snake wrapped batch"
    assert snake_wrapped.json()["metadata"]["batchResource"]["priority"] == "HIGH"
    assert snake_wrapped.json()["metadata"]["batchResource"]["inputConfig"]["instancesFormat"] == "jsonl"
    assert snake_wrapped.json()["metadata"]["batchResource"]["outputConfig"]["predictionsFormat"] == "jsonl"
    assert snake_wrapped.json()["response"]["responses"][0]["candidates"][0]["content"]["parts"][0]["text"] == "snake"
    assert snake_embedded.status_code == 200
    assert snake_embedded.json()["metadata"]["batchResource"]["displayName"] == "snake wrapped embed"
    assert len(snake_embedded.json()["response"]["embeddings"][0]["values"]) == 5
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
    assert operation.json()["metadata"]["batchResource"]["name"] == batch_operation["name"]
    assert operation.json()["metadata"]["state"] == "BATCH_STATE_SUCCEEDED"
    assert operation.json()["metadata"]["operation"] == batch_resource["operation"]
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
    filtered = client.get('/v1beta/batches?filter=displayName:"second"&return_partial_success=true')
    listed_operations = client.get("/v1beta/operations?page_size=1&return_partial_success=true")
    done_filtered = client.get('/v1beta/batches?filter=done=true AND metadata.batchResource.state="BATCH_STATE_SUCCEEDED"')
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
    snake_query_masked = client.patch(f"/v1beta/{first['name']}:updateGenerateContentBatch?update_mask=batch.display_name", json={
        "batch": {"display_name": "snake query masked"},
    })
    bad_patch = client.patch(f"/v1beta/{first['name']}:updateGenerateContentBatch?updateMask=state", json={
        "state": "BATCH_STATE_CANCELLED",
    })

    assert listed.status_code == 200
    assert len(listed.json()["batches"]) == 1
    assert listed.json()["nextPageToken"] == "1"
    assert filtered.status_code == 200
    assert [item["metadata"]["displayName"] for item in filtered.json()["operations"]] == ["second"]
    assert filtered.json()["unreachable"] == []
    assert listed_operations.status_code == 200
    assert len(listed_operations.json()["operations"]) == 1
    assert listed_operations.json()["unreachable"] == []
    assert done_filtered.status_code == 200
    assert len(done_filtered.json()["operations"]) == 2
    assert patched.status_code == 200
    assert patched.json()["metadata"]["batchResource"]["displayName"] == "patched"
    assert priority.status_code == 200
    assert priority.json()["metadata"]["batchResource"]["priority"] == "7"
    assert body_masked.status_code == 200
    assert body_masked.json()["metadata"]["batchResource"]["displayName"] == "body masked"
    assert snake_query_masked.status_code == 200
    assert snake_query_masked.json()["metadata"]["batchResource"]["displayName"] == "snake query masked"
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
    assert first_body["object"] == "interaction"
    assert first_body["created"] == first_body["createTime"]
    assert first_body["updated"] == first_body["updateTime"]
    assert first_body["created_at"] == first_body["created"]
    assert first_body["updated_at"] == first_body["updated"]
    assert first_body["outputText"] == "echo:first"
    assert first_body["output_text"] == "echo:first"
    assert first_body["output"]["modelVersion"] == "gemini-3-flash-agent"
    assert first_body["output"]["responseId"].startswith("resp_")
    assert first_body["usage"]["totalTokens"] > 0
    assert first_body["usage"]["total_tokens"] == first_body["usage"]["totalTokens"]
    assert first_body["usage"]["total_input_tokens"] == first_body["usage"]["inputTokens"]
    assert first_body["usage"]["total_output_tokens"] == first_body["usage"]["outputTokens"]
    assert first_body["usage"]["input_tokens"] == first_body["usage"]["inputTokens"]
    assert first_body["usage"]["output_tokens"] == first_body["usage"]["outputTokens"]
    assert first_body["usage"]["input_tokens_by_modality"][0]["modality"] == "TEXT"
    assert first_body["usage"]["input_tokens_by_modality"][0]["tokens"] == first_body["usage"]["inputTokens"]
    assert first_body["usage"]["output_tokens_by_modality"][0]["modality"] == "TEXT"
    assert first_body["usage"]["output_tokens_by_modality"][0]["tokens"] == first_body["usage"]["outputTokens"]
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
    listed = client.get("/v1beta/interactions?page_size=2&page_token=0")
    no_store = client.post("/v1beta/interactions", json={"input": "transient", "store": False})
    missing = client.get(f"/v1beta/{no_store.json()['name']}")
    cancelled = client.post(f"/v1beta/{first_body['name']}:cancel")
    deleted = client.delete(f"/v1beta/{first_body['name']}")

    assert fetched.status_code == 200
    assert listed.status_code == 200
    assert listed.json()["interactions"][0]["object"] == "interaction"
    assert first_body["name"] in {item["name"] for item in listed.json()["interactions"]}
    assert listed.json()["nextPageToken"] == ""
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


def test_gemini_interactions_native_computer_use_requires_action(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_INTERACTIONS_DIR", str(tmp_path / "interactions"))

    class FailingClient:
        def generate_raw(self, *, request, model=""):
            raise AssertionError("native computer_use interactions must not be forwarded upstream")

    monkeypatch.setattr(proxy, "_get_client", lambda: FailingClient())
    client = TestClient(proxy.app)

    created = client.post("/v1beta/interactions", json={
        "model": "gemini-3.5-flash",
        "input": "Search for Gemini API on Google.",
        "tools": [{
            "type": "computer_use",
            "environment": "browser",
        }],
    })

    assert created.status_code == 200
    body = created.json()
    assert body["status"] == "requires_action"
    assert body["model"] == "models/gemini-3.5-flash"
    assert body["output"]["modelVersion"] == "gemini-3.5-flash"
    assert body["output"]["candidates"][0]["content"]["parts"][0]["functionCall"]["name"] == "open_web_browser"
    assert body["output"]["computerUse"]["environment"] == "ENVIRONMENT_BROWSER"
    assert body["steps"][0]["type"] == "computer_use"
    assert body["steps"][0]["status"] == "requires_action"
    assert body["steps"][0]["environment"] == "ENVIRONMENT_BROWSER"
    assert body["requiredAction"]["type"] == "submit_computer_use_result"
    assert body["required_action"]["tool_calls"][0]["name"] == "open_web_browser"

    fetched = client.get(f"/v1beta/{body['name']}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "requires_action"

    transient = client.post("/v1beta/interactions", json={
        "input": "Open a browser.",
        "store": False,
        "tools": [{"type": "computer_use", "environment": "browser"}],
    })
    missing = client.get(f"/v1beta/{transient.json()['name']}")
    assert transient.status_code == 200
    assert missing.status_code == 404


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

    wrapped = client.post("/v1/interactions", json={
        "config": {"temperature": 0.1},
        "interaction": {
            "model": "models/gemini-3-flash-agent",
            "input": "wrapped",
            "store": False,
        },
    })
    background = client.post("/v1/interactions", json={
        "interaction": {
            "input": "async later",
            "background": True,
        }
    })

    fetched = client.get(f"/v1/{body['name']}")
    fetched_with_version_prefix = client.get(f"/v1/interactions/v1/{body['name']}")
    listed = client.get("/v1/interactions?pageSize=2&pageToken=0")
    colon_cancelled = client.post(f"/v1/{body['name']}:cancel")
    rest_cancelled = client.post(f"/v1/{body['name']}/cancel")
    background_cancelled = client.post(f"/v1/{background.json()['name']}:cancel")
    deleted = client.delete(f"/v1/{body['name']}")
    missing = client.get(f"/v1/{body['name']}")

    assert wrapped.status_code == 200
    assert wrapped.json()["object"] == "interaction"
    assert wrapped.json()["created_at"] == wrapped.json()["created"]
    assert wrapped.json()["output_text"] == "v1:wrapped"
    assert wrapped.json()["outputText"] == "v1:wrapped"
    assert background.status_code == 200
    assert background.json()["status"] == "in_progress"
    assert background.json()["background"] is True
    assert background.json()["usage"] == {}
    assert fetched.status_code == 200
    assert fetched_with_version_prefix.status_code == 200
    assert fetched_with_version_prefix.json()["name"] == body["name"]
    assert listed.status_code == 200
    assert {item["name"] for item in listed.json()["interactions"]} >= {body["name"], background.json()["name"]}
    assert colon_cancelled.status_code == 200
    assert rest_cancelled.status_code == 200
    assert background_cancelled.status_code == 200
    assert background_cancelled.json()["status"] == "cancelled"
    assert background_cancelled.json()["updated"] == background_cancelled.json()["updateTime"]
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
            "response_modalities": ["text"],
            "media_resolution": "low",
            "audio_timestamp": "false",
        },
    })

    assert text_interaction.status_code == 200
    parts = seen["request"]["contents"][0]["parts"]
    assert parts[0] == {"text": "describe"}
    assert parts[1]["inlineData"] == {"mimeType": "image/jpeg", "data": "aW1hZ2U="}
    assert seen["request"]["generationConfig"]["responseModalities"] == ["TEXT"]
    assert seen["request"]["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_LOW"
    assert seen["request"]["generationConfig"]["audioTimestamp"] is False

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

    configured_interaction = client.post("/v1beta/interactions", json={
        "input": "use config",
        "config": {
            "system_instruction": "Use concise answers.",
            "temperature": "0.25",
            "top_k": "12",
            "safety_settings": {"harassment": "none"},
            "function_declarations": {
                "name": "lookup",
                "description": "Find a record.",
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
            },
            "tool_config": {
                "mode": "any",
                "allowed_function_names": "lookup",
            },
        },
        "store": False,
    })

    assert configured_interaction.status_code == 200
    assert seen["request"]["systemInstruction"]["parts"] == [{"text": "Use concise answers."}]
    assert seen["request"]["generationConfig"]["temperature"] == 0.25
    assert seen["request"]["generationConfig"]["topK"] == 12
    assert seen["request"]["safetySettings"] == [{
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    }]
    assert seen["request"]["tools"][0]["functionDeclarations"][0]["parameters"] == {
        "type": "object",
        "properties": {"id": {"type": "string"}},
    }
    assert seen["request"]["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["lookup"],
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


def test_gemini_agents_crud_and_interaction_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_AGENTS_DIR", str(tmp_path / "agents"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_INTERACTIONS_DIR", str(tmp_path / "interactions"))
    seen = {}

    class FakeClient:
        def generate_raw(self, *, request, model=""):
            seen["request"] = request
            seen["model"] = model
            return {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "agent ok"}]}}]}}

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    created = client.post("/v1beta/agents", json={
        "agent": {
            "display_name": "Research Agent",
            "description": "Uses stored defaults.",
            "model": "models/gemini-3-flash-agent",
            "system_instruction": "Answer as the saved agent.",
            "tools": [{"function_declarations": [{"name": "lookup", "parameters": {"type": "object"}}]}],
            "tool_config": {"function_calling_config": {"mode": "auto"}},
            "base_environment": {"timezone": "Asia/Seoul"},
        }
    })
    body = created.json()
    fetched = client.get(f"/v1beta/{body['name']}")
    listed = client.get("/v1beta/agents?page_size=1")
    interacted = client.post("/v1beta/interactions", json={
        "agent": body["name"],
        "input": "hello",
        "store": False,
    })
    deleted = client.delete(f"/v1beta/{body['name']}")
    missing = client.get(f"/v1beta/{body['name']}")

    assert created.status_code == 200
    assert body["name"].startswith("agents/")
    assert body["id"] == body["name"].split("/", 1)[1]
    assert body["displayName"] == "Research Agent"
    assert body["display_name"] == "Research Agent"
    assert body["systemInstruction"]["parts"] == [{"text": "Answer as the saved agent."}]
    assert body["baseEnvironment"] == {"timezone": "Asia/Seoul"}
    assert fetched.status_code == 200
    assert fetched.json()["name"] == body["name"]
    assert listed.status_code == 200
    assert listed.json()["agents"][0]["name"] == body["name"]
    assert interacted.status_code == 200
    assert interacted.json()["agent"] == body["name"]
    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["systemInstruction"]["parts"] == [{"text": "Answer as the saved agent."}]
    assert seen["request"]["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    assert seen["request"]["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    assert "environment" not in seen["request"]
    assert deleted.status_code == 200
    assert deleted.json() == {}
    assert missing.status_code == 404


def test_gemini_webhooks_crud_and_v1_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_WEBHOOKS_DIR", str(tmp_path / "webhooks"))
    client = TestClient(proxy.app)

    created = client.post("/v1/webhooks", json={
        "config": {
            "target_uri": "https://example.test/hook",
            "new_signing_secret": True,
        },
        "webhook": {
            "display_name": "Batch updates",
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

    no_secret = client.post("/v1/webhooks", json={
        "config": {
            "target_uri": "https://example.test/no-secret",
            "new_signing_secret": False,
        },
        "webhook": {"event_types": ["webhooks.ping"]},
    })
    assert no_secret.status_code == 200
    assert "newSigningSecret" not in no_secret.json()
    assert "signingSecrets" not in no_secret.json()

    fetched = client.get(f"/v1/{webhook['name']}")
    listed = client.get("/v1/webhooks?page_size=2&page_token=0")
    patched = client.patch(f"/v1/{webhook['name']}?update_mask=displayName", json={
        "display_name": "Renamed",
        "target_uri": "https://example.test/ignored",
    })
    body_masked = client.patch(f"/v1/{webhook['name']}", json={
        "update_mask": "targetUri",
        "display_name": "Ignored by mask",
        "target_uri": "https://example.test/body-mask",
    })
    state_masked = client.patch(f"/v1/{webhook['name']}?updateMask=webhook.state", json={
        "config": {"state": "DISABLED"},
        "webhook": {"target_uri": "https://example.test/ignored-state-mask"},
    })

    assert fetched.status_code == 200
    assert fetched.json()["name"] == webhook["name"]
    assert listed.status_code == 200
    assert webhook["name"] in {item["name"] for item in listed.json()["webhooks"]}
    assert "newSigningSecret" not in fetched.json()
    assert patched.status_code == 200
    assert patched.json()["displayName"] == "Renamed"
    assert patched.json()["targetUri"] == "https://example.test/hook"
    assert body_masked.status_code == 200
    assert body_masked.json()["displayName"] == "Renamed"
    assert body_masked.json()["targetUri"] == "https://example.test/body-mask"
    assert state_masked.status_code == 200
    assert state_masked.json()["state"] == "disabled"
    assert state_masked.json()["targetUri"] == "https://example.test/body-mask"

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
    deleted_no_secret = client.delete(f"/v1/{no_secret.json()['name']}")
    missing = client.get(f"/v1/{webhook['name']}")
    assert deleted.status_code == 200
    assert deleted_no_secret.status_code == 200
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
    assert file_resource["expirationTime"]

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

    object_file_uri = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "contents": [{
            "role": "user",
            "parts": [
                {"fileData": {"fileUri": {"uri": file_resource["uri"], "mimeType": "text/plain"}}},
                {"fileData": {"file": file_resource}},
            ],
        }]
    })

    assert object_file_uri.status_code == 200
    parts = seen["request"]["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "text/plain"
    assert parts[0]["inlineData"]["data"]
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
    assert file_resource["state"] == "ACTIVE"
    assert file_resource["updateTime"]
    assert base64.b64decode(file_resource["sha256Hash"])

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
    assert official_file["state"] == "ACTIVE"
    assert official_file.get("expirationTime") is None
    official_download = client.get(f"/v1beta/{official_file['name']}:download")
    assert official_download.status_code == 404

    config_created = client.post("/v1beta/files", json={
        "file": {
            "displayName": "config-file.txt",
            "uri": "gs://bucket/config-file.txt",
        },
        "config": {"mime_type": "text/markdown", "sizeBytes": "9", "state": "file_state_processing"},
    })
    assert config_created.status_code == 200
    assert config_created.json()["file"]["mimeType"] == "text/markdown"
    assert config_created.json()["file"]["sizeBytes"] == "9"
    assert config_created.json()["file"]["state"] == "PROCESSING"

    config_file_created = client.post("/v1beta/files", json={
        "config": {
            "mime_type": "text/plain",
            "file": {
                "display_name": "config-wrapper.txt",
                "uri": "gs://bucket/config-wrapper.txt",
            },
        },
    })
    assert config_file_created.status_code == 200
    assert config_file_created.json()["file"]["displayName"] == "config-wrapper.txt"
    assert config_file_created.json()["file"]["mimeType"] == "text/plain"
    assert config_file_created.json()["file"]["uri"] == "gs://bucket/config-wrapper.txt"

    hex_hash_created = client.post("/v1beta/files", json={
        "file": {
            "displayName": "hex-hash.txt",
            "uri": "gs://bucket/hex-hash.txt",
            "sha256_hash": "00" * 32,
        },
    })
    assert hex_hash_created.status_code == 200
    assert hex_hash_created.json()["file"]["sha256Hash"] == "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

    official_registered = client.post("/v1beta/files:register", json={
        "uris": ["gs://bucket/one.txt", "gs://bucket/two.txt"],
        "config": {"mime_type": "text/plain", "source": "file_source_registered"},
        "files": [
            {
                "display_name": "one-custom.txt",
                "custom_metadata": [{"key": "source", "stringValue": "uris"}],
                "state": "state_failed",
            },
            {"display_name": "two-custom.txt"},
        ],
    })
    assert official_registered.status_code == 200
    official_registered_files = official_registered.json()["files"]
    assert [item["uri"] for item in official_registered_files] == [
        "gs://bucket/one.txt",
        "gs://bucket/two.txt",
    ]
    assert all(item["source"] == "REGISTERED" for item in official_registered_files)
    assert all(item["mimeType"] == "text/plain" for item in official_registered_files)
    assert [item["displayName"] for item in official_registered_files] == ["one-custom.txt", "two-custom.txt"]
    assert [item["state"] for item in official_registered_files] == ["FAILED", "ACTIVE"]
    assert official_registered_files[0]["customMetadata"][0]["stringValue"] == "uris"

    config_registered = client.post("/v1beta/files:register", json={
        "config": {
            "uris": ["gs://bucket/config-one.txt", "gs://bucket/config-two.txt"],
            "mime_type": "text/plain",
            "files": [
                {"display_name": "config-one.txt"},
                {"display_name": "config-two.txt", "state": "file_state_processing"},
            ],
        },
    })
    assert config_registered.status_code == 200
    config_registered_files = config_registered.json()["files"]
    assert [item["uri"] for item in config_registered_files] == [
        "gs://bucket/config-one.txt",
        "gs://bucket/config-two.txt",
    ]
    assert [item["displayName"] for item in config_registered_files] == ["config-one.txt", "config-two.txt"]
    assert [item["state"] for item in config_registered_files] == ["ACTIVE", "PROCESSING"]

    video = client.post("/v1beta/files:register", json={
        "file": {
            "displayName": "clip.mp4",
            "mimeType": "video/mp4",
            "uri": "gs://bucket/clip.mp4",
            "video_metadata": {"video_duration": "3s"},
        }
    })
    assert video.status_code == 200
    assert video.json()["file"]["videoMetadata"]["videoDuration"] == "3s"

    config_file_registered = client.post("/v1beta/files:register", json={
        "config": {
            "mime_type": "text/plain",
            "state": "file_state_processing",
            "file": {
                "display_name": "registered-config-wrapper.txt",
                "uri": "gs://bucket/registered-config-wrapper.txt",
            },
        },
    })
    assert config_file_registered.status_code == 200
    assert config_file_registered.json()["file"]["displayName"] == "registered-config-wrapper.txt"
    assert config_file_registered.json()["file"]["mimeType"] == "text/plain"
    assert config_file_registered.json()["file"]["state"] == "PROCESSING"

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
    assert finished.headers["x-goog-upload-status"] == "final"
    assert finished.headers["x-goog-upload-size-received"] == str(len(b"resumable body"))
    file_resource = finished.json()["file"]
    assert file_resource["displayName"] == "resumable.txt"
    assert file_resource["mimeType"] == "text/plain"
    assert file_resource["expirationTime"]

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
    assert config_finished.headers["x-goog-upload-status"] == "final"
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
    assert query_finished.headers["x-goog-upload-status"] == "final"
    assert query_finished.json()["file"]["displayName"] == "query-resumable.txt"
    assert query_finished.json()["file"]["mimeType"] == "text/plain"

    sdk_prefixed_started = client.post(
        "/v1beta/upload/v1beta/files",
        json={"file": {"displayName": "sdk-prefixed.txt", "mimeType": "text/plain"}},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
        },
    )
    assert sdk_prefixed_started.status_code == 200
    sdk_session_path = "/" + sdk_prefixed_started.headers["x-goog-upload-url"].split("/", 3)[3]

    sdk_prefixed_finished = client.post(
        sdk_session_path.replace("/upload/v1beta/files/", "/v1beta/upload/v1beta/files/"),
        content=b"sdk prefixed",
        headers={"X-Goog-Upload-Command": "upload, finalize"},
    )
    assert sdk_prefixed_finished.status_code == 200
    assert sdk_prefixed_finished.headers["x-goog-upload-status"] == "final"
    assert sdk_prefixed_finished.json()["file"]["displayName"] == "sdk-prefixed.txt"


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
        "model": "models/gemini-3-flash-agent",
        "display_name": "Wrapped cache",
        "config": {
            "contents": "wrapped cached context",
            "system_instruction": "wrapped cached system",
            "tool_config": {"mode": "none"},
            "safety_settings": {
                "harm_category": "HARM_CATEGORY_HARASSMENT",
                "harm_block_threshold": "BLOCK_ONLY_HIGH",
            },
        },
        "cachedContent": {
            "ttl": "60s",
        }
    })
    assert wrapped_created.status_code == 200
    assert wrapped_created.json()["model"] == "models/gemini-3-flash-agent"
    assert wrapped_created.json()["displayName"] == "Wrapped cache"
    assert wrapped_created.json()["ttl"] == "60s"
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
    assert wrapped_created.json()["usageMetadata"]["totalTokenCount"] > created.json()["usageMetadata"]["totalTokenCount"]
    assert wrapped_created.json()["usageMetadata"]["promptTokensDetails"][0]["modality"] == "TEXT"
    assert wrapped_created.json()["expireTime"]

    expire_created = client.post("/v1beta/cachedContents", json={
        "model": "models/gemini-3-flash-agent",
        "contents": "expire alias cache",
        "expire_time": "2099-02-01T00:00:00Z",
    })
    assert expire_created.status_code == 200
    assert expire_created.json()["expireTime"] == "2099-02-01T00:00:00Z"
    assert "ttl" not in expire_created.json()

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
        "expire_time": "2099-01-01T00:00:00Z"
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

    resource_response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "cachedContent": {"name": cache_name},
        "contents": [{"role": "user", "parts": [{"text": "resource prompt"}]}],
    })

    assert resource_response.status_code == 200
    assert seen["request"]["systemInstruction"]["parts"][0]["text"] == "cached system"
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "cached context"
    assert seen["request"]["contents"][1]["parts"][0]["text"] == "resource prompt"
    assert "cachedContent" not in seen["request"]

    wrapped_response = client.post("/v1beta/models/gemini-3-flash-agent:generateContent", json={
        "cached_content": wrapped_created.json()["name"],
        "contents": [{"role": "user", "parts": [{"text": "wrapped prompt"}]}],
    })

    assert wrapped_response.status_code == 200
    assert seen["request"]["systemInstruction"]["parts"][0]["text"] == "wrapped cached system"
    assert seen["request"]["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}
    assert seen["request"]["safetySettings"] == [{
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH",
    }]
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "wrapped cached context"
    assert seen["request"]["contents"][1]["parts"][0]["text"] == "wrapped prompt"

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
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    client = TestClient(proxy.app)

    created = client.post("/v1/corpora", json={"displayName": "Knowledge"})
    assert created.status_code == 200
    corpus_name = created.json()["name"]
    corpus_id = corpus_name.split("/", 1)[1]

    listed_corpora = client.get("/v1/corpora")
    fetched_corpus = client.get(f"/v1/{corpus_name}")
    patched_corpus = client.patch(f"/v1/{corpus_name}?updateMask=corpus.displayName", json={
        "display_name": "Knowledge updated",
    })

    assert listed_corpora.status_code == 200
    assert listed_corpora.json()["corpora"][0]["name"] == corpus_name
    assert fetched_corpus.status_code == 200
    assert fetched_corpus.json()["displayName"] == "Knowledge"
    assert patched_corpus.status_code == 200
    assert patched_corpus.json()["displayName"] == "Knowledge updated"
    proxy._gemini_store_operation({
        "name": "operations/corpus-op",
        "metadata": {"corpus": corpus_name},
        "done": True,
        "response": {"corpus": corpus_name},
    })
    corpus_operation = client.get(f"/v1beta/corpora/{corpus_id}/operations/corpus-op")
    assert corpus_operation.status_code == 200
    assert corpus_operation.json()["metadata"]["corpus"] == corpus_name

    document = client.post(f"/v1/corpora/{corpus_id}/documents", json={
        "displayName": "Launch notes",
        "customMetadata": [
            {"key": "source", "stringValue": "initial"},
            {"key": "tags", "string_list_value": {"values": ["atlas", "launch"]}},
        ],
    })
    assert document.status_code == 200
    doc_name = document.json()["name"]
    doc_id = doc_name.rsplit("/", 1)[-1]

    listed_docs = client.get(f"/v1/corpora/{corpus_id}/documents")
    fetched_doc = client.get(f"/v1/{doc_name}")
    patched_doc = client.patch(f"/v1/{doc_name}?updateMask=document.displayName", json={
        "displayName": "Launch notes updated",
        "customMetadata": [{"key": "source", "stringValue": "ignored"}],
    })

    assert listed_docs.status_code == 200
    assert listed_docs.json()["documents"][0]["name"] == doc_name
    assert fetched_doc.status_code == 200
    assert fetched_doc.json()["displayName"] == "Launch notes"
    assert patched_doc.status_code == 200
    assert patched_doc.json()["displayName"] == "Launch notes updated"
    assert patched_doc.json()["customMetadata"][0]["stringValue"] == "initial"
    assert patched_doc.json()["customMetadata"][1]["stringListValue"]["values"] == ["atlas", "launch"]

    wrapped_doc = client.post(f"/v1/corpora/{corpus_id}/documents?document_id=wrapped_doc", json={
        "document": {
            "display_name": "Wrapped document",
            "custom_metadata": [{"key": "source", "stringValue": "wrapped"}],
        }
    })
    assert wrapped_doc.status_code == 200
    assert wrapped_doc.json()["name"].endswith("/documents/wrapped_doc")
    assert wrapped_doc.json()["displayName"] == "Wrapped document"
    assert wrapped_doc.json()["customMetadata"][0]["stringValue"] == "wrapped"

    chunk = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks", json={
        "data": {"stringValue": "Project Atlas launch window is October."},
        "customMetadata": [{"key": "topic", "stringValue": "launch"}],
    })
    assert chunk.status_code == 200
    chunk_id = chunk.json()["name"].rsplit("/", 1)[-1]

    wrapped_chunk = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks?chunk_id=wrapped_chunk", json={
        "chunk": {
            "data": {"string_value": "Wrapped chunk text"},
            "custom_metadata": [{"key": "kind", "stringValue": "wrapped"}],
        }
    })
    assert wrapped_chunk.status_code == 200
    assert wrapped_chunk.json()["name"].endswith("/chunks/wrapped_chunk")
    assert wrapped_chunk.json()["data"]["stringValue"] == "Wrapped chunk text"
    assert wrapped_chunk.json()["customMetadata"][0]["stringValue"] == "wrapped"

    batch_created = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks:batchCreate", json={
        "requests": [{
            "chunk_id": "batch_one",
            "chunk": {
                "data": {"stringValue": "Batch chunk text"},
                "custom_metadata": [{"key": "batch", "stringValue": "keep"}],
            },
        }],
    })
    batch_updated = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks:batchUpdate", json={
        "requests": [{
            "chunk": {
                "name": "batch_one",
                "data": {"stringValue": "Batch chunk updated"},
                "custom_metadata": [{"key": "batch", "stringValue": "ignored"}],
            },
            "update_mask": "chunk.data",
        }],
    })

    queried = client.post(f"/v1/corpora/{corpus_id}:query", json={"query": "Atlas October"})
    doc_queried = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}:query", json={"query": "launch"})
    listed_chunks = client.get(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks")
    fetched_chunk = client.get(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks/{chunk_id}")
    patched_chunk = client.patch(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks/{chunk_id}?updateMask=chunk.data", json={
        "data": {"stringValue": "Project Atlas moved to November."},
        "customMetadata": [{"key": "topic", "stringValue": "ignored"}],
    })

    assert batch_created.status_code == 200
    assert batch_created.json()["chunks"][0]["name"].endswith("/chunks/batch_one")
    assert batch_created.json()["chunks"][0]["customMetadata"][0]["stringValue"] == "keep"
    assert batch_updated.status_code == 200
    assert batch_updated.json()["chunks"][0]["data"]["stringValue"] == "Batch chunk updated"
    assert batch_updated.json()["chunks"][0]["customMetadata"][0]["stringValue"] == "keep"
    assert queried.status_code == 200
    assert queried.json()["relevantChunks"][0]["chunk"]["name"] == chunk.json()["name"]
    assert doc_queried.status_code == 200
    assert chunk.json()["name"] in {item["name"] for item in listed_chunks.json()["chunks"]}
    assert fetched_chunk.json()["data"]["stringValue"].endswith("October.")
    assert patched_chunk.json()["data"]["stringValue"].endswith("November.")
    assert patched_chunk.json()["customMetadata"][0]["stringValue"] == "launch"

    batch_deleted = client.post(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks:batchDelete", json={
        "names": [f"{doc_name}/chunks/batch_one"],
    })
    assert batch_deleted.status_code == 200

    perm = client.post(f"/v1/corpora/{corpus_id}/permissions", json={
        "permission": {
            "email_address": "reader@example.com",
            "role": "reader",
            "grantee_type": "user",
        },
    })
    second_perm = client.post(f"/v1/corpora/{corpus_id}/permissions", json={
        "permission": {
            "email_address": "writer@example.com",
            "role": "writer",
            "grantee_type": "user",
        },
    })
    perm_id = perm.json()["name"].rsplit("/", 1)[-1]
    listed_perms = client.get(f"/v1/corpora/{corpus_id}/permissions?pageSize=1")
    listed_perms_next = client.get(
        f"/v1/corpora/{corpus_id}/permissions?page_size=1&page_token={listed_perms.json()['nextPageToken']}"
    )
    fetched_perm = client.get(f"/v1/corpora/{corpus_id}/permissions/{perm_id}")
    patched_perm = client.patch(f"/v1/corpora/{corpus_id}/permissions/{perm_id}?updateMask=permission.role", json={
        "permission": {
            "role": "writer",
            "email_address": "ignored@example.com",
        }
    })
    snake_query_perm = client.patch(f"/v1/corpora/{corpus_id}/permissions/{perm_id}?update_mask=permission.role", json={
        "permission": {
            "role": "reader",
            "email_address": "ignored@example.com",
        }
    })

    assert perm.status_code == 200
    assert second_perm.status_code == 200
    assert perm.json()["emailAddress"] == "reader@example.com"
    assert perm.json()["granteeType"] == "USER"
    assert listed_perms.status_code == 200
    assert len(listed_perms.json()["permissions"]) == 1
    assert listed_perms.json()["nextPageToken"] == "1"
    assert listed_perms_next.status_code == 200
    assert len(listed_perms_next.json()["permissions"]) == 1
    assert listed_perms_next.json()["nextPageToken"] == ""
    assert fetched_perm.json()["role"] == "READER"
    assert patched_perm.json()["role"] == "WRITER"
    assert patched_perm.json()["emailAddress"] == "reader@example.com"
    assert snake_query_perm.status_code == 200
    assert snake_query_perm.json()["role"] == "READER"
    assert snake_query_perm.json()["emailAddress"] == "reader@example.com"

    assert client.delete(f"/v1/corpora/{corpus_id}/documents/{doc_id}/chunks/{chunk_id}").status_code == 200
    assert client.delete(f"/v1/corpora/{corpus_id}/permissions/{perm_id}").status_code == 200
    assert client.delete(f"/v1/corpora/{corpus_id}/documents/{doc_id}").status_code == 200
    assert client.delete(f"/v1/{wrapped_doc.json()['name']}").status_code == 200
    assert client.delete(f"/v1/{corpus_name}").status_code == 200


def test_gemini_corpus_accepts_resource_wrappers(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_CORPORA_DIR", str(tmp_path / "corpora"))
    client = TestClient(proxy.app)

    created = client.post("/v1beta/corpora", json={
        "corpus": {
            "display_name": "Wrapped Knowledge",
        }
    })
    assert created.status_code == 200
    corpus_name = created.json()["name"]
    assert created.json()["displayName"] == "Wrapped Knowledge"

    patched = client.patch(f"/v1beta/{corpus_name}?update_mask=corpus.display_name", json={
        "corpus": {
            "display_name": "Wrapped Knowledge Updated",
        }
    })
    fetched = client.get(f"/v1beta/{corpus_name}")

    assert patched.status_code == 200
    assert patched.json()["displayName"] == "Wrapped Knowledge Updated"
    assert fetched.status_code == 200
    assert fetched.json()["displayName"] == "Wrapped Knowledge Updated"


def test_gemini_corpus_delete_requires_force_for_documents(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_CORPORA_DIR", str(tmp_path / "corpora"))
    client = TestClient(proxy.app)

    corpus = client.post("/v1beta/corpora", json={"displayName": "Force corpus"}).json()
    corpus_id = corpus["name"].split("/", 1)[1]
    document = client.post(f"/v1beta/corpora/{corpus_id}/documents", json={
        "displayName": "Blocking doc",
    })

    blocked = client.delete(f"/v1beta/{corpus['name']}")
    forced = client.delete(f"/v1beta/{corpus['name']}?force=true")

    assert document.status_code == 200
    assert blocked.status_code == 400
    assert blocked.json()["error"]["status"] == "FAILED_PRECONDITION"
    assert blocked.json()["error"]["details"][0]["fieldViolations"][0]["field"] == "force"
    assert forced.status_code == 200


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

    wrapped_created = client.post("/v1beta/fileSearchStores", json={
        "file_search_store": {
            "display_name": "wrapped notes",
            "embedding_model": "models/text-embedding-004",
            "chunking_config": {
                "white_space_config": {
                    "max_tokens_per_chunk": "128",
                    "max_overlap_tokens": "16",
                }
            },
        },
        "custom_metadata": [
            {"key": "wrapped", "string_value": "yes"},
            {"key": "tags", "string_list_value": {"values": ["notes", "wrapped"]}},
        ],
    })
    assert wrapped_created.status_code == 200
    assert wrapped_created.json()["displayName"] == "wrapped notes"
    assert wrapped_created.json()["embeddingModel"] == "models/text-embedding-004"
    assert wrapped_created.json()["customMetadata"][0]["stringValue"] == "yes"
    assert wrapped_created.json()["customMetadata"][1]["stringListValue"]["values"] == ["notes", "wrapped"]
    assert wrapped_created.json()["chunkingConfig"]["whiteSpaceConfig"]["maxTokensPerChunk"] == 128
    assert wrapped_created.json()["chunkingConfig"]["whiteSpaceConfig"]["maxOverlapTokens"] == 16

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
            "chunking_config": {"white_space_config": {}},
        }
    })
    assert imported.status_code == 200
    imported_doc = imported.json()["response"]["document"]
    assert imported_doc["displayName"] == "source import"
    assert imported_doc["customMetadata"][0]["stringValue"] == "files-api"
    assert imported_doc["chunkingConfig"] == {"whiteSpaceConfig": {}}

    wrapped_import = client.post(f"/v1beta/fileSearchStores/{store_id}:importFile", json={
        "file_metadata": {
            "file_name": uploaded_file["name"],
            "display_name": "wrapped source import",
            "custom_metadata": [{"key": "wrapped-import", "string_value": "yes"}],
            "chunking_config": {
                "white_space_config": {
                    "max_tokens_per_chunk": "64",
                    "max_overlap_tokens": "8",
                }
            },
        }
    })
    assert wrapped_import.status_code == 200
    wrapped_import_doc = wrapped_import.json()["response"]["document"]
    assert wrapped_import_doc["displayName"] == "wrapped source import"
    assert wrapped_import_doc["customMetadata"][0]["stringValue"] == "yes"
    assert wrapped_import_doc["chunkingConfig"]["whiteSpaceConfig"]["maxTokensPerChunk"] == 64
    assert wrapped_import_doc["chunkingConfig"]["whiteSpaceConfig"]["maxOverlapTokens"] == 8

    uploaded_doc = client.post(
        f"/upload/v1/fileSearchStores/{store_id}:uploadToFileSearchStore?displayName=direct.txt",
        json={
            "file": {
                "displayName": "direct.txt",
                "mimeType": "text/plain",
                "custom_metadata": [{"key": "source", "stringValue": "direct"}],
                "chunking_config": {"white_space_config": {}},
            },
            "content": "direct document",
        },
    )
    assert uploaded_doc.status_code == 200
    uploaded_doc_resource = uploaded_doc.json()["response"]["document"]
    assert uploaded_doc_resource["customMetadata"][0]["stringValue"] == "direct"
    assert uploaded_doc_resource["chunkingConfig"] == {"whiteSpaceConfig": {}}

    wrapped_uploaded_doc = client.post(
        f"/v1beta/fileSearchStores/{store_id}:uploadToFileSearchStore",
        json={
            "fileMetadata": {
                "display_name": "wrapped direct.txt",
                "mime_type": "text/plain",
                "custom_metadata": [{"key": "wrapped-upload", "string_value": "yes"}],
                "chunking_config": {"white_space_config": {"max_tokens_per_chunk": "32"}},
            },
            "text": "wrapped direct document",
        },
    )
    assert wrapped_uploaded_doc.status_code == 200
    wrapped_uploaded_doc_resource = wrapped_uploaded_doc.json()["response"]["document"]
    assert wrapped_uploaded_doc_resource["displayName"] == "wrapped direct.txt"
    assert wrapped_uploaded_doc_resource["customMetadata"][0]["stringValue"] == "yes"
    assert wrapped_uploaded_doc_resource["chunkingConfig"]["whiteSpaceConfig"]["maxTokensPerChunk"] == 32

    listed_stores = client.get("/v1/fileSearchStores")
    fetched_store = client.get(f"/v1/{store_name}")
    listed = client.get(f"/v1/fileSearchStores/{store_id}/documents")
    listed_snake_page = client.get(f"/v1/fileSearchStores/{store_id}/documents?page_size=1&page_token=1")
    assert listed_stores.status_code == 200
    assert store_name in {item["name"] for item in listed_stores.json()["fileSearchStores"]}
    assert fetched_store.status_code == 200
    assert fetched_store.json()["name"] == store_name
    assert fetched_store.json()["activeDocumentsCount"] == 4
    assert fetched_store.json()["pendingDocumentsCount"] == 0
    assert fetched_store.json()["failedDocumentsCount"] == 0
    assert int(fetched_store.json()["sizeBytes"]) == (
        len(b"source document")
        + len(b"source document")
        + len(b"direct document")
        + len(b"wrapped direct document")
    )
    assert listed.status_code == 200
    assert len(listed.json()["documents"]) == 4
    assert listed_snake_page.status_code == 200
    assert len(listed_snake_page.json()["documents"]) == 1
    assert listed_snake_page.json()["nextPageToken"] == "2"

    fetched = client.get(f"/v1/{imported_doc['name']}")
    assert fetched.status_code == 200
    assert fetched.json()["displayName"] == "source import"
    assert fetched.json()["customMetadata"][0]["key"] == "source"
    assert fetched.json()["chunkingConfig"] == {"whiteSpaceConfig": {}}

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
    v1beta_upload_operation = client.get(f"/v1beta/fileSearchStores/{store_id}/upload/operations/{uploaded_op_id}")
    v1beta_waited_upload_operation = client.post(f"/v1beta/fileSearchStores/{store_id}/upload/operations/{uploaded_op_id}:wait")
    media = client.get(f"/v1/fileSearchStores/{store_id}/media/{imported_doc['name'].rsplit('/', 1)[-1]}")
    assert nested_operation.status_code == 200
    assert nested_operation.json()["name"] == imported.json()["name"]
    assert nested_operations.status_code == 200
    assert imported.json()["name"] in {item["name"] for item in nested_operations.json()["operations"]}
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
    assert v1beta_upload_operation.status_code == 200
    assert v1beta_upload_operation.json()["name"] == uploaded_doc.json()["name"]
    assert v1beta_upload_operation.json()["done"] is True
    assert v1beta_upload_operation.json()["metadata"]["fileSearchStore"] == store_name
    assert v1beta_upload_operation.json()["response"]["document"]["name"] == uploaded_doc_resource["name"]
    assert v1beta_waited_upload_operation.status_code == 200
    assert v1beta_waited_upload_operation.json()["name"] == uploaded_doc.json()["name"]
    assert media.status_code == 200
    assert media.content == b"source document"

    deleted_upload_operation = client.delete(f"/v1/fileSearchStores/{store_id}/upload/operations/{uploaded_op_id}")
    blocked_doc_delete = client.delete(f"/v1/{imported_doc['name']}")
    deleted_doc = client.delete(f"/v1/{imported_doc['name']}?force=true")
    blocked_store_delete = client.delete(f"/v1/{store_name}")
    forced_store_delete = client.delete(f"/v1/{store_name}?force=true")

    assert deleted_upload_operation.status_code == 200
    assert blocked_doc_delete.status_code == 400
    assert blocked_doc_delete.json()["error"]["status"] == "FAILED_PRECONDITION"
    assert blocked_doc_delete.json()["error"]["details"][0]["fieldViolations"][0]["field"] == "force"
    assert deleted_doc.status_code == 200
    assert blocked_store_delete.status_code == 400
    assert blocked_store_delete.json()["error"]["status"] == "FAILED_PRECONDITION"
    assert blocked_store_delete.json()["error"]["details"][0]["fieldViolations"][0]["field"] == "force"
    assert forced_store_delete.status_code == 200


def test_gemini_file_search_documents_list_clamps_page_size(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_FILE_SEARCH_STORES_DIR", str(tmp_path / "fss"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    client = TestClient(proxy.app)

    store = client.post("/v1beta/fileSearchStores", json={"displayName": "many docs"}).json()
    store_id = store["name"].split("/", 1)[1]
    for index in range(12):
        created = client.post("/v1beta/fileSearchStores", json={"displayName": f"store-{index:02d}"})
        assert created.status_code == 200
    for index in range(25):
        uploaded = client.post(
            f"/upload/v1beta/fileSearchStores/{store_id}:uploadToFileSearchStore?displayName=doc-{index}.txt",
            json={"content": f"doc {index}"},
        )
        assert uploaded.status_code == 200

    stores_default = client.get("/v1beta/fileSearchStores")
    stores_next = client.get(f"/v1beta/fileSearchStores?pageToken={stores_default.json()['nextPageToken']}")
    stores_oversized = client.get("/v1beta/fileSearchStores?pageSize=999")
    docs_default = client.get(f"/v1beta/fileSearchStores/{store_id}/documents")
    listed = client.get(f"/v1beta/fileSearchStores/{store_id}/documents?pageSize=999")
    next_page = client.get(
        f"/v1beta/fileSearchStores/{store_id}/documents?page_size=999&page_token={listed.json()['nextPageToken']}"
    )

    assert stores_default.status_code == 200
    assert len(stores_default.json()["fileSearchStores"]) == 10
    assert stores_default.json()["nextPageToken"] == "10"
    assert stores_next.status_code == 200
    assert len(stores_next.json()["fileSearchStores"]) == 3
    assert stores_oversized.status_code == 200
    assert len(stores_oversized.json()["fileSearchStores"]) == 13
    assert docs_default.status_code == 200
    assert len(docs_default.json()["documents"]) == 10
    assert docs_default.json()["nextPageToken"] == "10"
    assert listed.status_code == 200
    assert len(listed.json()["documents"]) == 20
    assert listed.json()["nextPageToken"] == "20"
    assert next_page.status_code == 200
    assert len(next_page.json()["documents"]) == 5
    assert next_page.json()["nextPageToken"] == ""


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
        "tools": [{
            "file_search": {
                "file_search_store_names": [store["name"]],
                "metadata_filter": 'document.custom_metadata.project="atlas"',
                "top_k": "3",
            }
        }],
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
        "tuned_model_id": "my_tuned",
        "config": {
            "temperature": "0.4",
            "top_k": "32",
            "reader_project_numbers": ["123"],
        },
        "tunedModel": {
            "displayName": "My tuned",
            "base_model": "models/gemini-3-flash-agent",
            "tuning_task": {
                "hyperparameters": {
                    "epochCount": 2,
                    "batchSize": 4,
                    "learning_rate": "0.001",
                    "learning_rate_multiplier": "1.5",
                },
                "training_data": {"examples": {"examples": [{"text_input": "hi", "output": "hello"}]}},
            },
        },
    })
    assert created.status_code == 200
    tuned = created.json()["response"]
    assert tuned["name"] == "tunedModels/my_tuned"
    assert tuned["baseModel"] == "models/gemini-3-flash-agent"
    assert tuned["temperature"] == 0.4
    assert tuned["topK"] == 32
    assert tuned["readerProjectNumbers"] == ["123"]
    assert tuned["tuningTask"]["hyperparameters"]["epochCount"] == 2
    assert tuned["tuningTask"]["hyperparameters"]["learningRate"] == "0.001"
    assert tuned["tuningTask"]["hyperparameters"]["learningRateMultiplier"] == "1.5"
    assert tuned["tuningTask"]["trainingData"]["examples"]["examples"][0]["textInput"] == "hi"
    assert tuned["tuningTask"]["trainingData"]["examples"]["examples"][0]["output"] == "hello"
    created_op_id = created.json()["name"].split("/", 1)[1]
    query_created = client.post("/v1beta/tunedModels?tunedModelId=query_tuned", json={
        "tunedModel": {
            "displayName": "Query tuned",
            "baseModel": "models/gemini-3-flash-agent",
        },
    })

    listed = client.get("/v1/tunedModels")
    oversized_page = client.get("/v1beta/tunedModels?pageSize=1001")
    filtered = client.get('/v1beta/tunedModels?filter=displayName:"Query"')
    filtered_description = client.get('/v1beta/tunedModels?filter=description:"updated"')
    fetched = client.get("/v1/tunedModels/my_tuned")
    fetched_full_name = client.get("/v1beta/tunedModels/tunedModels/my_tuned")
    listed_operations = client.get("/v1/tunedModels/my_tuned/operations")
    filtered_operations = client.get(
        f"/v1beta/tunedModels/my_tuned/operations?filter=operation.name:{created_op_id}&return_partial_success=true"
    )
    fetched_operation = client.get(f"/v1/tunedModels/my_tuned/operations/{created_op_id}")
    waited_operation = client.post(f"/v1/tunedModels/my_tuned/operations/{created_op_id}:wait")
    cancelled_operation = client.post(f"/v1/tunedModels/my_tuned/operations/{created_op_id}:cancel")
    patched = client.patch("/v1/tunedModels/my_tuned", json={
        "tuned_model": {
            "description": "updated",
            "tuning_task": {"hyperparameters": {"epoch_count": 3}},
        }
    })
    assert listed.status_code == 200
    assert oversized_page.status_code == 200
    assert {item["name"] for item in oversized_page.json()["tunedModels"]} == {
        "tunedModels/my_tuned",
        "tunedModels/query_tuned",
    }
    assert query_created.status_code == 200
    assert query_created.json()["response"]["name"] == "tunedModels/query_tuned"
    assert filtered.status_code == 200
    assert [item["name"] for item in filtered.json()["tunedModels"]] == ["tunedModels/query_tuned"]
    assert filtered_description.status_code == 200
    assert filtered_description.json()["tunedModels"] == []
    assert fetched.json()["displayName"] == "My tuned"
    assert fetched_full_name.status_code == 200
    assert fetched_full_name.json()["name"] == "tunedModels/my_tuned"
    assert fetched.json()["supportedGenerationMethods"] == [
        "generateContent",
        "streamGenerateContent",
        "generateText",
        "batchGenerateContent",
        "countTokens",
        "computeTokens",
        "embedContent",
        "batchEmbedContents",
        "asyncBatchEmbedContent",
    ]
    assert listed_operations.status_code == 200
    assert listed_operations.json()["operations"][0]["name"] == created.json()["name"]
    assert filtered_operations.status_code == 200
    assert filtered_operations.json()["operations"][0]["name"] == created.json()["name"]
    assert filtered_operations.json()["unreachable"] == []
    assert fetched_operation.status_code == 200
    assert fetched_operation.json()["name"] == created.json()["name"]
    assert waited_operation.status_code == 200
    assert waited_operation.json()["name"] == created.json()["name"]
    assert cancelled_operation.status_code == 200
    assert cancelled_operation.json() == {}
    assert patched.json()["description"] == "updated"
    assert patched.json()["tuningTask"]["hyperparameters"]["epochCount"] == 3
    masked_description = client.patch(
        "/v1/tunedModels/my_tuned?updateMask=tunedModel.description",
        json={
            "tuned_model": {
                "description": "masked update",
                "display_name": "Ignored display",
                "tuning_task": {"hyperparameters": {"epoch_count": 7}},
            }
        },
    )
    masked_reader_projects = client.patch("/v1/tunedModels/my_tuned", json={
        "tuned_model": {
            "reader_project_numbers": ["456", "789"],
            "description": "ignored by body mask",
        },
        "update_mask": "reader_project_numbers",
    })
    unsupported_mask = client.patch("/v1/tunedModels/my_tuned?updateMask=unsupportedField", json={
        "unsupportedField": "value",
    })
    assert masked_description.status_code == 200
    assert masked_description.json()["description"] == "masked update"
    assert masked_description.json()["displayName"] == "My tuned"
    assert masked_description.json()["tuningTask"]["hyperparameters"]["epochCount"] == 3
    assert masked_reader_projects.status_code == 200
    assert masked_reader_projects.json()["readerProjectNumbers"] == ["456", "789"]
    assert masked_reader_projects.json()["description"] == "masked update"
    assert unsupported_mask.status_code == 400

    perm = client.post("/v1/tunedModels/my_tuned/permissions", json={
        "permission": {
            "email_address": "user@example.com",
            "role": "reader",
            "grantee_type": "user",
        },
    })
    assert perm.status_code == 200
    perm_id = perm.json()["name"].rsplit("/", 1)[-1]
    listed_perms = client.get("/v1/tunedModels/my_tuned/permissions")
    listed_perms_full_name = client.get("/v1beta/tunedModels/tunedModels/my_tuned/permissions")
    patched_perm = client.patch(f"/v1/tunedModels/my_tuned/permissions/{perm_id}", json={
        "permission": {
            "role": "writer",
            "email_address": "ignored@example.com",
        },
        "update_mask": "permission.role",
    })
    snake_query_perm = client.patch(f"/v1/tunedModels/my_tuned/permissions/{perm_id}?update_mask=permission.role", json={
        "permission": {
            "role": "writer",
            "email_address": "still-ignored@example.com",
        }
    })
    full_name_perm_patch = client.patch(
        f"/v1beta/tunedModels/tunedModels/my_tuned/permissions/{perm_id}?updateMask=permission.role",
        json={"permission": {"role": "reader", "email_address": "full-ignored@example.com"}},
    )
    promoted = client.post(f"/v1/tunedModels/my_tuned/permissions/{perm_id}:transferOwnership")
    fetched_perm = client.get(f"/v1/tunedModels/my_tuned/permissions/{perm.json()['name']}")
    assert listed_perms.status_code == 200
    assert listed_perms.json()["permissions"][0]["emailAddress"] == "user@example.com"
    assert listed_perms.json()["permissions"][0]["granteeType"] == "USER"
    assert listed_perms.json()["permissions"][0]["role"] == "READER"
    assert listed_perms_full_name.status_code == 200
    assert listed_perms_full_name.json()["permissions"][0]["name"] == perm.json()["name"]
    assert patched_perm.status_code == 200
    assert patched_perm.json()["role"] == "WRITER"
    assert patched_perm.json()["emailAddress"] == "user@example.com"
    assert snake_query_perm.status_code == 200
    assert snake_query_perm.json()["role"] == "WRITER"
    assert snake_query_perm.json()["emailAddress"] == "user@example.com"
    assert full_name_perm_patch.status_code == 200
    assert full_name_perm_patch.json()["role"] == "READER"
    assert promoted.status_code == 200
    assert fetched_perm.json()["role"] == "OWNER"

    model_owner_transfer = client.post("/v1beta/tunedModels/my_tuned:transferOwnership", json={
        "email_address": "new-owner@example.com",
    })
    owner_perms = client.get("/v1/tunedModels/my_tuned/permissions")
    paged_owner_perms = client.get("/v1beta/tunedModels/my_tuned/permissions?pageSize=1")
    paged_owner_perms_next = client.get(
        f"/v1beta/tunedModels/my_tuned/permissions?page_size=1&page_token={paged_owner_perms.json()['nextPageToken']}"
    )
    assert model_owner_transfer.status_code == 200
    assert model_owner_transfer.json()["owner"] == "new-owner@example.com"
    assert any(
        item["role"] == "OWNER" and item["emailAddress"] == "new-owner@example.com"
        for item in owner_perms.json()["permissions"]
    )
    assert any(
        item["role"] == "WRITER" and item["emailAddress"] == "user@example.com"
        for item in owner_perms.json()["permissions"]
    )
    assert len(paged_owner_perms.json()["permissions"]) == 1
    assert paged_owner_perms.json()["nextPageToken"] == "1"
    assert len(paged_owner_perms_next.json()["permissions"]) == 1
    assert paged_owner_perms_next.json()["nextPageToken"] == ""

    generated = client.post("/v1/tunedModels/my_tuned:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "hello tuned"}]}],
        "provider_options": {
            "google": {
                "system_instruction": "tuned provider system",
                "max_output_tokens": "13",
                "tool_config": {"function_calling_config": {"mode": "none"}},
            }
        },
        "config": {
            "response_mime_type": "text/plain",
        },
        "processing_options": {"media_resolution": "MEDIA_RESOLUTION_LOW"},
    })
    assert generated.status_code == 200
    assert seen["model"] == "gemini-3-flash-agent"
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "hello tuned"
    assert seen["request"]["systemInstruction"]["parts"][0]["text"] == "tuned provider system"
    assert seen["request"]["generationConfig"]["maxOutputTokens"] == 13
    assert seen["request"]["generationConfig"]["responseMimeType"] == "text/plain"
    assert seen["request"]["toolConfig"]["functionCallingConfig"] == {"mode": "NONE"}
    assert "processingOptions" not in seen["request"]
    generated_full_name = client.post("/v1beta/tunedModels/tunedModels/my_tuned:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "hello tuned full"}]}],
    })
    assert generated_full_name.status_code == 200
    assert generated_full_name.json()["candidates"][0]["content"]["parts"][0]["text"] == "tuned ok"

    text_generated = client.post("/v1beta/tunedModels/my_tuned:generateText", json={
        "prompt": {"text": "legacy tuned text"},
        "temperature": "0.1",
    })
    assert text_generated.status_code == 200
    assert text_generated.json()["modelVersion"] == "tunedModels/my_tuned"
    assert seen["request"]["contents"][0]["parts"][0]["text"] == "legacy tuned text"
    assert seen["request"]["generationConfig"]["temperature"] == 0.1

    with client.stream("POST", "/v1beta/tunedModels/my_tuned:streamGenerateContent", json={
        "contents": "hello tuned stream",
    }) as streamed:
        stream_body = streamed.read().decode()
    assert streamed.status_code == 200
    assert "data:" in stream_body
    assert "tuned ok" in stream_body

    tuned_batch = client.post("/v1beta/tunedModels/my_tuned:batchGenerateContent", json={
        "display_name": "tuned batch",
        "requests": [
            {"contents": [{"role": "user", "parts": [{"text": "batch tuned"}]}]},
        ],
    })
    tuned_async_embed = client.post("/v1/tunedModels/my_tuned:asyncBatchEmbedContent", json={
        "embed_content_batch": {
            "display_name": "tuned async embed",
            "requests": [
                {"content": {"parts": [{"text": "embed tuned"}]}},
            ],
        },
    })
    assert tuned_batch.status_code == 200
    assert tuned_batch.json()["metadata"]["batchResource"]["displayName"] == "tuned batch"
    assert tuned_batch.json()["response"]["responses"][0]["candidates"][0]["content"]["parts"][0]["text"] == "tuned ok"
    assert tuned_async_embed.status_code == 200
    assert tuned_async_embed.json()["metadata"]["batchResource"]["displayName"] == "tuned async embed"
    assert tuned_async_embed.json()["response"]["embeddings"][0]["values"]

    counted = client.post("/v1/tunedModels/my_tuned:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "hello tuned"}]}]
    })
    counted_full_name = client.post("/v1beta/tunedModels/tunedModels/my_tuned:countTokens", json={
        "contents": [{"role": "user", "parts": [{"text": "hello tuned full"}]}]
    })
    assert counted.status_code == 200
    assert counted.json()["totalTokens"] > 0
    assert counted.json()["promptTokensDetails"][0]["modality"] == "TEXT"
    assert counted.json()["cacheTokensDetails"] == []
    assert counted_full_name.status_code == 200
    assert counted_full_name.json()["totalTokens"] > 0

    computed = client.post("/v1beta/tunedModels/my_tuned:computeTokens", json={
        "contents": "hello tuned compute",
    })
    assert computed.status_code == 200
    assert computed.json()["tokensInfo"][0]["role"] == "user"
    assert computed.json()["tokensInfo"][0]["tokenIds"]

    embedded = client.post("/v1/tunedModels/my_tuned:embedContent", json={
        "content": {"parts": [{"text": "embed tuned"}]},
        "config": {"output_dimensionality": 8},
    })
    batch_embedded = client.post("/v1beta/tunedModels/my_tuned:batchEmbedContents", json={
        "requests": [
            {"content": {"parts": [{"text": "first tuned"}]}},
            {"content": {"parts": [{"text": "second tuned"}]}},
        ],
        "config": {"output_dimensionality": 6},
    })
    assert embedded.status_code == 200
    assert len(embedded.json()["embedding"]["values"]) == 8
    assert batch_embedded.status_code == 200
    assert len(batch_embedded.json()["embeddings"]) == 2
    assert len(batch_embedded.json()["embeddings"][0]["values"]) == 6

    deleted_perm = client.delete(f"/v1/tunedModels/my_tuned/permissions/{perm_id}")
    deleted_operation = client.delete(f"/v1/tunedModels/my_tuned/operations/{created_op_id}")
    deleted_model = client.delete("/v1/tunedModels/my_tuned")
    assert deleted_perm.status_code == 200
    assert deleted_operation.status_code == 200
    assert deleted_model.status_code == 200


def test_openai_image_generation_endpoint_is_removed():
    client = TestClient(proxy.app)

    created = client.post("/v1/images/generations", json={"prompt": "draw a square"})

    assert created.status_code == 404
    assert created.json()["error"]["status"] == "NOT_FOUND"
    assert "OpenAI-compatible endpoints have been removed" in created.json()["error"]["message"]


def test_gemini_image_model_generate_content_predict_and_generate_images(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_GENERATED_FILES_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("ANTIGRAVITY_GEMINI_OPERATIONS_DIR", str(tmp_path / "ops"))
    image_calls = []

    class FakeClient:
        def generate_image(self, *, prompt, output_dir, aspect_ratio="", image_size=""):
            image_calls.append({"prompt": prompt, "aspect_ratio": aspect_ratio, "image_size": image_size})
            output = output_dir / "gemini-image.png"
            output.write_bytes(b"gemini-image")
            return output

    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    client = TestClient(proxy.app)

    content = client.post("/v1beta/models/gemini-image-latest:generateContent", json={
        "contents": [{"role": "user", "parts": [{"text": "draw image"}]}],
        "generation_config": {"image_config": {"aspect_ratio": "16:9", "image_size": "2K"}},
    })
    predict = client.post("/v1beta/models/gemini-image-latest:predict", json={
        "instances": [{"prompt": "draw predict"}],
        "parameters": {"aspect_ratio": "1:1", "image_size": "1K"},
    })
    generated = client.post("/v1beta/models/gemini-image-latest:generateImages", json={
        "prompt": "draw generated",
        "config": {"aspect_ratio": "9:16", "image_size": "2K", "number_of_images": "2"},
    })
    config_prompt = client.post("/v1beta/models/gemini-image-latest:generateImages", json={
        "config": {
            "prompt": "draw config prompt",
            "image_config": {"aspect_ratio": "4:3", "image_size": "1K"},
        },
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
    assert len(generated.json()["generatedImages"]) == 2
    generated_image = generated.json()["generatedImages"][0]
    assert generated_image["image"]["imageBytes"]
    assert generated_image["generatedFile"]["name"].startswith("generatedFiles/")
    assert config_prompt.status_code == 200
    assert config_prompt.json()["generatedImages"][0]["image"]["imageBytes"]
    assert image_calls == [
        {"prompt": "draw image", "aspect_ratio": "16:9", "image_size": "2K"},
        {"prompt": "draw predict", "aspect_ratio": "1:1", "image_size": "1K"},
        {"prompt": "draw generated", "aspect_ratio": "9:16", "image_size": "2K"},
        {"prompt": "draw generated", "aspect_ratio": "9:16", "image_size": "2K"},
        {"prompt": "draw config prompt", "aspect_ratio": "4:3", "image_size": "1K"},
    ]

    listed = client.get("/v1beta/generatedFiles")
    assert listed.status_code == 200
    assert len(listed.json()["generatedFiles"]) == 5

    generated_file_name = generated_image["generatedFile"]["name"]
    fetched_file = client.get(f"/v1beta/{generated_file_name}")
    downloaded_file = client.get(f"/v1beta/{generated_file_name}:download")
    listed_operations = client.get("/v1beta/generatedFiles/operations")
    assert fetched_file.status_code == 200
    assert fetched_file.json()["name"] == generated_file_name
    assert fetched_file.json()["downloadUri"].endswith(":download")
    assert downloaded_file.status_code == 200
    assert downloaded_file.content == b"gemini-image"
    assert downloaded_file.headers["content-type"].startswith("image/png")
    assert listed_operations.status_code == 200
    operations = listed_operations.json()["operations"]
    operation = next(item for item in operations if item["metadata"]["generatedFile"] == generated_file_name)
    operation_id = operation["name"].split("/", 1)[1]

    fetched_operation = client.get(f"/v1beta/generatedFiles/operations/{operation_id}")
    fetched_scoped_operation = client.get(f"/v1beta/{generated_file_name}/operations/{operation_id}")
    waited_operation = client.post(f"/v1beta/generatedFiles/operations/{operation_id}:wait")
    cancelled_operation = client.post(f"/v1beta/generatedFiles/operations/{operation_id}:cancel")
    deleted_operation = client.delete(f"/v1beta/generatedFiles/operations/{operation_id}")
    missing_operation = client.get(f"/v1beta/generatedFiles/operations/{operation_id}")
    deleted_file = client.delete(f"/v1beta/{generated_file_name}")
    missing_file = client.get(f"/v1beta/{generated_file_name}")

    assert fetched_operation.status_code == 200
    assert fetched_operation.json()["name"] == operation["name"]
    assert fetched_operation.json()["done"] is True
    assert fetched_operation.json()["response"]["name"] == generated_file_name
    assert fetched_scoped_operation.status_code == 200
    assert fetched_scoped_operation.json()["name"] == operation["name"]
    assert waited_operation.status_code == 200
    assert waited_operation.json()["name"] == operation["name"]
    assert cancelled_operation.status_code == 200
    assert cancelled_operation.json() == {}
    assert deleted_operation.status_code == 200
    assert missing_operation.status_code == 404
    assert deleted_file.status_code == 200
    assert missing_file.status_code == 404


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
