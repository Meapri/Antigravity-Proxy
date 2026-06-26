import json

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import antigravity_proxy as proxy


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_RESPONSES_DB", str(tmp_path / "responses.sqlite3"))
    return TestClient(proxy.app)


def _chat_payload(text="hello", tool_calls=None):
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1,
        "model": "Gemini 3.5 Flash (High)",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }


def test_responses_create_string_input_persists_and_retrieves(tmp_path, monkeypatch):
    async def fake_chat(req):
        assert req.messages[-1].role == "user"
        assert req.messages[-1].content == "hello"
        return JSONResponse(_chat_payload("hi there"))

    monkeypatch.setattr(proxy, "chat_completions", fake_chat)
    client = _client(tmp_path, monkeypatch)

    created = client.post("/v1/responses", json={"model": "Gemini 3.5 Flash (High)", "input": "hello"}).json()
    retrieved = client.get(f"/v1/responses/{created['id']}").json()

    assert created["object"] == "response"
    assert created["status"] == "completed"
    assert created["output_text"] == "hi there"
    assert created["usage"] == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
    assert retrieved["id"] == created["id"]


def test_responses_store_false_is_not_retrievable(tmp_path, monkeypatch):
    async def fake_chat(req):
        return JSONResponse(_chat_payload("ephemeral"))

    monkeypatch.setattr(proxy, "chat_completions", fake_chat)
    client = _client(tmp_path, monkeypatch)

    created = client.post("/v1/responses", json={"model": "Gemini 3.5 Flash (High)", "input": "hello", "store": False}).json()
    missing = client.get(f"/v1/responses/{created['id']}")

    assert missing.status_code == 404
    assert missing.json()["error"]["type"] == "invalid_request_error"


def test_responses_input_parts_include_image(tmp_path, monkeypatch):
    seen = {}

    async def fake_chat(req):
        seen["content"] = req.messages[-1].content
        return JSONResponse(_chat_payload("vision ok"))

    monkeypatch.setattr(proxy, "chat_completions", fake_chat)
    client = _client(tmp_path, monkeypatch)

    response = client.post("/v1/responses", json={
        "model": "Gemini 3.5 Flash (High)",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ],
        }],
    })

    assert response.status_code == 200
    assert seen["content"][0] == {"type": "text", "text": "describe"}
    assert seen["content"][1]["type"] == "image_url"


def test_responses_rejects_unsupported_input_and_tools(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    bad_input = client.post("/v1/responses", json={
        "model": "Gemini 3.5 Flash (High)",
        "input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file_1"}]}],
    })
    bad_tool = client.post("/v1/responses", json={
        "model": "Gemini 3.5 Flash (High)",
        "input": "hello",
        "tools": [{"type": "file_search"}],
    })

    assert bad_input.status_code == 400
    assert bad_tool.status_code == 400
    assert bad_tool.json()["error"]["message"].startswith("Unsupported Responses tool type")


def test_responses_function_call_output_shape(tmp_path, monkeypatch):
    seen = {}

    async def fake_chat(req):
        seen["tools"] = req.tools
        seen["tool_choice"] = req.tool_choice
        return JSONResponse(_chat_payload(tool_calls=[{
            "id": "call_123",
            "type": "function",
            "function": {"name": "weather", "arguments": "{\"location\":\"Seoul\"}"},
        }]))

    monkeypatch.setattr(proxy, "chat_completions", fake_chat)
    client = _client(tmp_path, monkeypatch)

    response = client.post("/v1/responses", json={
        "model": "Gemini 3.5 Flash (High)",
        "input": "weather",
        "tools": [{"type": "function", "name": "weather", "parameters": {"type": "object", "properties": {}}}],
        "tool_choice": {"type": "function", "name": "weather"},
    }).json()

    assert response["output_text"] == ""
    assert response["output"][0]["type"] == "function_call"
    assert response["output"][0]["name"] == "weather"
    assert seen["tools"][0]["function"]["name"] == "weather"
    assert seen["tool_choice"]["function"]["name"] == "weather"


def test_responses_maps_web_search_text_format_and_reasoning(tmp_path, monkeypatch):
    seen = {}

    async def fake_chat(req):
        seen["tools"] = req.tools
        seen["text"] = req.model_extra.get("text")
        seen["reasoning"] = req.reasoning
        return JSONResponse(_chat_payload("{\"answer\":\"grounded\"}"))

    monkeypatch.setattr(proxy, "chat_completions", fake_chat)
    client = _client(tmp_path, monkeypatch)

    response = client.post("/v1/responses", json={
        "model": "Gemini 3.5 Flash (High)",
        "input": "latest docs?",
        "tools": [{"type": "web_search_preview"}],
        "text": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            }
        },
        "reasoning": {"effort": "low"},
    }).json()

    assert seen["tools"] == [{"type": "web_search_preview"}]
    assert seen["text"]["format"]["type"] == "json_schema"
    assert seen["reasoning"] == {"effort": "low"}
    assert response["text"]["format"]["type"] == "json_schema"
    assert response["reasoning"] == {"effort": "low"}


def test_responses_input_items_delete_and_previous_response(tmp_path, monkeypatch):
    seen = {}

    async def fake_chat(req):
        seen["messages"] = req.messages
        return JSONResponse(_chat_payload("stored context"))

    monkeypatch.setattr(proxy, "chat_completions", fake_chat)
    client = _client(tmp_path, monkeypatch)

    first = client.post("/v1/responses", json={"model": "Gemini 3.5 Flash (High)", "input": "first"}).json()
    items = client.get(f"/v1/responses/{first['id']}/input_items").json()
    second = client.post("/v1/responses", json={
        "model": "Gemini 3.5 Flash (High)",
        "previous_response_id": first["id"],
        "input": "second",
    }).json()
    deleted = client.delete(f"/v1/responses/{first['id']}").json()

    assert items["object"] == "list"
    assert items["data"][0]["role"] == "user"
    assert second["previous_response_id"] == first["id"]
    assert any(m.role == "user" and proxy._responses_message_text(m.content) == "first" for m in seen["messages"])
    assert any(m.role == "assistant" for m in seen["messages"])
    assert deleted == {"id": first["id"], "object": "response.deleted", "deleted": True}


def test_responses_input_tokens_cancel_background_and_compact(tmp_path, monkeypatch):
    async def fake_chat(req):
        return JSONResponse(_chat_payload("compact summary"))

    monkeypatch.setattr(proxy, "chat_completions", fake_chat)
    client = _client(tmp_path, monkeypatch)

    tokens = client.post("/v1/responses/input_tokens", json={"input": "hello world"}).json()
    queued = client.post("/v1/responses", json={"model": "Gemini 3.5 Flash (High)", "input": "later", "background": True}).json()
    cancelled = client.post(f"/v1/responses/{queued['id']}/cancel").json()
    compacted = client.post(f"/v1/responses/{cancelled['id']}/compact").json()

    assert tokens["input_tokens"] > 0
    assert queued["status"] == "queued"
    assert cancelled["status"] == "cancelled"
    assert compacted["metadata"]["compact_of"] == cancelled["id"]


def test_responses_streaming_events(tmp_path, monkeypatch):
    async def fake_chat(req):
        return JSONResponse(_chat_payload("stream text"))

    monkeypatch.setattr(proxy, "chat_completions", fake_chat)
    client = _client(tmp_path, monkeypatch)

    with client.stream("POST", "/v1/responses", json={"model": "Gemini 3.5 Flash (High)", "input": "hello", "stream": True}) as response:
        body = response.read().decode()

    assert "event: response.created" in body
    assert "event: response.output_text.delta" in body
    assert "event: response.completed" in body
    assert "data: [DONE]" in body
