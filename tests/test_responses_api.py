import pytest
from fastapi.testclient import TestClient

import antigravity_proxy as proxy


class FakeClient:
    def complete(self, *, system="", prompt="", memories=None, model=""):
        return "fake response text"

    def _build_gemini_request(self, **kwargs):
        return {"contents": [], "generationConfig": {}}

    def _extract_text(self, data):
        return None


@pytest.fixture()
def client(monkeypatch, tmp_path):
    db_path = tmp_path / "responses.sqlite3"
    monkeypatch.setenv("ANTIGRAVITY_RESPONSES_DB", str(db_path))
    monkeypatch.setattr(proxy, "_get_client", lambda: FakeClient())
    return TestClient(proxy.app)


def test_responses_create_string_input(client):
    resp = client.post("/v1/responses", json={"model": "gemini-3-flash", "input": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output_text"] == "fake response text"


def test_responses_input_tokens(client):
    resp = client.post("/v1/responses/input_tokens", json={
        "model": "gemini-3-flash",
        "input": "count my tokens",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response.input_tokens"
    assert isinstance(body["input_tokens"], int)


def test_responses_crud(client):
    # Create with store=True so it's persisted
    create = client.post("/v1/responses", json={
        "model": "gemini-3-flash",
        "input": "hello",
        "store": True,
    })
    assert create.status_code == 200
    response_id = create.json()["id"]

    get = client.get(f"/v1/responses/{response_id}")
    assert get.status_code == 200
    assert get.json()["id"] == response_id

    items = client.get(f"/v1/responses/{response_id}/input_items")
    assert items.status_code == 200
    assert items.json()["object"] == "list"

    cancel = client.post(f"/v1/responses/{response_id}/cancel")
    assert cancel.status_code == 200

    delete = client.delete(f"/v1/responses/{response_id}")
    assert delete.status_code == 200
    assert delete.json()["deleted"] is True


def test_responses_compact(client):
    create = client.post("/v1/responses", json={
        "model": "gemini-3-flash",
        "input": "hello",
        "store": True,
    })
    assert create.status_code == 200
    response_id = create.json()["id"]

    compact = client.post(f"/v1/responses/{response_id}/compact")
    assert compact.status_code == 200
    assert compact.json()["object"] == "response"


def test_responses_auth_returns_openai_shape(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_PROXY_API_KEY", "secret")
    c = TestClient(proxy.app)
    resp = c.post("/v1/responses", json={"model": "gemini-3-flash", "input": "hi"})
    assert resp.status_code == 401
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "authentication_error"
