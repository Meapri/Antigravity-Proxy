from fastapi.testclient import TestClient

import antigravity_proxy as proxy


def test_responses_api_surface_is_removed():
    client = TestClient(proxy.app)

    checks = [
        client.post("/v1/responses", json={"model": "x", "input": "hello"}),
        client.post("/v1/responses/input_tokens", json={"input": "hello"}),
        client.get("/v1/responses/resp_test"),
        client.delete("/v1/responses/resp_test"),
        client.post("/v1/responses/resp_test/cancel"),
        client.get("/v1/responses/resp_test/input_items"),
        client.post("/v1/responses/resp_test/compact"),
    ]

    for response in checks:
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["status"] == "NOT_FOUND"
        assert "OpenAI-compatible endpoints have been removed" in body["error"]["message"]
