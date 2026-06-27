#!/usr/bin/env python3
"""Compatibility proxy for Gemini native Computer Use experiments.

This proxy intentionally does not import or patch the Antigravity proxy. It
forwards normal Gemini REST traffic to an upstream proxy and only handles the
Computer Use result-submission turn locally.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


DEFAULT_LISTEN_HOST = "0.0.0.0"
DEFAULT_LISTEN_PORT = 8766
DEFAULT_UPSTREAM = "http://127.0.0.1:8765"


def truncate(value: str, limit: int = 4000) -> str:
    return value if len(value) <= limit else value[:limit] + "...[truncated]"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def estimate_tokens(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return max(1, len(text) // 4)


def json_response_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def interaction_function_results(contents: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not isinstance(contents, list):
        return results
    for content in contents:
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            function_response = part.get("functionResponse") or part.get("function_response")
            if isinstance(function_response, dict):
                results.append(function_response)
    return results


def completion_text(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    latest = results[-1]
    response = latest.get("response")
    if isinstance(response, dict):
        if response.get("ok") is False:
            return "Computer use action failed: " + str(response.get("error") or response)
        result = response.get("result")
        if isinstance(result, dict):
            capture = result.get("capture")
            if isinstance(capture, dict) and capture.get("summary"):
                return "Done. The computer use action completed and the browser state was observed."
        if response.get("url"):
            return "Done. The browser is open."
    return "Done. The computer use action completed."


def model_name_from_body(body: dict[str, Any]) -> str:
    raw = str(body.get("model") or body.get("modelName") or body.get("model_name") or "gemini-3.5-flash")
    return raw if raw.startswith("models/") else f"models/{raw}"


def completed_interaction(body: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    created = now_iso()
    name = "interactions/native-cua-" + uuid.uuid4().hex
    model = model_name_from_body(body)
    text = completion_text(results)
    usage = {
        "inputTokens": estimate_tokens(body),
        "outputTokens": estimate_tokens(text),
        "totalTokens": estimate_tokens(body) + estimate_tokens(text),
    }
    output = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": text}],
            },
            "finishReason": "STOP",
        }],
        "modelVersion": model.rsplit("/", 1)[-1],
        "usageMetadata": {
            "promptTokenCount": usage["inputTokens"],
            "candidatesTokenCount": usage["outputTokens"],
            "totalTokenCount": usage["totalTokens"],
        },
    }
    model_content = output["candidates"][0]["content"]
    contents = body.get("contents") if isinstance(body.get("contents"), list) else []
    return {
        "name": name,
        "id": name.rsplit("/", 1)[-1],
        "model": model,
        "agent": body.get("agent"),
        "status": "completed",
        "created": created,
        "updated": created,
        "createTime": created,
        "updateTime": created,
        "previousInteractionId": body.get("previousInteractionId") or body.get("previous_interaction_id"),
        "input": body.get("input", body.get("messages", body.get("contents"))),
        "request": body,
        "output": output,
        "outputText": text,
        "output_text": text,
        "history": contents + [model_content],
        "steps": [
            {
                "type": "computer_use_result",
                "status": "completed",
                "results": results,
            },
            {
                "type": "model_output",
                "status": "completed",
                "content": [{"type": "text", "text": text, "annotations": []}],
                "outputText": text,
                "response": output,
            },
        ],
        "usage": usage,
        "usageMetadata": output["usageMetadata"],
    }


def is_interactions_path(path: str) -> bool:
    normalized = path.split("?", 1)[0].rstrip("/")
    return normalized.endswith("/interactions")


class CompatibilityProxy(BaseHTTPRequestHandler):
    server_version = "gemini-native-cua-proxy/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    @property
    def upstream(self) -> str:
        return str(self.server.upstream).rstrip("/")  # type: ignore[attr-defined]

    def read_body(self) -> bytes:
        length = int(self.headers.get("content-length") or "0")
        return self.rfile.read(length) if length else b""

    def send_json(self, status: int, payload: Any) -> None:
        body = json_response_bytes(payload)
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            lower = key.lower()
            if lower in {"connection", "transfer-encoding", "content-encoding", "content-length"}:
                continue
            self.send_header(key, value)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/health", "/__native_cua/health"}:
            self.send_json(200, {"ok": True, "upstream": self.upstream})
            return
        self.forward()

    def do_HEAD(self) -> None:
        self.forward()

    def do_DELETE(self) -> None:
        self.forward()

    def do_PATCH(self) -> None:
        self.forward()

    def do_PUT(self) -> None:
        self.forward()

    def do_POST(self) -> None:
        body_bytes = self.read_body()
        if is_interactions_path(self.path):
            try:
                body = json.loads(body_bytes.decode("utf-8", "replace") or "{}")
            except json.JSONDecodeError:
                body = None
            if isinstance(body, dict):
                results = interaction_function_results(body.get("contents"))
                if results:
                    self.send_json(200, completed_interaction(body, results))
                    return
        self.forward(body_bytes)

    def forward(self, body: bytes | None = None) -> None:
        if body is None and self.command in {"POST", "PUT", "PATCH"}:
            body = self.read_body()
        url = self.upstream + self.path
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
        }
        request = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_body = response.read()
                self.send_bytes(response.status, dict(response.headers.items()), response_body)
        except urllib.error.HTTPError as error:
            error_body = error.read()
            self.send_bytes(error.code, dict(error.headers.items()), error_body)
        except Exception as exc:
            self.send_json(502, {"error": {"message": truncate(str(exc), 1000), "status": "BAD_GATEWAY"}})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini native Computer Use compatibility proxy.")
    parser.add_argument("--host", default=os.environ.get("NATIVE_CUA_PROXY_HOST", DEFAULT_LISTEN_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NATIVE_CUA_PROXY_PORT", DEFAULT_LISTEN_PORT)))
    parser.add_argument("--upstream", default=os.environ.get("NATIVE_CUA_UPSTREAM", DEFAULT_UPSTREAM))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), CompatibilityProxy)
    server.upstream = args.upstream.rstrip("/")  # type: ignore[attr-defined]
    print(f"gemini-native-cua proxy listening on {args.host}:{args.port}, upstream={server.upstream}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
