import importlib.machinery
import importlib.util
from pathlib import Path


CLI_PATH = Path(__file__).resolve().parents[1] / "tools" / "gemini-native-cua" / "bin" / "gemini-native-cua"


def load_cli_module():
    loader = importlib.machinery.SourceFileLoader("gemini_native_cua_cli_test", str(CLI_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_parser_defaults_to_auto_executor():
    cli = load_cli_module()

    args = cli.build_parser().parse_args(["--base-url", "http://127.0.0.1:8766/v1beta", "run", "Open https://example.com"])

    assert args.executor == "auto"
    assert args.browser == cli.DEFAULT_BROWSER
    assert args.prompt == "Open https://example.com"


def test_navigate_to_url_prefers_executor_native_navigate():
    cli = load_cli_module()

    class BrowserLike:
        def __init__(self):
            self.calls = []

        def navigate(self, url):
            self.calls.append(("navigate", url))
            return {"url": url, "capture": {"summary": url}}

        def send_key_no_capture(self, key):  # pragma: no cover - must not be called
            raise AssertionError(f"unexpected key call: {key}")

    executor = BrowserLike()
    result = cli.navigate_to_url(executor, "https://example.com")

    assert result["url"] == "https://example.com"
    assert executor.calls == [("navigate", "https://example.com")]


def test_auto_executor_falls_back_to_cua_when_browser_fails(monkeypatch, tmp_path):
    cli = load_cli_module()

    class FailingBrowser:
        def __init__(self, log_dir, browser):
            pass

        def execute(self, name, args):
            raise cli.BridgeError("browser unavailable")

    class WorkingCua:
        def __init__(self, log_dir, browser):
            pass

        def execute(self, name, args):
            return {"executor": "cua", "name": name, "args": args}

    monkeypatch.setattr(cli, "BrowserExecutor", FailingBrowser)
    monkeypatch.setattr(cli, "CuaExecutor", WorkingCua)

    executor = cli.AutoExecutor(tmp_path, "firefox")
    result = executor.execute("open_web_browser", {})

    assert result == {"executor": "cua", "name": "open_web_browser", "args": {}}


def test_browser_executor_helper_uses_json_subprocess(monkeypatch, tmp_path):
    cli = load_cli_module()
    helper = tmp_path / "playwright-browser-executor.js"
    helper.write_text("#!/usr/bin/env node\n")

    calls = []

    class Completed:
        returncode = 0
        stdout = '{"ok":true,"url":"https://example.com","viewport":{"width":1000,"height":800}}'
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "find_display", lambda: ":99")
    monkeypatch.setattr(cli, "find_xauthority", lambda: "/tmp/xauth")

    executor = cli.BrowserExecutor(tmp_path / "logs", "firefox")
    executor.helper = helper
    data = executor._call_helper("navigate", {"url": "https://example.com"})

    assert data["ok"] is True
    assert data["url"] == "https://example.com"
    assert calls[0][0] == ["node", str(helper)]
    assert '"action": "navigate"' in calls[0][1]["input"]
    assert calls[0][1]["env"]["DISPLAY"] == ":99"
    assert calls[0][1]["env"]["XAUTHORITY"] == "/tmp/xauth"
