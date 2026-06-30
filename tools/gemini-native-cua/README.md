# Gemini Native Computer Use Bridge

This tool lets Hermes use Gemini/Antigravity native Computer Use through the
main Antigravity proxy on `:8765`.

## Architecture

```text
Hermes native_computer_use
  -> gemini-native-cua CLI
    -> Antigravity Gemini proxy :8765
      -> local Computer Use Interactions loop
      -> Antigravity / Gemini backend for normal generation
```

The main Antigravity proxy now handles both halves of the Computer Use
Interactions loop:

1. initial `tools: [{"type":"computer_use", ...}]` requests return a local
   Computer Use `functionCall` such as `open_web_browser`; and
2. follow-up `/interactions` requests containing submitted `functionResponse`
   parts are completed locally by the same `:8765` service.

The older `proxy/gemini_native_cua_proxy.py` sidecar on `:8766` is retained only
as a backwards-compatible shim for existing deployments. New Hermes
configuration should point directly at the `:8765` prefixed Gemini URL:

```text
http://127.0.0.1:8765/generativelanguage.googleapis.com/v1beta
```

## Functional Integration

`gemini-native-cua` is not just a transport shim. A single run can combine:

- Gemini native Computer Use tool calls, such as `open_web_browser`.
- Hermes/cua-driver GUI actions for windows, keys, clicks, and scrolling.
- `xdotool` text-input fallback when AT-SPI text entry hangs.
- DOM link fallback when browser accessibility does not expose page content.

The CLI records these actions into a hybrid session trace under:

```text
~/.local/share/gemini-native-cua/logs/hybrid-session-*.json
```

Each trace keeps the user prompt, browser target, current URL, native actions,
Hermes GUI actions, and DOM fallback actions in one ordered event stream. This
makes native Computer Use and Hermes Computer Use behave as one cooperative
workflow rather than two isolated tools.

## Files

- `bin/gemini-native-cua`: CLI bridge from Gemini native Computer Use actions to
  `cua-driver` or the bundled browser executor.
- `proxy/gemini_native_cua_proxy.py`: legacy dependency-free compatibility
  proxy for deployments that still need a `:8766` sidecar.
- `systemd/gemini-native-cua-proxy.service`: legacy user service for the
  compatibility proxy on port `8766`.
- `scripts/install.sh`: installs the CLI, proxy, and systemd service.

## Install

From the repository root:

```bash
tools/gemini-native-cua/scripts/install.sh
```

Then point Hermes/Gemini configuration at:

```text
http://<host>:8765/generativelanguage.googleapis.com/v1beta
```

The legacy sidecar, when used, still listens on:

```text
http://127.0.0.1:8766/generativelanguage.googleapis.com/v1beta
```

## Executors

The CLI supports three execution modes:

- `--executor browser`: uses the bundled Playwright Chromium helper for browser
  navigation, typing, clicking, scrolling, screenshots, and DOM-backed element
  capture. This path does not require `cua-driver` for normal browser tasks and
  defaults to headless mode for service reliability. Set
  `GEMINI_NATIVE_CUA_HEADLESS=0` when a visible browser is required.
- `--executor cua`: uses the original `cua-driver`/`xdotool` desktop executor.
- `--executor auto`: tries the browser executor first and falls back to
  `cua-driver` when the browser executor cannot perform an action. This is the
  default used by the Hermes `native_computer_use` plugin.

The Playwright helper and its npm dependencies live inside this tool directory,
so Hermes and Antigravity updates do not need source patches for the browser
executor path.

## Verify

```bash
gemini-native-cua doctor
gemini-native-cua --base-url http://127.0.0.1:8765/generativelanguage.googleapis.com/v1beta \
  --max-actions 6 run "Open https://example.com"
gemini-native-cua --base-url http://127.0.0.1:8765/generativelanguage.googleapis.com/v1beta \
  --max-actions 6 run --planner-steps 3 \
  "Open https://example.com and open the More information link"
```

Use `--no-hybrid-trace` if you do not want a session trace for a run.

## Notes

- Native Computer Use result submission is integrated into `antigravity_proxy.py`
  itself; the `:8766` sidecar is no longer required for new installs.
- Browser navigation uses `xdotool` when available because Epiphany/AT-SPI can
  hang on text input and accessibility captures.
- DOM link fallback can follow normal HTML links when the browser accessibility
  tree does not expose page content.
