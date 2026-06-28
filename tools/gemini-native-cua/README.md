# Gemini Native Computer Use Bridge

This tool lets Hermes use Gemini/Antigravity native Computer Use without
patching `antigravity_proxy.py`.

## Architecture

```text
Hermes / gemini-native-cua
  -> gemini-native-cua compatibility proxy :8766
    -> Antigravity Gemini proxy :8765
      -> Antigravity / Gemini backend
```

The compatibility proxy forwards normal Gemini REST traffic to the existing
Antigravity proxy. It only handles the `/interactions` result-submission turn
for native Computer Use by converting submitted `functionResponse` parts into a
completed interaction.

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
  `cua-driver`.
- `proxy/gemini_native_cua_proxy.py`: dependency-free compatibility proxy.
- `systemd/gemini-native-cua-proxy.service`: user service for the compatibility
  proxy on port `8766`.
- `scripts/install.sh`: installs the CLI, proxy, and systemd service.

## Install

From the repository root:

```bash
tools/gemini-native-cua/scripts/install.sh
```

Then point Hermes/Gemini configuration at:

```text
http://<host>:8766/generativelanguage.googleapis.com/v1beta
```

The upstream Antigravity proxy should remain available on:

```text
http://127.0.0.1:8765
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
gemini-native-cua --max-actions 6 run "Open https://example.com"
gemini-native-cua --max-actions 6 run --planner-steps 3 \
  "Open https://example.com and open the More information link"
```

Use `--no-hybrid-trace` if you do not want a session trace for a run.

## Notes

- The bridge intentionally lives outside the Antigravity proxy implementation.
  Updating the proxy should not require reapplying source patches.
- Browser navigation uses `xdotool` when available because Epiphany/AT-SPI can
  hang on text input and accessibility captures.
- DOM link fallback can follow normal HTML links when the browser accessibility
  tree does not expose page content.
