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

## Verify

```bash
gemini-native-cua doctor
gemini-native-cua --max-actions 6 run "Open https://example.com"
gemini-native-cua --max-actions 6 run --planner-steps 3 \
  "Open https://example.com and open the More information link"
```

## Notes

- The bridge intentionally lives outside the Antigravity proxy implementation.
  Updating the proxy should not require reapplying source patches.
- Browser navigation uses `xdotool` when available because Epiphany/AT-SPI can
  hang on text input and accessibility captures.
- DOM link fallback can follow normal HTML links when the browser accessibility
  tree does not expose page content.
