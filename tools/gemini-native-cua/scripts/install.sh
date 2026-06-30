#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_LIB="${INSTALL_LIB:-$HOME/.local/lib/gemini-native-cua}"
INSTALL_BIN="${INSTALL_BIN:-$HOME/.local/bin}"
SYSTEMD_USER_DIR="${SYSTEMD_USER_DIR:-$HOME/.config/systemd/user}"
SERVICE_NAME="gemini-native-cua-proxy.service"
INSTALL_LEGACY_PROXY="${INSTALL_LEGACY_PROXY:-0}"

mkdir -p "$INSTALL_LIB" "$INSTALL_BIN"

install -m 0755 "$ROOT/bin/gemini-native-cua" "$INSTALL_BIN/gemini-native-cua"
install -m 0755 "$ROOT/bin/playwright-browser-executor.js" "$INSTALL_LIB/playwright-browser-executor.js"
install -m 0644 "$ROOT/package.json" "$INSTALL_LIB/package.json"
if [[ -f "$ROOT/package-lock.json" ]]; then
  install -m 0644 "$ROOT/package-lock.json" "$INSTALL_LIB/package-lock.json"
fi
if command -v npm >/dev/null 2>&1; then
  npm --prefix "$INSTALL_LIB" install --omit=dev
else
  echo "Warning: npm not found; browser executor dependencies were not installed." >&2
fi

if [[ "$INSTALL_LEGACY_PROXY" == "1" ]]; then
  mkdir -p "$SYSTEMD_USER_DIR"
  install -m 0755 "$ROOT/proxy/gemini_native_cua_proxy.py" "$INSTALL_LIB/gemini_native_cua_proxy.py"
  install -m 0644 "$ROOT/systemd/$SERVICE_NAME" "$SYSTEMD_USER_DIR/$SERVICE_NAME"
  systemctl --user daemon-reload
  systemctl --user enable --now "$SERVICE_NAME"
  echo "Legacy compatibility proxy health: http://127.0.0.1:8766/health"
fi

echo "Installed gemini-native-cua."
echo "Set Hermes Gemini base_url to the integrated Antigravity proxy:"
echo "  http://<host>:8765/generativelanguage.googleapis.com/v1beta"
echo "Set INSTALL_LEGACY_PROXY=1 only for older deployments that still require :8766."
