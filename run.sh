#!/usr/bin/env bash
# Cineforge one-click launcher (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .installing ]; then
  echo "[Cineforge] Setup already running elsewhere; wait for it to finish."
  exit 0
fi

if [ ! -f .installed ]; then
  echo "[Cineforge] First-time setup..."
  : > .installing
  if ! bash setup.sh; then
    rm -f .installing
    echo "[Cineforge] Setup FAILED. See docs/TROUBLESHOOTING.md"
    exit 1
  fi
  rm -f .installing
fi

PYEXE="./.venv/bin/python"
[ -x "$PYEXE" ] || PYEXE="python3"
exec "$PYEXE" -m cineforge gui
