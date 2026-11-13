#!/usr/bin/env bash
# ChatAutopsy — one-command launcher.
#
#   ./start.sh                         open the dashboard
#   ./start.sh analyze ~/Downloads/X   analyse an export and open the dashboard
#   ./start.sh demo | llm setup | doctor | list   (see ./start.sh --help)
#
# The first run creates the Python environment in .venv and installs everything (a few minutes);
# later runs start immediately. Reinstalls automatically when pyproject.toml changes.
set -euo pipefail
cd "$(dirname "$0")"

find_python() {
  # newest first: PATH, then Homebrew (including keg-only python@3.x), then system python3
  local candidates=("${PYTHON:-}")
  for v in 15 14 13 12 11 10; do candidates+=("python3.$v"); done
  for d in /opt/homebrew/opt/python@3.* /usr/local/opt/python@3.*; do
    [ -d "$d/bin" ] && candidates+=("$d"/bin/python3.*)
  done
  candidates+=(/opt/homebrew/bin/python3 /usr/local/bin/python3 python3)
  for c in "${candidates[@]}"; do
    [ -z "$c" ] && continue
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ]; then
  if ! PY="$(find_python)"; then
    echo "✗ Python ≥ 3.10 is required."
    if command -v brew >/dev/null 2>&1; then
      read -r -p "Install Python 3.12 with Homebrew? [y/N] " ans
      if [[ "$ans" =~ ^[sSyY] ]]; then brew install python@3.12; PY="$(find_python)"; else exit 1; fi
    else
      echo "  Install Python from https://www.python.org/downloads/ and run ./start.sh again"; exit 1
    fi
  fi
  echo "🐍 Creating the environment with $("$PY" --version) …"
  "$PY" -m venv .venv
fi

STAMP=.venv/.chatautopsy-install
HASH="$( (cat pyproject.toml; uname -sm) | shasum | cut -d' ' -f1)"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$HASH" ]; then
  echo "📦 Installing ChatAutopsy and its dependencies (first run only)…"
  EXTRAS="embeddings"
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then EXTRAS="all"; fi
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -e ".[${EXTRAS}]"
  else
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -e ".[${EXTRAS}]"
  fi
  echo "$HASH" > "$STAMP"
fi

export HF_HUB_DISABLE_TELEMETRY=1
exec .venv/bin/chatautopsy "$@"
