#!/usr/bin/env bash
# One-command start: deps + crawler + publisher.
# Usage: ./start.sh [first_year] [last_year]     (default: 2011 1983)
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# --- 1) dependencies (idempotent) -------------------------------------------
PY=python3
[ -x "$DIR/.venv/bin/python3" ] && PY="$DIR/.venv/bin/python3"

if "$PY" -c "import requests, bs4, kafka, redis, pydantic" 2>/dev/null; then
  echo "Dependencies OK, skipping install."
else
  echo "Installing dependencies..."
  installed=0
  # Try 1: the active venv's pip (if a venv exists)
  [ "$PY" != python3 ] && "$PY" -m pip install -r requirements.txt >/dev/null 2>&1 && installed=1
  # Try 2: system pip, with the common fallbacks
  [ $installed = 0 ] && pip3 install -r requirements.txt >/dev/null 2>&1 && installed=1
  [ $installed = 0 ] && pip3 install --user -r requirements.txt >/dev/null 2>&1 && installed=1
  [ $installed = 0 ] && pip3 install --break-system-packages -r requirements.txt >/dev/null 2>&1 && installed=1
  # Try 3 (last resort): create the kit's own virtualenv and install there
  if [ $installed = 0 ]; then
    echo "System pip blocked -> creating virtualenv .venv ..."
    python3 -m venv "$DIR/.venv" 2>/dev/null \
      || { echo "ERROR: venv failed. Install it first:  sudo apt install python3-venv   (or: sudo dnf install python3)"; exit 1; }
    "$DIR/.venv/bin/python3" -m pip install -r requirements.txt >/dev/null 2>&1 && installed=1
  fi
  [ $installed = 1 ] || { echo "ERROR: could not install dependencies (check internet / pip / venv)"; exit 1; }
fi
# Prove it before starting anything
"$([ -x "$DIR/.venv/bin/python3" ] && echo "$DIR/.venv/bin/python3" || echo python3)" \
  -c "import requests, bs4, kafka, redis, pydantic" \
  || { echo "ERROR: dependencies still missing — check logs above"; exit 1; }
echo "Dependencies OK."

# --- 2) don't double-start ----------------------------------------------------
pgrep -f "crawl_range.sh"  >/dev/null && echo "Crawler already running, leaving it." \
  || { setsid nohup bash crawl_range.sh "${1:-2011}" "${2:-1983}" >/dev/null 2>&1 & echo "Crawler started ($1-${2})."; }
pgrep -f "publish_loop.sh" >/dev/null && echo "Publisher already running, leaving it." \
  || { setsid nohup bash publish_loop.sh >/dev/null 2>&1 & echo "Publisher started (every 5 min)."; }

echo ""
echo "Done. Watch:  tail -f logs/crawl_range.log"
