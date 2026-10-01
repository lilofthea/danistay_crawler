#!/usr/bin/env bash
# One-command start: deps + crawler + publisher.
# Usage: ./start.sh [first_year] [last_year]     (default: 2011 1983)
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# --- 1) dependencies (idempotent) -------------------------------------------
if python3 -c "import requests, bs4, kafka, redis, pydantic" 2>/dev/null; then
  echo "Dependencies OK, skipping install."
else
  echo "Installing dependencies..."
  PIP=()
  if command -v pip3 >/dev/null; then PIP=(pip3); elif command -v pip >/dev/null; then PIP=(pip); else echo "ERROR: pip not found"; exit 1; fi
  "${PIP[@]}" install -r requirements.txt 2>/dev/null \
    || "${PIP[@]}" install --user -r requirements.txt 2>/dev/null \
    || "${PIP[@]}" install --break-system-packages -r requirements.txt \
    || { echo "ERROR: pip install failed (check python3/pip + internet)"; exit 1; }
fi
python3 -c "import requests, bs4, kafka, redis, pydantic" \
  || { echo "ERROR: dependencies still missing — check logs above"; exit 1; }

# --- 2) don't double-start ----------------------------------------------------
pgrep -f "crawl_range.sh"  >/dev/null && echo "Crawler already running, leaving it." \
  || { setsid nohup bash crawl_range.sh "${1:-2011}" "${2:-1983}" >/dev/null 2>&1 & echo "Crawler started ($1-${2})."; }
pgrep -f "publish_loop.sh" >/dev/null && echo "Publisher already running, leaving it." \
  || { setsid nohup bash publish_loop.sh >/dev/null 2>&1 & echo "Publisher started (every 5 min)."; }

echo ""
echo "Done. Watch:  tail -f logs/crawl_range.log"
