#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
source env.sh
mkdir -p logs
while true; do
  "$DIR/venv/bin/python" -m producers.danistay_producer >> logs/publish.log 2>&1
  sleep 300
done
