#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
pip install -r requirements.txt
setsid nohup bash crawl_range.sh "${1:-2011}" "${2:-1983}" > /dev/null 2>&1 &
setsid nohup bash publish_loop.sh > /dev/null 2>&1 &
echo "Crawler (range ${1:-2011}-${2:-1983}) + publisher started."
echo "Logs: $DIR/logs/"
