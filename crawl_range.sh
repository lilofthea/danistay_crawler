#!/usr/bin/env bash
# Crawl a year range, newest first: crawl_range.sh 2011 1983
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
source env.sh
START=${1:-2011}; END=${2:-1983}
mkdir -p logs
LOG=logs/crawl_range.log
exec 9>logs/crawl.lock
flock -n 9 || { echo "already running"; exit 1; }

for y in $(seq "$START" -1 "$END"); do
  echo "==== YEAR $y started $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$LOG"
  tries=0
  while true; do
    python3 crawler.py detail \
      --baslangic "01.01.$y" --bitis "31.12.$y" \
      --direction asc --resume --delay 3.0 \
      >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then break;
    elif [ $rc -eq 2 ]; then
      tries=$((tries+1))
      [ $tries -ge 30 ] && { echo "YEAR $y: captcha x$tries, skipping (re-run later)" >> "$LOG"; break; }
      echo "YEAR $y: captcha, sleeping 30 min ($tries/30)" >> "$LOG"; sleep 1800
    else
      tries=$((tries+1))
      [ $tries -ge 5 ] && { echo "YEAR $y: errors x$tries, skipping" >> "$LOG"; break; }
      echo "YEAR $y: error rc=$rc, retry in 5 min ($tries/5)" >> "$LOG"; sleep 300
    fi
  done
done
echo "==== RANGE DONE $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$LOG"
