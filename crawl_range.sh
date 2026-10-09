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
  # Separate counters: captcha pauses must not use up the error budget.
  # Both reset whenever a run saved new decisions, so they only count a
  # streak of runs that got nowhere.
  captchas=0; errors=0
  while true; do
    rc=0
    before=$(wc -l < data/index.jsonl 2>/dev/null || echo 0)
    "$PY" crawler.py detail \
      --baslangic "01.01.$y" --bitis "31.12.$y" \
      --direction asc --resume --delay 3.0 \
      >> "$LOG" 2>&1 || rc=$?
    after=$(wc -l < data/index.jsonl 2>/dev/null || echo 0)
    [ "$after" -gt "$before" ] && { captchas=0; errors=0; }
    if [ $rc -eq 0 ]; then break;
    elif [ $rc -eq 2 ]; then
      # captcha usually clears in under 30 min (the run after a 30 min pause
      # saved decisions 12 times out of 15): wait 10, then 20, then 30 min
      # while it persists; the counter resets once decisions are saved again
      captchas=$((captchas+1))
      [ $captchas -ge 30 ] && { echo "YEAR $y: captcha x$captchas, skipping (re-run later)" >> "$LOG"; break; }
      wait=$(( captchas < 3 ? 600 * captchas : 1800 ))
      echo "YEAR $y: captcha, sleeping $((wait / 60)) min ($captchas/30) $(date '+%H:%M')" >> "$LOG"; sleep $wait 9>&-   # 9>&-: a leftover sleep must not keep holding the lock
    else
      # site timeouts can last hours: back off 5, 10, 20, then 30 min,
      # up to 30 tries (~14 h) without progress before giving up
      errors=$((errors+1))
      [ $errors -ge 30 ] && { echo "YEAR $y: errors x$errors, skipping" >> "$LOG"; break; }
      wait=$(( errors < 4 ? 300 * (1 << (errors - 1)) : 1800 ))
      echo "YEAR $y: error rc=$rc, retry in $((wait / 60)) min ($errors/30)" >> "$LOG"; sleep $wait 9>&-   # 9>&-: a leftover sleep must not keep holding the lock
    fi
  done
done
echo "==== RANGE DONE $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$LOG"
