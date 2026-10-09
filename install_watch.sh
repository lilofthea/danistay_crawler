#!/usr/bin/env bash
# Installs the new-decision watcher (watch_new.py) as a cron job:
# every 5 days at 03:00 (days 1, 6, 11, 16, 21, 26, 31 of the month).
# flock skips a run if the previous one is still going. Log: logs/watch.log
# Idempotent: safe to re-run.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/logs"
JOB="0 3 */5 * * cd $DIR && DIR=$DIR . ./env.sh && flock -n logs/watch.lock \"\$PY\" watch_new.py >> logs/watch.log 2>&1"

# "|| true": an empty crontab must not end the subshell under set -e
( { crontab -l 2>/dev/null | grep -v "watch_new.py"; } || true
  echo "$JOB"
) | crontab -

echo "Cron now:"
crontab -l | grep watch_new.py
