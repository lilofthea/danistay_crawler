#!/usr/bin/env bash
# One-command health check:  ./status.sh
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
source env.sh 2>/dev/null
BROKER_HOST=${REDPANDA_BROKERS%%:*}; BROKER_PORT=${REDPANDA_BROKERS##*:}
ok(){ printf '  \033[32m✓\033[0m %s\n'; } bad(){ printf '  \033[31m✗\033[0m %s\n'; }

echo "== 1. dependencies =="
for m in "requests" "bs4" "kafka" "redis" "pydantic"; do
  python3 -c "import $m" 2>/dev/null && ok "$m" || bad "$m  (fix: ./start.sh — it installs them)"
done

echo "== 2. network (havelsan link + central pipeline) =="
curl -s --max-time 8 http://www.msftconnecttest.com/connecttest.txt | grep -q "Microsoft Connect Test" \
  && ok "internet up" || bad "internet DOWN (check ~/.config/havelsan-login.log)"
timeout 5 bash -c "echo > /dev/tcp/$BROKER_HOST/$BROKER_PORT" 2>/dev/null \
  && ok "central broker $BROKER_HOST:$BROKER_PORT reachable" \
  || bad "central broker NOT reachable (LAN / firewall?)"

echo "== 3. processes =="
pgrep -f "crawler.py detail" >/dev/null && ok "crawler running" \
  || { pgrep -f "crawl_range.sh" >/dev/null && ok "crawler running (in backoff/sleep)" || bad "crawler NOT running (fix: ./start.sh)"; }
pgrep -f "publish_loop.sh" >/dev/null && ok "publisher running" || bad "publisher NOT running (fix: ./start.sh)"

echo "== 4. crawl progress =="
tail -n 2 logs/crawl_range.log 2>/dev/null | sed 's/^/    /' || bad "no logs/crawl_range.log yet"

echo "== 5. publish (last activity) =="
tail -n 2 logs/publish.log 2>/dev/null | grep -E "Produced|complete|ERROR" | tail -n 2 | sed 's/^/    /' \
  || echo "    (no publish output yet — first run may take a few minutes)"

echo "== 6. data on disk =="
echo "    $(wc -l < data/index.jsonl 2>/dev/null || echo 0) decisions indexed, $(ls data/docs 2>/dev/null | wc -l) doc files"
