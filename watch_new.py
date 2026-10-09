#!/usr/bin/env python3
"""
Find decisions the site added after we crawled a date, and fetch them.

The site's date filter gives a cheap count (recordsTotal, one request). For
every year in data/index.jsonl the count on the site is compared with ours;
where they differ the range is split (year -> months -> days) and only the
days that differ are crawled again, from page 1 with --resume, so only the
missing decisions are downloaded. The publisher then sends them on as usual.

Only dates the main crawler has already passed are checked: from the day
after the oldest date we have in a year (that day may still be in progress)
to the end of that year.

Usage:
  python3 watch_new.py              # check and fetch
  python3 watch_new.py --dry-run    # only report the days that differ
Run every 5 days by cron (see install_watch.sh); log: logs/watch.log
"""

import argparse
import calendar
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import crawler

ROOT = Path(__file__).parent
INDEX_FILE = ROOT / "data" / "index.jsonl"
PROBE_DELAY = 3.0       # seconds between count requests


def parse(d: str) -> date:
    dd, mm, yy = d.split(".")
    return date(int(yy), int(mm), int(dd))


def fmt(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def local_counts() -> Counter:
    c = Counter()
    with open(INDEX_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                c[parse(json.loads(line)["kararTarihi"])] += 1
            except Exception:
                pass
    return c


def site_count(a: date, b: date) -> int:
    time.sleep(PROBE_DELAY)
    r = crawler.fetch_detail_page(1, 1, {
        "baslangicTarihi": fmt(a), "bitisTarihi": fmt(b),
        "siralama": "3", "siralamaDirection": "desc"})
    if not r["ok"]:
        raise RuntimeError(f"count {fmt(a)}-{fmt(b)}: {r['error']}")
    return crawler.get_rows(r)[1]


def months(a: date, b: date):
    d = a
    while d <= b:
        end = date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
        yield d, min(end, b)
        d = end + timedelta(days=1)


def days(a: date, b: date):
    d = a
    while d <= b:
        yield d, d
        d += timedelta(days=1)


def find_diffs(a: date, b: date, local: Counter, log) -> list[tuple[date, int, int]]:
    """Days in [a, b] where the site has more decisions than we do."""
    ours = sum(n for d, n in local.items() if a <= d <= b)
    theirs = site_count(a, b)
    if theirs <= ours:
        return []
    if a == b:
        return [(a, theirs, ours)]
    log(f"  {fmt(a)}-{fmt(b)}: site {theirs}, local {ours} -> splitting")
    split = months if (a.year, a.month) != (b.year, b.month) else days
    out = []
    for x, y in split(a, b):
        out += find_diffs(x, y, local, log)
    return out


def crawl_day(d: date) -> int:
    """Re-crawl one day from page 1; returns the crawler's exit code."""
    with open(ROOT / "logs" / "watch.log", "a", encoding="utf-8") as logf:
        return subprocess.call(
            [str(ROOT / "venv" / "bin" / "python"), str(ROOT / "crawler.py"), "detail",
             "--baslangic", fmt(d), "--bitis", fmt(d), "--start-page", "1",
             "--direction", "asc", "--resume", "--delay", "3.0"],
            cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report differences, fetch nothing")
    args = ap.parse_args()
    crawler.BASE_DELAY = 3.0
    (ROOT / "logs").mkdir(exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        if sys.stdout.isatty():   # under cron stdout is the same log file
            print(line, flush=True)
        with open(ROOT / "logs" / "watch.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    local = local_counts()
    if not local:
        sys.exit("index is empty")
    log(f"==== watch started{' (dry run)' if args.dry_run else ''}")

    diffs = []
    try:
        for year in sorted({d.year for d in local}, reverse=True):
            oldest = min(d for d in local if d.year == year)
            a, b = oldest + timedelta(days=1), date(year, 12, 31)
            if a > b:
                continue
            log(f"year {year}: checking {fmt(a)}-{fmt(b)} "
                f"(before {fmt(a)} the main crawler is still working)")
            diffs += find_diffs(a, b, local, log)
    except crawler.CaptchaError as e:
        log(f"captcha while counting, stopping (next run retries): {e}")
        sys.exit(2)
    except Exception as e:   # e.g. site timeouts; the next run retries
        log(f"error while counting, stopping (next run retries): {e}")
        sys.exit(1)

    if not diffs:
        log("no new decisions on the site")
        return
    log(f"{len(diffs)} day(s) with new decisions: " +
        ", ".join(f"{fmt(d)} (+{t - o})" for d, t, o in diffs))
    if args.dry_run:
        return

    before = sum(1 for _ in open(INDEX_FILE, encoding="utf-8"))
    for d, theirs, ours in diffs:
        log(f"fetching {fmt(d)}: site {theirs}, local {ours}")
        rc = crawl_day(d)
        if rc == 2:
            log("captcha, stopping (next run retries)")
            break
        if rc != 0:
            log(f"crawler exit {rc} for {fmt(d)}, continuing")
    after = sum(1 for _ in open(INDEX_FILE, encoding="utf-8"))
    log(f"==== watch done: {after - before} new decision(s) saved")


if __name__ == "__main__":
    main()
