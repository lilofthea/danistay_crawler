#!/usr/bin/env python3
"""
Danıştay Karar Arama crawler (https://karararama.danistay.gov.tr)

Endpoints (reverse-engineered, all POST JSON body {"data": {...}}):
  /aramalist       - keyword search (andKelimeler / orKelimeler / notAndKelimeler / notOrKelimeler)
  /aramadetaylist  - structured search (daire, esas/karar year+no ranges, dates, mevzuat)
  GET /getDokuman?id=<id>&arananKelime=<kw> - full decision HTML (text in #hiddencontent)

Notes:
  - reCAPTCHA is OFF by default but can be switched on server-side under load.
    When that happens the API returns metadata.FMC == reCaptchaTimeout / EXCEPTION.
    The crawler detects this and aborts cleanly (resume later).
  - The site requires at least one search criterion for /aramadetaylist
    (daire, or esas year+no, or karar year+no, or date range, or mevzuat).

Usage examples:
  python3 crawler.py keyword "kamulaştırma"
  python3 crawler.py keyword "icra takibi" --or "tahsil" --limit 1000
  python3 crawler.py detail --daire "1. Daire"
  python3 crawler.py detail --esas-yil 2020 --esas-ilk 1 --esas-son 99999
  python3 crawler.py detail --baslangic "01.01.2023" --bitis "31.12.2023"
"""

import argparse
import json
import re
import sys
import time
from html import unescape
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

BASE = "https://karararama.danistay.gov.tr"
OUT_DIR = Path(__file__).parent / "data"
INDEX_FILE = OUT_DIR / "index.jsonl"
DOCS_DIR = OUT_DIR / "docs"
META_FILE = OUT_DIR / "crawl_meta.jsonl"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Content-Type": "application/json; charset=utf-8",
    "Origin": BASE,
    "Referer": BASE + "/",
    "Accept": "*/*",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


class CaptchaError(RuntimeError):
    """Server switched on reCAPTCHA - stop and resume later."""


def sleep(seconds: float):
    time.sleep(seconds)


# ---------------------------------------------------------------- throttling
# The server throttles by IP. Rather than walking back into a 429 at the same
# rate, every rejection doubles a global delay multiplier; a run of clean
# responses halves it again. Net yield is higher than a fixed --delay because
# almost nothing gets rejected.
BASE_DELAY = 1.0          # set from --delay at startup
THROTTLE = 1.0            # multiplier on BASE_DELAY
THROTTLE_MAX = 30.0
DECAY_AFTER = 20          # clean responses needed before easing off
_clean_streak = 0

# Captcha strikes must persist across calls: the server turns reCAPTCHA on
# globally, not per request, so a counter scoped to one post_json() can never
# reach the limit and the clean-stop-and-resume path never fires. A full
# recovery to BASE_DELAY clears the strikes.
CAPTCHA_LIMIT = 3
_captcha_strikes = 0

# Escalating backoff while the site keeps rejecting us. A fixed 60 s retry
# re-triggers the penalty over and over (observed: hours of 429/captcha loops
# with zero progress). Escalate: 1m -> 3m -> 10m -> 30m -> 1h (capped);
# resets only after a real clean run.
BLOCK_BACKOFFS = [60, 180, 600, 1800, 3600]
_block_escalation = 0


def _fmt_wait(s: int) -> str:
    return f"{s // 3600}h" if s >= 3600 else (f"{s // 60} min" if s >= 60 else f"{s}s")


def _save_throttle():
    """Persist the throttle multiplier so a watchdog restart doesn't reset it
    to base delay (which instantly re-triggers the 429 storm)."""
    try:
        (OUT_DIR / ".throttle").write_text(f"{THROTTLE:.2f}\n")
    except OSError:
        pass


def _load_throttle():
    global THROTTLE
    try:
        v = float((OUT_DIR / ".throttle").read_text().strip() or 1)
        THROTTLE = max(1.0, min(v, THROTTLE_MAX))
        if THROTTLE > 1.0:
            print(f"  ~ restored persisted throttle x{THROTTLE:.1f} "
                  f"(delay {current_delay():.0f}s)", flush=True)
    except (OSError, ValueError):
        pass


def current_delay() -> float:
    return BASE_DELAY * THROTTLE


def note_throttled(what: str, hard: bool = False):
    """A 429/503/captcha came back - slow down."""
    global THROTTLE, _clean_streak, _block_escalation
    _clean_streak = 0
    _block_escalation = min(_block_escalation + 1, len(BLOCK_BACKOFFS) - 1)
    if THROTTLE < THROTTLE_MAX:
        THROTTLE = THROTTLE_MAX if hard else min(THROTTLE * 2, THROTTLE_MAX)
        print(f"  ~ throttle up, delay now {current_delay():.1f}s ({what})", flush=True)
    _save_throttle()


def note_captcha() -> int:
    """Count a captcha response and slow to the cap. Returns the strike total."""
    global _captcha_strikes
    _captcha_strikes += 1
    note_throttled("captcha", hard=True)
    return _captcha_strikes


def note_clean():
    """A request succeeded - ease back toward BASE_DELAY after a clean run."""
    global THROTTLE, _clean_streak, _captcha_strikes, _block_escalation
    if THROTTLE <= 1.0:
        if _block_escalation:
            _block_escalation = 0
        return
    _clean_streak += 1
    if _clean_streak >= DECAY_AFTER:
        _clean_streak = 0
        THROTTLE = max(THROTTLE / 2, 1.0)
        _block_escalation = max(_block_escalation - 1, 0)
        print(f"  ~ throttle down, delay now {current_delay():.1f}s", flush=True)
        _save_throttle()
        if THROTTLE <= 1.0 and _captcha_strikes:
            _captcha_strikes = 0
            print("  ~ recovered to base delay, captcha strikes cleared", flush=True)


def post_json(path: str, data: dict, max_retries: int = 4) -> dict:
    """POST a search/list request and return parsed JSON. Retries on network errors."""
    url = BASE + path
    last_err = None
    for attempt in range(max_retries):
        try:
            r = SESSION.post(url, json={"data": data}, timeout=60)
            if r.status_code == 200:
                payload = r.json()
                meta = payload.get("metadata") or {}
                if meta.get("FMTY") == "ERROR":
                    msg = meta.get("FMTE", "")
                    if "reCaptcha" in msg or "Captcha" in msg:
                        strikes = note_captcha()
                        if strikes >= CAPTCHA_LIMIT:
                            raise CaptchaError("reCAPTCHA enabled by server: " + msg)
                        wait = BLOCK_BACKOFFS[_block_escalation]
                        print(f"  ! captcha error ({strikes}/{CAPTCHA_LIMIT}), "
                              f"backing off {_fmt_wait(wait)}", flush=True)
                        sleep(wait)
                        continue
                    return {"ok": False, "error": msg, "payload": payload}
                note_clean()
                return {"ok": True, "payload": payload}
            if r.status_code in (429, 503):
                note_throttled(str(r.status_code))
                wait = 30 * (attempt + 1)
                print(f"  ! {r.status_code} rate-limited, waiting {wait}s", flush=True)
                sleep(wait)
                continue
            print(f"  ! HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return {"ok": False, "error": f"HTTP {r.status_code}", "payload": None}
        except requests.RequestException as e:
            last_err = e
            wait = 10 * (attempt + 1)
            print(f"  ! network error ({e}), retry in {wait}s", flush=True)
            sleep(wait)
    raise RuntimeError(f"request to {path} failed after {max_retries} retries: {last_err}")


def fetch_detail_page(page_size: int, page: int, criteria: dict) -> dict:
    data = dict(criteria)
    data["pageSize"] = str(page_size)
    data["pageNumber"] = str(page)
    return post_json("/aramadetaylist", data)


def get_rows(result: dict):
    inner = (result.get("payload") or {}).get("data") or {}
    return inner.get("data") or [], inner.get("recordsTotal") or 0, inner.get("recordsFiltered") or 0


def getDokuman(dok_id: str, aranan_kelime: str) -> str:
    """Fetch full decision HTML. Returns raw page HTML ('' on failure)."""
    url = f"{BASE}/getDokuman?id={quote(dok_id)}&arananKelime={quote(aranan_kelime or ' ')}"
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 200:
                note_clean()
                return r.text
            if r.status_code in (429, 503):
                note_throttled(str(r.status_code))
                sleep(30 * (attempt + 1))
                continue
            print(f"  ! getDokuman HTTP {r.status_code} for id={dok_id}", flush=True)
            return ""
        except requests.RequestException as e:
            sleep(10 * (attempt + 1))
    return ""


def html_to_text(page_html: str) -> tuple[str, str]:
    """Extract (visible_html, plain_text) from a getDokuman page.

    Full text lives in <p id="hiddencontent"> as HTML-escaped HTML.
    If that is missing, fall back to the #content region.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    hidden = soup.find("p", id="hiddencontent")
    if hidden and hidden.string:
        raw_html = unescape(hidden.string)
    else:
        content = soup.find(id="content")
        raw_html = content.decode_contents() if content else soup.decode()
    inner = BeautifulSoup(raw_html, "html.parser")
    for tag in inner(["script", "style"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", inner.get_text("\n")).strip()
    return raw_html.strip(), text


def load_done() -> set:
    done = set()
    if INDEX_FILE.exists():
        with open(INDEX_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(str(json.loads(line)["id"]))
                except Exception:
                    pass
    return done


def append_index(row: dict):
    with open(INDEX_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_meta(obj: dict):
    with open(META_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def crawl(args):
    global BASE_DELAY
    BASE_DELAY = args.delay
    _load_throttle()
    OUT_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    done = load_done() if args.resume else set()
    total_saved_before = len(done)

    if args.mode == "keyword":
        def builder(page_size, page):
            data = {"pageSize": str(page_size), "pageNumber": str(page)}
            if args.kelime:
                data["andKelimeler"] = args.kelime
            if args.or_kelime:
                data["orKelimeler"] = args.or_kelime
            if args.not_kelime:
                data["notAndKelimeler"] = args.not_kelime
            return data
        list_fetch = lambda ps, pg: post_json("/aramalist", builder(ps, pg))
        desc = f"keyword: AND={args.kelime} OR={args.or_kelime} NOT={args.not_kelime}"
    else:
        criteria = {}
        if args.daire:
            criteria["daire"] = args.daire
        if args.esas_yil:
            criteria["esasYil"] = str(args.esas_yil)
        if args.esas_ilk:
            criteria["esasIlkSiraNo"] = str(args.esas_ilk)
        if args.esas_son:
            criteria["esasSonSiraNo"] = str(args.esas_son)
        if args.karar_yil:
            criteria["kararYil"] = str(args.karar_yil)
        if args.karar_ilk:
            criteria["kararIlkSiraNo"] = str(args.karar_ilk)
        if args.karar_son:
            criteria["kararSonSiraNo"] = str(args.karar_son)
        if args.baslangic:
            criteria["baslangicTarihi"] = args.baslangic
        if args.bitis:
            criteria["bitisTarihi"] = args.bitis
        if args.mevzuat_no:
            criteria["mevzuatNumarasi"] = args.mevzuat_no
        if args.mevzuat_adi:
            criteria["mevzuatAdi"] = args.mevzuat_adi
        if args.madde:
            criteria["madde"] = args.madde
        criteria["siralama"] = str(args.siralama)
        criteria["siralamaDirection"] = args.direction
        list_fetch = lambda ps, pg: fetch_detail_page(ps, pg, criteria)
        desc = f"detail: {criteria}"

    print(f"=== Danıştay crawl | {desc}", flush=True)
    probe = list_fetch(1, 1)
    if not probe["ok"]:
        print(f"ERROR: {probe['error']}")
        sys.exit(1)
    rows, total, filtered = get_rows(probe)
    print(f"recordsTotal={total} (filtered={filtered})", flush=True)
    if args.dry_run:
        print(json.dumps(rows[:3], ensure_ascii=False, indent=1))
        return
    if args.limit and args.limit < total:
        total = args.limit
        print(f"(capped to --limit {args.limit})", flush=True)

    page_size = min(args.page_size, 100)
    saved = 0
    page = 1
    try:
        while True:
            result = list_fetch(page_size, page)
            if not result["ok"]:
                print(f"ERROR on page {page}: {result['error']}")
                break
            rows, total, _ = get_rows(result)
            if not rows:
                break
            for row in rows:
                rid = str(row["id"])
                if rid in done:
                    continue
                # save metadata first (cheap), then full text
                append_index(row)
                done.add(rid)
                append_meta({"ts": int(time.time()), "id": rid,
                             "crawl_desc": desc})
                if not args.no_fulltext:
                    html = getDokuman(rid, row.get("arananKelime", ""))
                    if html:
                        raw_html, text = html_to_text(html)
                        (DOCS_DIR / f"{rid}.html").write_text(raw_html, encoding="utf-8")
                        (DOCS_DIR / f"{rid}.txt").write_text(
                            f"Danıştay | {row.get('daireKurul')} | E:{row.get('esasNo')} "
                            f"K:{row.get('kararNo')} T:{row.get('kararTarihi')}\n"
                            f"{'=' * 60}\n{text}\n", encoding="utf-8")
                        sleep(current_delay())
                    else:
                        print(f"  ! no full text for id={rid}", flush=True)
                saved += 1
                if saved % 25 == 0:
                    print(f"  saved {saved} (total done {len(done)})", flush=True)
                if args.limit and saved >= args.limit:
                    print("reached --limit")
                    return
            if args.limit and len(done) - total_saved_before >= args.limit:
                return
            if page * page_size >= total:
                break
            page += 1
            sleep(current_delay())
    except CaptchaError as e:
        print(f"\n!! {e}\nStopping. Re-run with --resume to continue.", file=sys.stderr)
        sys.exit(2)

    print(f"\nDone. saved={saved} index_total={len(done)}")
    print(f"Index: {INDEX_FILE}\nDocs:  {DOCS_DIR}")


def main():
    p = argparse.ArgumentParser(description="Danıştay karar arama crawler")
    p.add_argument("mode", choices=["keyword", "detail"])
    # keyword mode
    p.add_argument("kelime", nargs="*", help="AND keywords (keyword mode)")
    p.add_argument("--or", dest="or_kelime", nargs="*", help="OR keywords")
    p.add_argument("--not", dest="not_kelime", nargs="*", help="NOT keywords")
    # detail mode
    p.add_argument("--daire", help='e.g. "1. Daire"')
    p.add_argument("--esas-yil", type=int)
    p.add_argument("--esas-ilk", type=int)
    p.add_argument("--esas-son", type=int)
    p.add_argument("--karar-yil", type=int)
    p.add_argument("--karar-ilk", type=int)
    p.add_argument("--karar-son", type=int)
    p.add_argument("--baslangic", help="dd.MM.yyyy")
    p.add_argument("--bitis", help="dd.MM.yyyy")
    p.add_argument("--mevzuat-no")
    p.add_argument("--mevzuat-adi")
    p.add_argument("--madde")
    p.add_argument("--siralama", type=int, default=3, help="1=esas 2=karar 3=tarih")
    p.add_argument("--direction", choices=["asc", "desc"], default="desc")
    # common
    p.add_argument("--page-size", type=int, default=50)
    p.add_argument("--limit", type=int, default=0, help="max docs to save (0=all)")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between full-text fetches")
    p.add_argument("--no-fulltext", action="store_true", help="only save the index, no full text")
    p.add_argument("--resume", action="store_true", help="skip ids already in index.jsonl")
    p.add_argument("--dry-run", action="store_true", help="just show total count")
    args = p.parse_args()
    crawl(args)


if __name__ == "__main__":
    main()
