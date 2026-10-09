"""
Danıştay producer — wired to the LIVE danistay crawler dataset.

Source (env DANISTAY_DATA, default /home/ayilik/Desktop/projects/danistay/data):
  data/index.jsonl      one metadata object per decision (id, esasNo, kararNo, ...)
  data/docs/<id>.txt    full text of each decision

Incremental: remembers the last processed index.jsonl line in
legal_ingestion/.state/danistay.state, so cron runs only emit NEW decisions
(the crawler keeps appending as it works through 2026 → 1983).

Run:
  python -m producers.danistay_producer            # process new docs since last run
  python -m producers.danistay_producer --max-rows 2000
  python -m producers.danistay_producer --reset    # re-scan from line 0 (bloom still dedups)
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path

from producers.base_producer import LegalDocumentProducer
from schemas.document import RawDocument

logger = logging.getLogger(__name__)

DATA = Path(os.getenv("DANISTAY_DATA", "/home/ayilik/Desktop/projects/danistay/data"))
STATE = Path(os.getenv("LEGAL_STATE_DIR",
                       "/home/ayilik/Desktop/projects/legal_ingestion/.state")) / "danistay.state"
BASE_URL = "https://karararama.danistay.gov.tr"

_DDMMYYYY = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")


def to_iso_date(s: str) -> str:
    """Site gives dd.MM.yyyy — normalize to ISO for downstream."""
    if not s:
        return None
    m = _DDMMYYYY.match(s.strip())
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else s


def load_state() -> int:
    if STATE.exists():
        try:
            return int(STATE.read_text().strip() or 0)
        except ValueError:
            return 0
    return 0


def save_state(line_no: int):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(str(line_no))


def iter_new_index_lines(start_line: int, max_rows: int = 0):
    """Yield (line_no, meta_dict) for new index.jsonl lines."""
    path = DATA / "index.jsonl"
    if not path.exists():
        logger.warning(f"index file missing: {path}")
        return
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < start_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON at index.jsonl line {i}, skipping")
            if max_rows and (i - start_line + 1) >= max_rows:
                break


def run(max_rows: int = 0, reset: bool = False, no_dedup: bool = False):
    producer = LegalDocumentProducer(source="danistay")
    produced = skipped = missing = 0
    start_line = 0 if reset else load_state()
    last_line = start_line

    logger.info(f"Danıştay scan from index line {start_line} (data: {DATA})")

    try:
        for line_no, meta in iter_new_index_lines(start_line, max_rows):
            doc_id = meta.get("id", "")
            txt_path = DATA / "docs" / f"{doc_id}.txt"
            if not txt_path.exists():
                missing += 1      # crawler hasn't fetched full text yet — next run picks it up
                last_line = line_no + 1
                continue
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                last_line = line_no + 1
                continue

            doc = RawDocument(
                id=f"danistay:{doc_id}",          # deterministic (was: random UUID)
                kaynak="danistay",
                url=f"{BASE_URL}/arama/getDokuman?id={doc_id}",
                text=text,
                esas_no=meta.get("esasNo"),
                karar_no=meta.get("kararNo"),
                daire=meta.get("daireKurul"),
                karar_tarihi=to_iso_date(meta.get("kararTarihi", "")),
                metadata={"doc_id": doc_id, "aranan_kelime": meta.get("arananKelime")},
            )
            # a failed delivery raises: the checkpoint stays on this line,
            # so the next run retries it
            if producer.produce(doc, force=no_dedup):
                produced += 1
            else:
                skipped += 1
            last_line = line_no + 1
            save_state(last_line)   # checkpoint every doc (cheap)
    finally:
        save_state(last_line)
        producer.flush()
        producer.close()

    logger.info(f"Danıştay run complete: {produced} produced, {skipped} dedup, "
                f"{missing} awaiting full text (next run)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=0, help="cap docs per run (0 = all new)")
    ap.add_argument("--reset", action="store_true", help="rescan from index line 0")
    ap.add_argument("--no-dedup", action="store_true",
                    help="send even if already marked seen (re-send lost docs)")
    args = ap.parse_args()
    run(max_rows=args.max_rows, reset=args.reset, no_dedup=args.no_dedup)
