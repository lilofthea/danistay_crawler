#!/usr/bin/env python3
"""
Labelling pipeline for crawled Danıştay decisions, all steps in one place.

  1. fetch    (crawl_range.sh / crawler.py, already running) writes
              data/index.jsonl + data/docs/<id>.txt; this script picks up
              every decision added there since its last run
  2. regex    mask_types.py labels the "..." masks its rules recognise
  3. llm fix  the LLM sees the text with every mask numbered plus the regex
              labels; it fills the unknown ones and corrects wrong ones
  4. review   the LLM sees the labelled text and corrects any label it finds
              wrong
  5. publish  the labelled decision goes to its own Redpanda topic

Steps 2-4 write each finished decision to a local outbox
(data/labeled/labeled.jsonl, plus data/labeled/<id>.txt for reading).
Step 5 sends the outbox to Redpanda when LABELED_BROKERS is set; until then
it only labels, and the backlog is sent once the broker is configured.
If the LLM is unreachable the run stops and the same decisions are retried
next time, so nothing half-labelled is ever written.

Config (env.sh):
  LABELED_BROKERS  Redpanda for labelled output, e.g. "10.150.41.5:19093"
                   (unset = label only, don't publish)
  LABELED_TOPIC    default "legal.danistay.labeled"
  LLM_API          default the vLLM server below
  LABELED_DIR      outbox folder, default data/labeled

Usage:
  python3 labeling_pipeline.py                # label new decisions + publish, then exit
  python3 labeling_pipeline.py --loop         # keep going, check every 5 min
  python3 labeling_pipeline.py --limit 5      # only the next 5 decisions (testing)
  python3 labeling_pipeline.py --publish-only # just send the outbox
State: .state/labeling.state (index lines done), .state/labeling_published.state
Review log: logs/review_changes.log - every label the review (step 4) changed
compared to the LLM fix (step 3), with the decision id and context
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from mask_types import PLACEHOLDER, classify, masks

ROOT = Path(__file__).parent
DATA = Path(os.getenv("DANISTAY_DATA", ROOT / "data"))
INDEX_FILE = DATA / "index.jsonl"
DOCS_DIR = DATA / "docs"
OUT_DIR = Path(os.getenv("LABELED_DIR", DATA / "labeled"))
OUTBOX = OUT_DIR / "labeled.jsonl"
STATE_DIR = Path(os.getenv("LEGAL_STATE_DIR", ROOT / ".state"))
LABEL_STATE = STATE_DIR / "labeling.state"
PUBLISH_STATE = STATE_DIR / "labeling_published.state"
REVIEW_LOG = ROOT / "logs" / "review_changes.log"

API = os.getenv("LLM_API", "http://10.150.96.44:8112/v1/chat/completions")
MODEL = "/model"   # vLLM serves Qwen/Qwen3.6-27B-FP8 under this id
BROKERS = [b for b in os.getenv("LABELED_BROKERS", "").split(",") if b]
TOPIC = os.getenv("LABELED_TOPIC", "legal.danistay.labeled")
PIPELINE_VERSION = "1"   # bump when rules or prompts change meaningfully

ERROR_PAGE_MARKER = "Ana sayfaya gitmek için tıklayınız"
BASE_URL = "https://karararama.danistay.gov.tr"
LABELS = ["PER", "ORG", "LOC", "COU", "DAT", "REF", "MON", "ELLIPSIS", "MASK"]

# Shared by both LLM steps; conventions match mask_types.py.
LABEL_DEFS = """Etiketler:
- PER: kişi adı veya kişiyi tanımlayan bilgi (davacı, avukat, hukuk müşaviri, tanık,
  mirasçı, kod adı, kullanıcı adı). Taraf başlıklarında ("DAVACI : ⟦n⟧") kurum adı
  geçmiyorsa kişidir.
- ORG: kurum, şirket, idare, sendika, okul, marka, yazılım/program adı ya da bu adın
  bir parçası. Kurum adının içindeki şehir/ilçe de ORG'dur ("⟦n⟧ Vergi Dairesi Müdürlüğü",
  "⟦n⟧ Barosu" -> ORG).
- LOC: yer (il, ilçe, mahalle, köy, mezra, sokak, adres) ve ada/parsel/pafta/blok/kat
  numaraları. Kurum adından sonra "/" veya "-" ile gelen şehir de LOC'tur
  ("... Bakanlığı / ⟦n⟧" -> LOC).
- COU: mahkeme adı veya adının parçası; mahkeme adındaki şehir de COU'dur
  ("⟦1⟧ İdare Mahkemesi", "⟦2⟧ Bölge İdare Mahkemesi ⟦3⟧ İdari Dava Dairesi" -> COU, COU, COU).
- DAT: tarih.
- REF: numara (dosya, esas, karar, işlem, belge, plaka, telefon, IMEI, GSM, ID, vergi
  kimlik, ruhsat, takip, ödeme emri numarası), "⟦n⟧ sayılı" listeleri dahil.
- MON: para tutarı.
- ELLIPSIS: sansür değil; alıntıda atlanmış metin (tırnak içindeki veya mevzuat
  alıntısındaki "(...)", "..."). Tırnak içinde bile olsa ardından "adında/isimli bir
  kişi" geliyorsa PER'dir.
- MASK: bağlamdan tür çıkarılamıyor (ör. sansürlenmiş unvan/meslek)."""

FIX_SYSTEM = f"""Sen Türk idare hukuku kararları üzerinde çalışan bir anonimleştirme uzmanısın.
Danıştay kararlarında kişisel bilgiler "..." ile sansürlenmiştir. Sana verilen metinde her
sansürlü yer ⟦n⟧ şeklinde numaralandırıldı. Metnin altında, kural tabanlı bir sistemin her
numara için önerdiği etiket var; "?" önerisi olmayan yerdir.

Görevin:
- "?" olan her numaraya bağlama bakarak etiket ver.
- Önerilen etiket yanlışsa düzelt; doğruysa aynen bırak. Emin değilsen öneriyi koru.

{LABEL_DEFS}

Metni değiştirme, gerçek değeri tahmin etme; sadece türü ver.
Cevabı sadece JSON olarak ver ve TÜM numaraları içersin: {{"1": "PER", "2": "DAT", ...}}"""

REVIEW_SYSTEM = f"""Sen Türk idare hukuku kararları üzerinde çalışan bir anonimleştirme uzmanısın.
Danıştay kararlarındaki "..." sansürleri türlerine göre etiketlendi. Sana verilen metinde
her sansür ⟦n:ETİKET⟧ şeklinde yazılı. Her etiketi bağlamına göre kontrol et.

{LABEL_DEFS}

Sadece YANLIŞ olan etiketleri doğru etiketle birlikte ver. Emin olmadığın etiketi değiştirme.
Cevabı sadece JSON olarak ver: {{"7": "LOC", "12": "PER"}}. Hepsi doğruysa {{}} ver."""


# ------------------------------------------------------------------ helpers

BAD_ANSWERS_LOG = ROOT / "logs" / "llm_bad_answers.log"


def normalize(answer) -> dict:
    """Bring the model's JSON to {"1": "PER", ...}: unwrap a single nested
    object ({"labels": {...}}), and accept keys like "⟦1⟧", "#1", "1."."""
    if isinstance(answer, dict) and len(answer) == 1:
        ((key, inner),) = answer.items()
        if isinstance(inner, dict) and not re.search(r"\d", str(key)):
            answer = inner
    if not isinstance(answer, dict):
        return {}
    out = {}
    for k, v in answer.items():
        m = re.search(r"\d+", str(k))
        if m:
            out[m.group(0)] = v.get("label", "") if isinstance(v, dict) else v
    return out


def chat(system: str, user: str, expect=None) -> dict:
    """One LLM call returning {"<n>": "<LABEL>"}.

    Retried (3 tries) when the answer has no usable JSON, or - if `expect`
    lists the numbers that must be answered - when fewer than half of them
    are. Unusable answers are written to logs/llm_bad_answers.log. Earlier
    this returned {} silently, which skipped the LLM step for ~9% of
    decisions in the 500-decision test."""
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 8000,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    expect = [str(n) for n in expect] if expect is not None else None
    best, problem = {}, ""
    for attempt in range(3):
        req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            choice = json.loads(r.read())["choices"][0]
        content = re.sub(r"(?s)<think>.*?</think>", "", choice["message"]["content"] or "")
        m = re.search(r"(?s)\{.*\}", content)
        try:
            ans = normalize(json.loads(m.group(0))) if m else None
        except json.JSONDecodeError:
            ans = None
        if ans is None:
            problem = f"no JSON (finish={choice.get('finish_reason')})"
        elif expect and sum(k in ans for k in expect) * 2 < len(expect):
            problem = f"{sum(k in ans for k in expect)}/{len(expect)} answered"
            best = max(best, ans, key=len)
        else:
            return ans
        BAD_ANSWERS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(BAD_ANSWERS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} try {attempt + 1}: {problem} | "
                    f"{content[:300]!r}\n")
    if best:
        return best          # partial answer: unanswered numbers keep earlier labels
    raise ValueError(f"unusable answer from model after 3 tries: {problem}")


def render(text: str, spans, reps) -> str:
    """Replace spans with reps; same spacing rules as mask_types.type_masks."""
    out, pos = [], 0
    for (s, e), rep in zip(spans, reps):
        out.append(text[pos:s])
        if not rep.startswith(("…", "⟦")):
            if s and (text[s - 1].isalnum() or text[s - 1] == "."):
                rep = " " + rep
            if e < len(text) and text[e].isalnum():
                rep += " "
        out.append(rep)
        pos = e
    out.append(text[pos:])
    return "".join(out)


def valid(answer: dict, k: int):
    lab = str(answer.get(str(k), "")).strip().upper()
    return lab if lab in LABELS else None


def to_iso(d: str):
    m = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})$", (d or "").strip())
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else d


# ------------------------------------------------------------ steps 2 - 4

def label_text(text: str, strict: bool = True) -> dict:
    """Run regex -> LLM fix -> LLM review on one decision's text.

    Returns spans and the labels after each step. With strict=True an LLM
    failure raises (production: retry later); with strict=False the previous
    step's labels are kept and the failure is listed in "errors".
    """
    spans = list(masks(text))
    regex = [classify(text, s, e) for s, e in spans]               # step 2
    fix, final, errors = list(regex), list(regex), []
    if not spans:
        return {"spans": spans, "regex": regex, "fix": fix, "final": final, "errors": errors}

    def fail(step, e):
        if strict:
            raise RuntimeError(f"{step}: {e}") from e
        errors.append(f"{step}: {e}")

    # step 3: fill the unknowns, correct wrong regex labels
    numbered = render(text, spans, [f"⟦{i}⟧" for i in range(1, len(spans) + 1)])
    hints = "\n".join(f"{i}: {'?' if lab == 'MASK' else lab}" for i, lab in enumerate(regex, 1))
    try:
        ans = chat(FIX_SYSTEM, f"METİN:\n{numbered}\n\nÖNERİLER:\n{hints}",
                   expect=range(1, len(spans) + 1))
        missing = sum(valid(ans, i + 1) is None for i in range(len(spans)))
        if missing:   # unanswered numbers keep the regex label
            errors.append(f"llm fix: {missing}/{len(spans)} numara cevapsız")
        fix = [valid(ans, i + 1) or regex[i] for i in range(len(spans))]
    except Exception as e:
        fail("llm fix", e)

    # step 4: review the labelled text
    final = list(fix)
    labelled = render(text, spans, [f"⟦{i}:{lab}⟧" for i, lab in enumerate(fix, 1)])
    try:
        ans = chat(REVIEW_SYSTEM, labelled)
        for k in ans:
            if str(k).isdigit() and 1 <= int(k) <= len(spans):
                final[int(k) - 1] = valid(ans, int(k)) or final[int(k) - 1]
    except Exception as e:
        fail("llm review", e)
    return {"spans": spans, "regex": regex, "fix": fix, "final": final, "errors": errors}


def label_decision(meta: dict) -> dict | None:
    """Label one index row; None if it has no real text (error page / missing)."""
    doc_id = str(meta["id"])
    path = DOCS_DIR / f"{doc_id}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if ERROR_PAGE_MARKER in text:
        return None
    res = label_text(text, strict=True)
    source = ["regex" if a == b == c else "llm_fix" if b == c else "llm_review"
              for a, b, c in zip(res["regex"], res["fix"], res["final"])]
    return {
        "id": f"danistay:{doc_id}",
        "doc_id": doc_id,
        "kaynak": "danistay",
        "url": f"{BASE_URL}/getDokuman?id={doc_id}",
        "esas_no": meta.get("esasNo"),
        "karar_no": meta.get("kararNo"),
        "daire": meta.get("daireKurul"),
        "karar_tarihi": to_iso(meta.get("kararTarihi")),
        "text": render(text, res["spans"], [PLACEHOLDER[l] for l in res["final"]]),
        "masks": [{"n": i + 1, "start": s, "end": e, "label": lab, "source": src,
                   "regex": a}
                  for i, ((s, e), lab, src, a)
                  in enumerate(zip(res["spans"], res["final"], source, res["regex"]))],
        "label_counts": dict(Counter(res["final"])),
        "pipeline_version": PIPELINE_VERSION,
        "model": "Qwen/Qwen3.6-27B-FP8",
        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warnings": res["errors"],
        # step 4 overruling step 3: logged to REVIEW_LOG, not published
        "_review_changes": [
            {"n": i + 1, "regex": a, "fix": b, "final": c,
             "context": " ".join((text[max(0, s - 70):s] + "⟦" + text[s:e] + "⟧"
                                  + text[e:e + 50]).split())}
            for i, ((s, e), a, b, c)
            in enumerate(zip(res["spans"], res["regex"], res["fix"], res["final"]))
            if b != c],
    }


# ------------------------------------------------------------------ state

def read_state(path: Path) -> int:
    try:
        return int(path.read_text().strip() or 0)
    except (FileNotFoundError, ValueError):
        return 0


def write_state(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(str(n))
    tmp.replace(path)


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


# ------------------------------------------------------- run steps 2 - 4

def label_new(limit: int = 0, workers: int = 4, batch: int = 20) -> int:
    """Label index lines added since the last run. Returns how many were done.

    Lines are processed in batches; a batch is written and checkpointed only
    when every decision in it succeeded, in index order.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = read_state(LABEL_STATE)
    with open(INDEX_FILE, encoding="utf-8") as f:
        lines = f.readlines()
    todo = lines[start:]
    if limit:
        todo = todo[:limit]
    if not todo:
        return 0
    log(f"labelling {len(todo)} decision(s) from index line {start}")
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for b in range(0, len(todo), batch):
            chunk = [json.loads(l) for l in todo[b:b + batch]]
            try:
                results = list(pool.map(label_decision, chunk))
            except Exception as e:
                log(f"stopping, LLM step failed (will retry these next run): {e}")
                break
            REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTBOX, "a", encoding="utf-8") as out, \
                    open(REVIEW_LOG, "a", encoding="utf-8") as rlog:
                for rec in results:
                    if rec is None:
                        continue
                    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    for c in rec.pop("_review_changes"):
                        rlog.write(f"{stamp}  {rec['doc_id']}  #{c['n']}  "
                                   f"{c['fix']} -> {c['final']}  (regex: {c['regex']})  "
                                   f"| {c['context']}\n")
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    (OUT_DIR / f"{rec['doc_id']}.txt").write_text(rec["text"], encoding="utf-8")
            done += len(chunk)
            write_state(LABEL_STATE, start + done)
            changed = sum(m["source"] != "regex" for r in results if r for m in r["masks"])
            log(f"  {start + done}/{len(lines)} labelled ({changed} label(s) from the LLM in this batch)")
    return done


# ---------------------------------------------------------------- step 5

def publish() -> int:
    """Send outbox lines not yet published. Returns how many were sent."""
    if not BROKERS:
        return 0
    if not OUTBOX.exists():
        return 0
    from kafka import KafkaProducer   # only needed once a broker is configured
    start = read_state(PUBLISH_STATE)
    with open(OUTBOX, encoding="utf-8") as f:
        pending = f.readlines()[start:]
    if not pending:
        return 0
    try:
        producer = KafkaProducer(bootstrap_servers=BROKERS, acks="all", retries=5,
                                 max_block_ms=10_000,
                                 key_serializer=lambda k: k.encode("utf-8"),
                                 value_serializer=lambda v: v.encode("utf-8"))
    except Exception as e:   # broker down: keep labelling, send next run
        log(f"publish skipped, broker unreachable ({BROKERS}): {e}")
        return 0
    sent = 0
    try:
        for line in pending:
            rec = json.loads(line)
            # block per message: the checkpoint only moves past delivered ones
            producer.send(TOPIC, key=rec["id"], value=line.rstrip("\n")).get(timeout=30)
            sent += 1
            write_state(PUBLISH_STATE, start + sent)
    except Exception as e:
        log(f"publish stopped after {sent} (will resume there next run): {e}")
    finally:
        producer.flush()
        producer.close()
    log(f"published {sent} labelled decision(s) to {TOPIC}")
    return sent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--loop", action="store_true", help="keep running, check every 5 min")
    ap.add_argument("--limit", type=int, default=0, help="label at most N decisions (testing)")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM requests")
    ap.add_argument("--publish-only", action="store_true", help="only send the outbox")
    args = ap.parse_args()
    if not BROKERS:
        log("LABELED_BROKERS not set: labelling only, publishing waits for a broker")
    while True:
        if not args.publish_only:
            label_new(limit=args.limit, workers=args.workers)
        publish()
        if not args.loop:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
