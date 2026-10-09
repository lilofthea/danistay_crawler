#!/usr/bin/env python3
"""
Three-step labelling of the anonymisation masks ("...") in a random sample of
decisions, all written to a local folder (never to data/docs, so the
publisher can't send any of it):

  1. regex   mask_types.py labels what its rules recognise, the rest is "?"
  2. fix     the LLM sees the original text with every mask numbered plus the
             regex suggestions; it fills the "?" ones and corrects wrong ones
  3. review  the LLM sees the text with the step-2 labels written in and
             returns corrections for any label it finds wrong

Output (data/pipeline_<n>/):
  ids.txt            the sampled decision ids (reuse with --ids)
  original/<id>.txt  untouched copy
  regex/<id>.txt     after step 1
  fix/<id>.txt       after step 2
  final/<id>.txt     after step 3; labels that differ from regex are starred: [KİŞİ*]
  masks.jsonl        per mask: regex, fix, final labels + context
  changes.txt        every label the LLM filled or changed, grouped by step
  summary.txt        counts

Usage:
  python3 label_pipeline.py                  # 50 random decisions
  python3 label_pipeline.py -n 20 --seed 3
  python3 label_pipeline.py --ids data/pipeline_50/ids.txt --out data/pipeline_50_v2
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mask_types import PLACEHOLDER

from labeling_pipeline import ERROR_PAGE_MARKER, LABELS, label_text, render

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_FILE = ROOT / "data" / "index.jsonl"


def pick(n: int, seed: int) -> list[str]:
    """Random decisions with real text, excluding the set the regex rules were
    tuned on (data/labeled_100), so the test is on unseen decisions."""
    tuned = {p.stem for p in (ROOT / "data" / "labeled_100" / "original").glob("*.txt")}
    ids = []
    with open(INDEX_FILE, encoding="utf-8") as f:
        for line in f:
            doc_id = str(json.loads(line)["id"])
            p = DOCS_DIR / f"{doc_id}.txt"
            if (doc_id not in tuned and p.exists()
                    and ERROR_PAGE_MARKER not in p.read_text(encoding="utf-8")):
                ids.append(doc_id)
    random.Random(seed).shuffle(ids)
    return ids[:n]


def label_doc(doc_id: str):
    text = (DOCS_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
    # same steps as the production pipeline; keep earlier labels on LLM errors
    res = label_text(text, strict=False)
    spans, regex, fix, final, errors = (res[k] for k in ("spans", "regex", "fix", "final", "errors"))

    rows = [{"id": doc_id, "n": i + 1, "start": s, "end": e,
             "regex": a, "fix": b, "final": c,
             "context": " ".join((text[max(0, s - 60):s] + "⟦" + text[s:e] + "⟧"
                                  + text[e:e + 40]).split())}
            for i, ((s, e), a, b, c) in enumerate(zip(spans, regex, fix, final))]
    star = lambda lab, changed: (PLACEHOLDER[lab][:-1] + "*]"
                                 if changed and PLACEHOLDER[lab].endswith("]")
                                 else PLACEHOLDER[lab] + ("*" if changed else ""))
    texts = {
        "regex": render(text, spans, [PLACEHOLDER[l] for l in regex]),
        "fix": render(text, spans, [PLACEHOLDER[l] for l in fix]),
        "final": render(text, spans, [star(c, c != a) for a, c in zip(regex, final)]),
    }
    return doc_id, text, texts, rows, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ids", type=Path, help="file with decision ids to use instead of sampling")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=4, help="parallel documents")
    args = ap.parse_args()

    ids = (args.ids.read_text().split() if args.ids else pick(args.n, args.seed))
    out = args.out or ROOT / "data" / f"pipeline_{len(ids)}"
    if out.exists():
        sys.exit(f"{out} already exists; remove it or pass --out")
    for sub in ("original", "regex", "fix", "final"):
        (out / sub).mkdir(parents=True)
    (out / "ids.txt").write_text("\n".join(ids) + "\n")

    rows, errors = [], []
    with ThreadPoolExecutor(args.workers) as pool:
        for k, (doc_id, text, texts, doc_rows, errs) in enumerate(pool.map(label_doc, ids), 1):
            shutil.copy2(DOCS_DIR / f"{doc_id}.txt", out / "original" / f"{doc_id}.txt")
            for sub, t in texts.items():
                (out / sub / f"{doc_id}.txt").write_text(t, encoding="utf-8")
            rows += doc_rows
            errors += [f"{doc_id}: {e}" for e in errs]
            print(f"[{k}/{len(ids)}] {doc_id}: {len(doc_rows)} sansür, "
                  f"adım2 {sum(r['fix'] != r['regex'] for r in doc_rows)} değişiklik, "
                  f"adım3 {sum(r['final'] != r['fix'] for r in doc_rows)} değişiklik"
                  + (f"  HATA: {errs}" if errs else ""), flush=True)

    with open(out / "masks.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # changes.txt: what each LLM step did, grouped by (from -> to)
    lines = []
    for title, a, b in (("ADIM 2 (model: doldurma + düzeltme)", "regex", "fix"),
                        ("ADIM 3 (model: kontrol)", "fix", "final")):
        groups = defaultdict(list)
        for r in rows:
            if r[a] != r[b]:
                groups[(r[a], r[b])].append(r)
        lines += [f"===== {title}: {sum(map(len, groups.values()))} değişiklik", ""]
        for (x, y), rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"## {x} -> {y}: {len(rs)}")
            lines += [f"  {r['id']} #{r['n']}  {r['context']}" for r in rs]
            lines.append("")
    (out / "changes.txt").write_text("\n".join(lines), encoding="utf-8")

    n = len(rows)
    filled = sum(r["regex"] == "MASK" and r["fix"] != "MASK" for r in rows)
    corrected = sum(r["regex"] != "MASK" and r["fix"] != r["regex"] for r in rows)
    reviewed = sum(r["final"] != r["fix"] for r in rows)
    s = [f"{len(ids)} karar, {n} sansür", "",
         f"Adım 1 regex : {sum(r['regex'] != 'MASK' for r in rows)} etiketledi, "
         f"{sum(r['regex'] == 'MASK' for r in rows)} boş bıraktı",
         f"Adım 2 model : {filled} boşu doldurdu, {corrected} regex etiketini değiştirdi",
         f"Adım 3 model : {reviewed} etiketi değiştirdi",
         f"Sonuçta regex'ten farklı: {sum(r['final'] != r['regex'] for r in rows)}", "",
         f"  {'':<10}{'regex':>8}{'adım2':>8}{'son':>8}"]
    for lab in LABELS:
        s.append(f"  {PLACEHOLDER[lab]:<10}" + "".join(
            f"{sum(r[m] == lab for r in rows):>8}" for m in ("regex", "fix", "final")))
    if errors:
        s += ["", "Model hataları (o adımda önceki etiketler korundu):"] + errors
    (out / "summary.txt").write_text("\n".join(s) + "\n", encoding="utf-8")
    print("\n" + "\n".join(s))


if __name__ == "__main__":
    main()
