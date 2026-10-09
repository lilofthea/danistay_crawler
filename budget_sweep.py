#!/usr/bin/env python3
"""
Pick the context size (tokens) for mask labelling: label the same decisions
with the full text and with sentence chunks of several token budgets, and
measure how often each agrees with
  pipeline  the pipeline's final labels (regex + LLM fix + review, full text)
  full      the same LLM and prompt given the whole text

Decisions: <src>/ids.txt (default data/pipeline_50). Results are cached per decision and
budget in <out>/raw/, so an interrupted run continues.

Usage:  python3 budget_sweep.py [--budgets 128 256 384 512 768 1024] [--workers 4]
        python3 budget_sweep.py --src data/pipeline_500 --out data/budget_sweep_500
Output: <out>/report.txt
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mask_types import masks
from window_try import chunk_labels, full_labels

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "pipeline_50"
OUT = ROOT / "data" / "budget_sweep"   # changed by --src / --out


def run(doc_id: str, budget):
    """Labels for one decision at one budget ("full" = whole text), cached."""
    f = OUT / "raw" / f"{doc_id}.{budget}.json"
    if f.exists():
        return json.loads(f.read_text())
    text = (ROOT / "data" / "docs" / f"{doc_id}.txt").read_text(encoding="utf-8")
    spans = list(masks(text))
    if budget == "full":
        labels, sent = full_labels(text, spans), None
    else:
        labels, chunks = chunk_labels(text, spans, budget)
        sent = sum(t for t, _, _ in chunks)
    res = {"labels": {str(k): v for k, v in labels.items()}, "tokens_sent": sent}
    f.write_text(json.dumps(res))
    return res


def main():
    global SRC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 384, 512, 768, 1024])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--src", type=Path, default=SRC, help="pipeline output with ids.txt + masks.jsonl")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    SRC, OUT = args.src, args.out
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    ids = (SRC / "ids.txt").read_text().split()
    pipe = defaultdict(dict)
    for r in map(json.loads, open(SRC / "masks.jsonl", encoding="utf-8")):
        pipe[r["id"]][str(r["n"])] = r["final"]

    jobs = [(d, b) for d in ids for b in ["full"] + args.budgets]
    with ThreadPoolExecutor(args.workers) as pool:
        results = {}
        for k, ((d, b), res) in enumerate(zip(jobs, pool.map(lambda j: run(*j), jobs)), 1):
            results[(d, b)] = res
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} done", flush=True)

    total = sum(len(pipe[d]) for d in ids)
    full_vs_pipe = sum(results[(d, "full")]["labels"].get(n) == lab
                       for d in ids for n, lab in pipe[d].items())
    lines = [f"{len(ids)} karar, {total} sansür", "",
             f"{'bağlam':<10}{'pipeline ile':>15}{'tam metin ile':>16}{'gönderilen token':>19}",
             f"{'tam metin':<10}{full_vs_pipe / total:>14.1%}{'—':>16}{'—':>19}"]
    per = {}
    for b in args.budgets:
        vs_pipe = sum(results[(d, b)]["labels"].get(n) == lab
                      for d in ids for n, lab in pipe[d].items())
        vs_full = sum(results[(d, b)]["labels"].get(n) == results[(d, "full")]["labels"].get(n)
                      for d in ids for n in pipe[d])
        sent = sum(results[(d, b)]["tokens_sent"] or 0 for d in ids)
        per[b] = (vs_pipe / total, vs_full / total)
        lines.append(f"{str(b) + ' token':<10}{vs_pipe / total:>14.1%}{vs_full / total:>15.1%}{sent:>19,}")

    # disagreements with the pipeline, for the budgets around the choice
    lines += ["", "Pipeline'dan farklı etiketler (bütçe başına):"]
    for b in args.budgets:
        diff = [(d, n, lab, results[(d, b)]["labels"].get(n)) for d in ids
                for n, lab in pipe[d].items() if results[(d, b)]["labels"].get(n) != lab]
        lines.append(f"\n## {b} token: {len(diff)} fark")
        lines += [f"  {d} #{n}: pipeline {lab}, parça {got}" for d, n, lab, got in diff]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len(args.budgets)]))


if __name__ == "__main__":
    main()
