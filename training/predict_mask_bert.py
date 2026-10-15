#!/usr/bin/env python3
"""
Label the "..." masks of a raw Danıştay decision with the trained model.

Masks are found the same way as mask_types.py ("…" or two or more dots; in
"Av...." the abbreviation dot is kept), replaced by [SANSUR], and the text is
read in overlapping 512-token windows (stride 128); each mask takes the
prediction from the window where it sits furthest from the edges.

Usage:
  python predict_mask_bert.py --model runs/v1/best decision.txt            # print labelled text
  python predict_mask_bert.py --model runs/v1/best --json decision.txt     # print labels
  python predict_mask_bert.py --model runs/v1/best kararlar/ --out-dir etiketli/
        # every .txt in kararlar/ -> etiketli/<name>.txt + <name>.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, BertTokenizerFast

from train_mask_bert import MASK_TOKEN, tr_lower

MASK_RE = re.compile(r"…+(?:\.+)?|\.{2,}")
PLACEHOLDER = {"PER": "[KİŞİ]", "ORG": "[KURUM]", "LOC": "[YER]", "COU": "[MAHKEME]",
               "DAT": "[TARİH]", "REF": "[ATIF]", "MON": "[TUTAR]", "ELLIPSIS": "…",
               "MASK": "[SANSÜR]"}


def find_masks(text):
    for m in MASK_RE.finditer(text):
        s = m.start()
        if text[s] == "." and m.end() - s >= 4 and s and text[s - 1].isalpha():
            s += 1                       # "Av...." -> keep "Av."
        yield s, m.end()


def label_text(text, tok, model):
    """Return (mask spans, predicted labels) for one decision text."""
    sansur = tok.convert_tokens_to_ids(MASK_TOKEN)
    spans = list(find_masks(text))
    if not spans:
        return spans, []
    masked, pos = [], 0
    for s, e in spans:
        masked += [text[pos:s], f" {MASK_TOKEN} "]
        pos = e
    masked = "".join(masked + [text[pos:]])

    lowered = tr_lower(masked)
    # character position of each [SANSUR] in the text -> mask index
    mask_at = {m.start(): i for i, m in enumerate(re.finditer(re.escape(MASK_TOKEN), lowered))}
    enc = tok(lowered, truncation=True, max_length=512, stride=128,
              return_overflowing_tokens=True, return_offsets_mapping=True,
              padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()
                          if k in ("input_ids", "attention_mask", "token_type_ids")}).logits.cpu()
    best = {}                            # mask index -> (distance from window edge, label id)
    for w in range(enc["input_ids"].shape[0]):
        n_real = int(enc["attention_mask"][w].sum())
        for p, t in enumerate(enc["input_ids"][w].tolist()):
            if t != sansur:
                continue
            idx = mask_at.get(int(offsets[w, p, 0]))
            if idx is None:
                continue
            dist = min(p, n_real - 1 - p)
            if idx not in best or dist > best[idx][0]:
                best[idx] = (dist, int(logits[w, p].argmax()))
    return spans, [model.config.id2label[best[i][1]] if i in best else "MASK"
                   for i in range(len(spans))]


def render(text, spans, labels):
    out, pos = [], 0
    for (s, e), l in zip(spans, labels):
        out += [text[pos:s], PLACEHOLDER[l]]
        pos = e
    return "".join(out + [text[pos:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="decision .txt files or folders (default: stdin)")
    ap.add_argument("--model", required=True, help="trained model dir (runs/<name>/best)")
    ap.add_argument("--out-dir", type=Path,
                    help="write <name>.txt (labelled text) and <name>.json (labels) here; "
                         "without it, a single input is printed")
    ap.add_argument("--json", action="store_true", help="print labels as JSON instead of text")
    args = ap.parse_args()

    tok = BertTokenizerFast.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    if torch.cuda.is_available():
        model.to("cuda")

    files = []
    for p in map(Path, args.inputs):
        files += sorted(p.glob("*.txt")) if p.is_dir() else [p]

    if not args.out_dir:
        if len(files) > 1:
            sys.exit("several inputs: give --out-dir to save them")
        text = files[0].read_text(encoding="utf-8") if files else sys.stdin.read()
        spans, labels = label_text(text, tok, model)
        if args.json:
            print(json.dumps([{"start": s, "end": e, "label": l}
                              for (s, e), l in zip(spans, labels)], ensure_ascii=False, indent=1))
        else:
            print(render(text, spans, labels))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(files, 1):
        text = f.read_text(encoding="utf-8")
        spans, labels = label_text(text, tok, model)
        (args.out_dir / f"{f.stem}.txt").write_text(render(text, spans, labels), encoding="utf-8")
        (args.out_dir / f"{f.stem}.json").write_text(json.dumps(
            [{"start": s, "end": e, "label": l} for (s, e), l in zip(spans, labels)],
            ensure_ascii=False), encoding="utf-8")
        print(f"[{k}/{len(files)}] {f.name}: {len(spans)} sansür", flush=True)


if __name__ == "__main__":
    main()
