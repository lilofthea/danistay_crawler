#!/usr/bin/env python3
"""
Build the training set for a BERT model that labels anonymisation masks.

Source: the labelled decisions from Redpanda topic legal.danistay.masked,
downloaded to data/datasets/danistay_masked_raw.jsonl. In their text every
mask is written as a placeholder ([KİŞİ], [KURUM], ...) or "…" (omitted
quote text); the placeholder counts match the per-message category counts
exactly, so each one is one original mask.

Each placeholder is turned back into a single [SANSUR] token - what the model
will see at inference, where the input has "..." - and its label is kept:

  text   "... VEKİLİ : Av. [SANSUR] KARŞI TARAF (DAVACI) : [SANSUR] ..."
  labels ["PER", "PER"]            (one per [SANSUR], in order)

Steps: drop site error/captcha pages and decisions without masks, keep one
copy per decision, split into chunks of whole sentences of at most 510
BERTurk-Legal tokens ([CLS]/[SEP] make 512; sentences over the limit are
split at "; ", then ", ", then whitespace), drop chunks without masks, and
split train/validation/test by decision (80/10/10, by a hash of the id).
Exact-duplicate chunks (boilerplate) are kept once in train, and dropped
from validation/test when the same text is in train.

Usage:
  python3 build_mask_dataset.py
  python3 build_mask_dataset.py --src data/datasets/danistay_masked_raw.jsonl \\
      --out data/datasets/danistay_mask_v1 --max-tokens 510
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from tokenizers import BertWordPieceTokenizer

ROOT = Path(__file__).parent
VOCAB = ROOT / "data" / "datasets" / "tokenizer" / "KocLab-Bilkent_vocab.txt"
MASK_TOKEN = "[SANSUR]"

PLACEHOLDERS = {"[KİŞİ]": "PER", "[KURUM]": "ORG", "[YER]": "LOC", "[MAHKEME]": "COU",
                "[TARİH]": "DAT", "[ATIF]": "REF", "[TUTAR]": "MON", "[SANSÜR]": "MASK",
                "…": "ELLIPSIS"}
LABELS = ["PER", "ORG", "LOC", "COU", "DAT", "REF", "MON", "ELLIPSIS", "MASK"]
PLACEHOLDER_RE = re.compile("|".join(re.escape(p) for p in PLACEHOLDERS))
ERROR_PAGE_MARKERS = ("Ben Robot Değilim", "Ana sayfaya gitmek için tıklayınız")

# Sentence end: "." "!" "?" ";" + whitespace + uppercase/digit/quote/bracket,
# where the word before starts lowercase (rules out "Av.", "Ltd.", "9.").
SENT_END = re.compile(r"[.!?;](?=\s+[A-ZÇĞİÖŞÜ0-9\"“'(\[])")
WORD_BEFORE = re.compile(r"([\wçğıöşüÇĞİÖŞÜâîû'’]+)$")
LOWER_ABBREV = {"vb", "vs", "md", "bkz", "s", "sy", "no", "nr", "örn", "yy", "bşk"}


def tr_lower(s: str) -> str:
    """Turkish lowercasing (BERTurk-Legal is uncased): I -> ı, İ -> i."""
    return s.replace("I", "ı").replace("İ", "i").lower()


class Counter512:
    def __init__(self, vocab: Path):
        self.tok = BertWordPieceTokenizer(str(vocab), lowercase=False, strip_accents=False,
                                          clean_text=True, handle_chinese_chars=False)
        self.tok.add_special_tokens([MASK_TOKEN])

    def count(self, pieces: list[str]) -> list[int]:
        prepared = [tr_lower(p).replace(tr_lower(MASK_TOKEN), MASK_TOKEN) for p in pieces]
        return [len(e.ids) - 2 for e in self.tok.encode_batch(prepared)]   # minus [CLS]/[SEP]


def to_masked(text: str) -> tuple[str, list[str]]:
    """Placeholders -> [SANSUR]; returns (text, labels in order)."""
    labels = []

    def sub(m):
        labels.append(PLACEHOLDERS[m.group(0)])
        return MASK_TOKEN
    return PLACEHOLDER_RE.sub(sub, text), labels


def sentences(text: str) -> list[str]:
    cuts = {0, len(text)}
    cuts.update(m.end() for m in re.finditer(r"\n", text))
    for m in SENT_END.finditer(text):
        w = WORD_BEFORE.search(text, max(0, m.start() - 40), m.start())
        if w and w.group(1)[0].islower() and w.group(1).lower() not in LOWER_ABBREV:
            cuts.add(m.end())
    cuts = sorted(cuts)
    return [text[a:b] for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]


def fit(pieces: list[str], counter: Counter512, limit: int) -> list[tuple[str, int]]:
    """Split pieces over `limit` tokens at "; ", then ", ", then whitespace."""
    out = []
    for piece, n in zip(pieces, counter.count(pieces)):
        if n <= limit:
            out.append((piece, n))
            continue
        for sep in (r"(?<=;)\s+", r"(?<=,)\s+", r"\s+"):
            parts = [p for p in re.split(sep, piece) if p]
            if len(parts) > 1:
                break
        if len(parts) == 1:            # one unsplittable "word" (should not happen)
            out.append((piece, n))
            continue
        joiner = " "
        out.extend(fit_merge(parts, counter, limit, joiner))
    return out


def fit_merge(parts, counter, limit, joiner):
    """Merge parts back greedily up to `limit`; parts still too long recurse."""
    counts = counter.count(parts)
    out, cur, cur_n = [], [], 0
    for p, n in zip(parts, counts):
        if n > limit:
            if cur:
                out.append((joiner.join(cur), cur_n)); cur, cur_n = [], 0
            out.extend(fit([p], counter, limit))
            continue
        if cur and cur_n + n > limit:
            out.append((joiner.join(cur), cur_n)); cur, cur_n = [], 0
        cur.append(p); cur_n += n
    if cur:
        out.append((joiner.join(cur), cur_n))
    return out


def chunk(text: str, counter: Counter512, limit: int) -> list[tuple[str, int]]:
    """Consecutive whole-sentence chunks of at most `limit` tokens."""
    pieces = fit(sentences(text), counter, limit)
    chunks, cur, cur_n = [], "", 0
    for p, n in pieces:
        if cur and cur_n + n > limit:
            chunks.append((cur, cur_n)); cur, cur_n = "", 0
        cur += p; cur_n += n
    if cur.strip():
        chunks.append((cur, cur_n))
    # the sum of piece counts is an estimate at the joins; recount exactly
    exact = counter.count([c for c, _ in chunks])
    return [(c, n) for (c, _), n in zip(chunks, exact)]


def split_of(doc_id: str) -> str:
    h = int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % 100
    return "test" if h < 10 else "validation" if h < 20 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "datasets" / "danistay_masked_raw.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "datasets" / "danistay_mask_v1")
    ap.add_argument("--max-tokens", type=int, default=510)
    args = ap.parse_args()
    counter = Counter512(VOCAB)

    # 1) clean + dedup (keep the most recently labelled copy)
    stats = Counter()
    best = {}
    for line in open(args.src, encoding="utf-8"):
        r = json.loads(line)
        stats["messages"] += 1
        if any(m in r["text"] for m in ERROR_PAGE_MARKERS):
            stats["dropped_error_page"] += 1
            continue
        doc_id = r["metadata"]["doc_id"]
        if doc_id in best:
            stats["dropped_duplicate"] += 1
            if r.get("masked_at", "") <= best[doc_id].get("masked_at", ""):
                continue
        best[doc_id] = r

    # 2) convert, chunk, split
    records = {s: [] for s in ("train", "validation", "test")}
    for doc_id, r in best.items():
        text, labels = to_masked(r["text"])
        if not labels:
            stats["dropped_no_masks"] += 1
            continue
        split = split_of(doc_id)
        k, seen = 0, 0
        for ctext, n in chunk(text, counter, args.max_tokens):
            m = ctext.count(MASK_TOKEN)
            if not m:
                continue
            records[split].append({
                "id": f"{doc_id}-{k}", "doc_id": doc_id, "chunk": k,
                "text": ctext.strip(), "labels": labels[seen:seen + m], "n_tokens": n,
                "daire": r.get("daire"), "karar_tarihi": r.get("karar_tarihi"),
            })
            seen += m
            k += 1
        assert seen == len(labels), f"{doc_id}: {seen} masks chunked, {len(labels)} labels"

    # 3) exact-duplicate chunks: boilerplate (headers, standard rulings) repeats
    # across decisions. Keep one copy in train, and drop validation/test
    # chunks whose text is also in train, or the scores would be inflated.
    key = lambda r: hashlib.md5(" ".join(r["text"].split()).encode()).hexdigest()
    train_labels = {}
    for split in ("train", "validation", "test"):
        kept, local = [], set()
        for r in records[split]:
            h = key(r)
            if split != "train" and h in train_labels:
                stats[f"dropped_{split}_also_in_train"] += 1
                continue
            if h in local:
                stats[f"dropped_{split}_duplicate_chunk"] += 1
                if split == "train" and train_labels[h] != r["labels"]:
                    stats["duplicate_chunks_with_different_labels"] += 1
                continue
            local.add(h)
            if split == "train":
                train_labels[h] = r["labels"]
            kept.append(r)
        records[split] = kept

    args.out.mkdir(parents=True, exist_ok=True)
    labels_by_split = {s: Counter() for s in records}
    docs_by_split, chunks_by_split, tok_lens = Counter(), Counter(), []
    for split, rs in records.items():
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                labels_by_split[split].update(r["labels"])
                tok_lens.append(r["n_tokens"])
        chunks_by_split[split] = len(rs)
        docs_by_split[split] = len({r["doc_id"] for r in rs})

    (args.out / "labels.json").write_text(json.dumps(LABELS, ensure_ascii=False), encoding="utf-8")
    summary = {
        "source": str(args.src.name), "max_tokens": args.max_tokens, "mask_token": MASK_TOKEN,
        "cleaning": dict(stats), "decisions": dict(docs_by_split), "chunks": dict(chunks_by_split),
        "labels": {s: dict(c) for s, c in labels_by_split.items()},
        "chunk_tokens": {"max": max(tok_lens), "mean": round(sum(tok_lens) / len(tok_lens)),
                         "over_limit": sum(n > args.max_tokens for n in tok_lens)},
    }
    (args.out / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
