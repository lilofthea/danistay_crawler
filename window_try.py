#!/usr/bin/env python3
"""
Experiment: does the LLM label the masks as well when it only sees up to 512
tokens around them (what a BERT-size model would see) as with the full text?

The decision is labelled twice with the same prompt and no regex hints:
  full    the whole text, every mask numbered (compare_methods.SYSTEM)
  chunks  only the parts with masks, as chunks of whole sentences (header
          lines count as sentences; a sentence over 512 tokens is split at
          "; " then ", ") of at most 512 tokens (model tokenizer):
          start at the sentence of the first unlabelled mask, add whole
          sentences left and right while it fits, label the unlabelled masks
          in it, repeat. No chunk starts or ends mid-sentence.
Output: data/window_try/<id>.txt - agreement, every mask where they differ
(with the pipeline's final label from data/pipeline_50 as a third opinion),
and the chunks themselves.

Usage:  python3 window_try.py 1036821400 [--size 512]
        python3 window_try.py 1036821400 --sentences 2   # sentence +-2 sentences
"""

import argparse
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

from compare_methods import SYSTEM
from labeling_pipeline import LABELS, chat, render
from mask_types import masks

ROOT = Path(__file__).parent
SERVER = "http://10.150.96.44:8112"

# Sentence end: "." "!" "?" ";" followed by whitespace and an uppercase
# letter, digit, quote or bracket, where the word before it starts lowercase
# ("istenilmektedir."). That rules out masks ("..."), ordinals ("9.") and the
# capitalised abbreviations in names ("Av.", "Elk.", "San.", "Ltd.", "Şti.").
# Newlines always split, so each header line is its own piece.
SENT_END = re.compile(r"[.!?;](?=\s+[A-ZÇĞİÖŞÜ0-9\"“'(])")
WORD_BEFORE = re.compile(r"([\wçğıöşüÇĞİÖŞÜâîû'’]+)$")
LOWER_ABBREV = {"vb", "vs", "md", "bkz", "s", "sy", "no", "nr", "örn", "yy", "bşk"}


def call(path: str, payload: dict) -> dict:
    req = urllib.request.Request(SERVER + path,
                                 data=json.dumps({"model": "/model", **payload}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def ntokens(s: str) -> int:
    return call("/tokenize", {"prompt": s, "add_special_tokens": False})["count"]


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence / line, covering the whole text."""
    cuts = {0, len(text)}
    for m in re.finditer(r"\n", text):
        cuts.add(m.end())
    for m in SENT_END.finditer(text):
        w = WORD_BEFORE.search(text, max(0, m.start() - 40), m.start())
        if w and w.group(1)[0].islower() and w.group(1).lower() not in LOWER_ABBREV:
            cuts.add(m.end())
    cuts = sorted(cuts)
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]


def fit(text: str, pieces, size: int, spans) -> list[tuple[int, int, int]]:
    """Split pieces longer than `size` tokens at clause boundaries ("; ",
    then ", "), never inside a mask, and merge the clauses back greedily up to
    `size`. Returns (start, end, tokens) for every piece."""
    out = []
    for a, b in pieces:
        t = ntokens(text[a:b])
        if t <= size:
            out.append((a, b, t))
            continue
        clauses = [(a, b)]
        for sep in (r";\s+", r",\s+"):
            cuts = [a + m.end() for m in re.finditer(sep, text[a:b])
                    if not any(s < a + m.end() < e for s, e in spans)]
            if cuts:
                bounds = [a] + cuts + [b]
                clauses = [(x, y) for x, y in zip(bounds, bounds[1:]) if text[x:y].strip()]
                break
        cur_a, cur_t = clauses[0][0], 0
        for x, y in clauses:
            tt = ntokens(text[x:y])
            if cur_t and cur_t + tt > size:
                out.append((cur_a, x, cur_t))
                cur_a, cur_t = x, 0
            cur_t += tt
        out.append((cur_a, clauses[-1][1], cur_t))
    return out


def ask(numbered: str, numbers: list[int]) -> dict:
    user = (f"Etiketlenecek numaralar: {', '.join(map(str, numbers))}\n"
            f"Sadece bu numaraları etiketle; cevaptaki anahtarlar tam olarak bu "
            f"numaralar olsun.\n\n{numbered}")
    try:
        ans = chat(SYSTEM, user, expect=numbers)
    except ValueError:            # unusable after retries: count as unanswered
        ans = {}
    if len(numbers) == 1 and str(numbers[0]) not in ans and len(ans) == 1:
        # one mask asked, one label given under another key ({"1": ...}):
        # the model renumbered it; the label is still for the only mask asked
        ans = {str(numbers[0]): next(iter(ans.values()))}
    out = {}
    for n in numbers:
        lab = str(ans.get(str(n), "")).strip().upper()
        out[n] = lab if lab in LABELS else "MASK"
    return out


def full_labels(text: str, spans) -> dict:
    """LLM labels with the whole text, every mask numbered."""
    n = len(spans)
    return ask(render(text, spans, [f"⟦{i}⟧" for i in range(1, n + 1)]), list(range(1, n + 1)))


def chunk_labels(text: str, spans, size: int):
    """LLM labels from chunks of whole sentences of at most `size` tokens,
    grown from the first unlabelled mask. Returns (labels, chunks) where each
    chunk is (tokens, labelled mask numbers, numbered chunk text)."""
    n = len(spans)
    pieces = fit(text, sentences(text), size, spans)
    sents = [(a, b) for a, b, _ in pieces]
    toks = [t for _, _, t in pieces]
    sent_of = {i: next(k for k, (a, b) in enumerate(sents) if a <= s < b)
               for i, (s, e) in enumerate(spans, 1)}
    labels, chunks = {}, []
    for i in range(1, n + 1):
        if i in labels:
            continue
        lo = hi = sent_of[i]
        while sents[hi][1] < spans[i - 1][1]:   # mask straddling a cut
            hi += 1
        total = sum(toks[lo:hi + 1])
        grew = True
        while grew:                             # add whole sentences, alternating sides
            grew = False
            if hi + 1 < len(sents) and total + toks[hi + 1] <= size:
                hi += 1; total += toks[hi]; grew = True
            if lo > 0 and total + toks[lo - 1] <= size:
                lo -= 1; total += toks[lo]; grew = True
        a, b = sents[lo][0], sents[hi][1]
        inside = [k for k, (s, e) in enumerate(spans, 1) if a <= s and e <= b]
        todo = [k for k in inside if k not in labels]
        local = [(s - a, e - a) for k, (s, e) in enumerate(spans, 1) if k in inside]
        chunk = render(text[a:b], local, [f"⟦{k}⟧" for k in inside])
        labels.update(ask(chunk, todo))
        chunks.append((total, todo, chunk))
    return labels, chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--sentences", type=int, default=None,
                    help="instead of filling up to --size: the mask's sentence +-N sentences")
    args = ap.parse_args()
    text = (ROOT / "data" / "docs" / f"{args.id}.txt").read_text(encoding="utf-8")
    spans = list(masks(text))
    n = len(spans)

    # 1) full text
    full = ask(render(text, spans, [f"⟦{i}⟧" for i in range(1, n + 1)]), list(range(1, n + 1)))

    # 2) sentence chunks around the masks
    pieces = fit(text, sentences(text), args.size, spans)
    sents = [(a, b) for a, b, _ in pieces]
    toks = [t for _, _, t in pieces]
    sent_of = {}
    for i, (s, e) in enumerate(spans, 1):
        sent_of[i] = next(k for k, (a, b) in enumerate(sents) if a <= s < b)

    chunked, chunks = {}, []
    if args.sentences is not None:
        # fixed context: the mask's sentence plus N sentences on each side;
        # only the masks of the middle sentence are labelled in each call
        N = args.sentences
        for k in sorted(set(sent_of.values())):
            lo, hi = max(0, k - N), min(len(sents) - 1, k + N)
            a, b = sents[lo][0], sents[hi][1]
            todo = [i for i in range(1, n + 1) if sent_of[i] == k]
            inside = [i for i, (s, e) in enumerate(spans, 1) if a <= s and e <= b]
            local = [(s - a, e - a) for i, (s, e) in enumerate(spans, 1) if i in inside]
            chunk = render(text[a:b], local, [f"⟦{i}⟧" for i in inside])
            chunked.update(ask(chunk, todo))
            chunks.append((sum(toks[lo:hi + 1]), todo, chunk))
    for i in range(1, n + 1):
        if i in chunked:
            continue
        lo = hi = sent_of[i]
        # a mask can straddle a sentence cut only if the splitter is wrong;
        # make sure its whole span is inside
        while sents[hi][1] < spans[i - 1][1]:
            hi += 1
        total = sum(toks[lo:hi + 1])
        grew = True
        while grew:                       # add whole sentences, alternating sides
            grew = False
            if hi + 1 < len(sents) and total + toks[hi + 1] <= args.size:
                hi += 1; total += toks[hi]; grew = True
            if lo > 0 and total + toks[lo - 1] <= args.size:
                lo -= 1; total += toks[lo]; grew = True
        a, b = sents[lo][0], sents[hi][1]
        inside = [k for k, (s, e) in enumerate(spans, 1) if a <= s and e <= b]
        todo = [k for k in inside if k not in chunked]
        local = [(s - a, e - a) for k, (s, e) in enumerate(spans, 1) if k in inside]
        chunk = render(text[a:b], local, [f"⟦{k}⟧" for k in inside])
        chunked.update(ask(chunk, todo))
        chunks.append((total, todo, chunk))

    pipe = {}
    pf = ROOT / "data" / "pipeline_50" / "masks.jsonl"
    if pf.exists():
        pipe = {r["n"]: r["final"] for r in map(json.loads, open(pf, encoding="utf-8"))
                if r["id"] == args.id}

    doc_tokens = sum(toks)
    sent_tokens = sum(t for c in chunks for t in [c[0]])
    same = sum(full[i] == chunked[i] for i in full)
    lines = [
        f"{args.id}: {doc_tokens} token, {len(sents)} cümle/satır, {n} sansür",
        (f"{len(chunks)} parça (sansürlü cümle ± {args.sentences} cümle), " if args.sentences is not None
         else f"{len(chunks)} parça (en çok {args.size} token, tam cümle), ")
        + f"toplam {sent_tokens} token gönderildi; "
        f"parça boyları: {[c[0] for c in chunks]}",
        f"tam metin ile parça aynı etiket: {same}/{n} ({same / n:.0%})",
        f"pipeline sonucuyla aynı: tam metin {sum(full[i] == pipe.get(i) for i in full)}/{n}, "
        f"parça {sum(chunked[i] == pipe.get(i) for i in full)}/{n}",
        f"tam metin: {dict(Counter(full.values()))}",
        f"parça    : {dict(Counter(chunked.values()))}", "",
        f"{'#':>3}  {'TAM':<9}{'PARÇA':<9}{'PIPELINE':<9} bağlam"]
    for i, (s, e) in enumerate(spans, 1):
        if full[i] != chunked[i]:
            ctx = " ".join((text[max(0, s - 70):s] + "⟦" + text[s:e] + "⟧" + text[e:e + 45]).split())
            lines.append(f"{i:>3}  {full[i]:<9}{chunked[i]:<9}{pipe.get(i, '-'):<9} {ctx}")
    lines += ["", "=" * 70, "PARÇALAR"]
    for k, (t, todo, chunk) in enumerate(chunks, 1):
        lines += ["", f"--- parça {k}: {t} token, etiketlenen #{todo}", chunk.strip()]
    out = ROOT / "data" / "window_try"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.id}.s{args.sentences}.txt" if args.sentences is not None else f"{args.id}.txt"
    (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8 + sum(full[i] != chunked[i] for i in full)]))


if __name__ == "__main__":
    main()
