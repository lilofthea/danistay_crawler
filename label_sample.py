#!/usr/bin/env python3
"""
Label the masks ("...") of a random sample of decisions: rules first
(mask_types.py), then the LLM only for the masks the rules left as MASK.

Everything is written to a local folder outside data/docs, so the publisher
(which reads only data/index.jsonl + data/docs) never sends any of it.

Output (data/labeled_<n>/):
  original/<id>.txt   untouched copy of the decision
  labeled/<id>.txt    placeholders filled; LLM-filled ones are starred: [KİŞİ*]
  masks.jsonl         one line per mask: id, n, start, end, label, source, context
  summary.txt         counts per label and source

Usage:
  python3 label_sample.py              # 100 random decisions
  python3 label_sample.py -n 20 --seed 7
"""

import argparse
import json
import random
import re
import shutil
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mask_types import PLACEHOLDER, classify, masks

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_FILE = ROOT / "data" / "index.jsonl"
API = "http://10.150.96.44:8112/v1/chat/completions"
MODEL = "/model"   # vLLM serves Qwen/Qwen3.6-27B-FP8 under this id
ERROR_PAGE_MARKER = "Ana sayfaya gitmek için tıklayınız"

LLM_LABELS = {"PER", "ORG", "LOC", "COU", "DAT", "REF", "MON", "ELLIPSIS", "MASK"}

SYSTEM = """Sen Türk idare hukuku kararları üzerinde çalışan bir anonimleştirme uzmanısın.
Danıştay kararlarında kişisel bilgiler "..." ile sansürlenmiştir. Sana verilen metinde
sansürlü yerlerin bir kısmı türüne göre zaten etiketlendi ([KİŞİ], [KURUM], [YER],
[MAHKEME], [TARİH], [ATIF], [TUTAR]). Bunlar doğru kabul edilir; onlara dokunma.
Türü belirlenemeyen sansürler ⟦n⟧ şeklinde numaralandırıldı. Sadece bu numaralar
için, bağlama bakarak sansürlenen bilginin türünü belirle.

Etiketler:
- PER: kişi adı veya kişiyi tanımlayan bilgi (davacı, avukat, tanık, kod adı, kullanıcı adı)
- ORG: kurum, şirket, idare veya marka adı ya da adının bir parçası
- LOC: yer (il, ilçe, mahalle, köy, sokak, ada/parsel, adres)
- COU: mahkeme adı veya adının parçası; mahkeme adındaki şehir de COU'dur
  ("⟦1⟧ İdare Mahkemesi" -> COU, "⟦2⟧ Bölge İdare Mahkemesi ⟦3⟧ İdari Dava Dairesi" -> COU, COU)
- DAT: tarih
- REF: numara (dosya, karar, işlem, belge, plaka, telefon, IMEI, GSM, ID, şifre, ruhsat, takip no)
- MON: para tutarı
- ELLIPSIS: sansür değil, alıntıda atlanan metin (tırnak içinde veya "(...)" şeklinde)
- MASK: bağlamdan tür çıkarılamıyor

Kurallar:
- Metni değiştirme. Gerçek değeri tahmin etme; sadece türü ver.
- Her numara için tam olarak bir etiket ver, hiçbirini atlama.
- Cevabı sadece JSON olarak ver: {"1": "PER", "2": "REF", ...}"""


def pick(n: int, seed: int) -> list[str]:
    """Random decisions that have real full text."""
    ids = []
    with open(INDEX_FILE, encoding="utf-8") as f:
        for line in f:
            doc_id = str(json.loads(line)["id"])
            p = DOCS_DIR / f"{doc_id}.txt"
            if p.exists() and ERROR_PAGE_MARKER not in p.read_text(encoding="utf-8"):
                ids.append(doc_id)
    random.Random(seed).shuffle(ids)
    return ids[:n]


def ask_llm(prompt_text: str, numbers: list[int]) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"Etiketlenecek numaralar: {', '.join(map(str, numbers))}\n\n{prompt_text}"},
        ],
        "temperature": 0,
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    content = re.sub(r"(?s)<think>.*?</think>", "", content)
    m = re.search(r"(?s)\{.*\}", content)
    if not m:
        raise ValueError(f"no JSON in model answer: {content[:300]}")
    return json.loads(m.group(0))


def placeholder(label: str, from_llm: bool) -> str:
    rep = PLACEHOLDER[label]
    if from_llm and label != "MASK":
        rep = rep[:-1] + "*]" if rep.endswith("]") else rep + "*"
    return rep


def render(text: str, spans, reps) -> str:
    """Replace spans with reps, same spacing rules as mask_types.type_masks."""
    out, pos = [], 0
    for (s, e), rep in zip(spans, reps):
        out.append(text[pos:s])
        if not rep.startswith("…") and not rep.startswith("⟦"):
            if s and (text[s - 1].isalnum() or text[s - 1] == "."):
                rep = " " + rep
            if e < len(text) and text[e].isalnum():
                rep += " "
        out.append(rep)
        pos = e
    out.append(text[pos:])
    return "".join(out)


def label_doc(doc_id: str):
    text = (DOCS_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
    spans = list(masks(text))
    labels = [classify(text, s, e) for s, e in spans]
    source = ["regex"] * len(spans)

    unknown = [i for i, lab in enumerate(labels) if lab == "MASK"]
    error = None
    if unknown:
        # regex labels filled in as context, unknowns numbered 1..k
        num = {i: k for k, i in enumerate(unknown, 1)}
        reps = [f"⟦{num[i]}⟧" if i in num else PLACEHOLDER[labels[i]]
                for i in range(len(spans))]
        try:
            prompt = render(text, spans, reps)
            for attempt in range(3):   # the model occasionally emits broken JSON
                try:
                    answer = ask_llm(prompt, list(num.values()))
                    break
                except ValueError:
                    if attempt == 2:
                        raise
            for i, k in num.items():
                lab = str(answer.get(str(k), "")).strip().upper()
                if lab in LLM_LABELS:
                    labels[i] = lab
                source[i] = "llm"
        except Exception as e:   # keep the regex result, report the failure
            error = str(e)

    reps = [placeholder(lab, src == "llm") for lab, src in zip(labels, source)]
    records = [{"id": doc_id, "n": i + 1, "start": s, "end": e, "label": lab,
                "source": src,
                "context": " ".join((text[max(0, s - 60):s] + "⟦" + text[s:e] + "⟧"
                                     + text[e:e + 40]).split())}
               for i, ((s, e), lab, src) in enumerate(zip(spans, labels, source))]
    return doc_id, text, render(text, spans, reps), records, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM requests")
    args = ap.parse_args()

    out = ROOT / "data" / f"labeled_{args.n}"
    if out.exists():
        sys.exit(f"{out} already exists; remove it or pick another -n")
    (out / "original").mkdir(parents=True)
    (out / "labeled").mkdir()

    ids = pick(args.n, args.seed)
    for doc_id in ids:
        shutil.copy2(DOCS_DIR / f"{doc_id}.txt", out / "original" / f"{doc_id}.txt")

    counts, errors, all_records = Counter(), [], []
    with ThreadPoolExecutor(args.workers) as pool:
        for k, (doc_id, text, labeled, records, error) in enumerate(
                pool.map(label_doc, ids), 1):
            (out / "labeled" / f"{doc_id}.txt").write_text(labeled, encoding="utf-8")
            all_records += records
            counts.update((r["label"], r["source"]) for r in records)
            if error:
                errors.append(f"{doc_id}: {error}")
            print(f"[{k}/{len(ids)}] {doc_id}: {len(records)} sansür, "
                  f"{sum(r['source'] == 'llm' for r in records)} modele soruldu"
                  + (f"  HATA: {error}" if error else ""), flush=True)

    with open(out / "masks.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    daire = Counter()
    with open(INDEX_FILE, encoding="utf-8") as f:
        meta = {str(json.loads(l)["id"]): json.loads(l) for l in f}
    for doc_id in ids:
        daire[meta[doc_id].get("daireKurul")] += 1

    total = sum(counts.values())
    lines = [f"{len(ids)} karar, {total} sansür (seed={args.seed})", "",
             f"{'etiket':<10} {'regex':>7} {'llm':>7}"]
    for lab in ["PER", "ORG", "LOC", "COU", "DAT", "REF", "MON", "ELLIPSIS", "MASK"]:
        lines.append(f"{PLACEHOLDER[lab]:<10} {counts[(lab, 'regex')]:>7} {counts[(lab, 'llm')]:>7}")
    lines += ["", "daireler:"] + [f"  {d}: {c}" for d, c in daire.most_common()]
    if errors:
        lines += ["", "model hataları (bu kararlarda regex sonucu kaldı):"] + errors
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
