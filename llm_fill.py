#!/usr/bin/env python3
"""
Label the anonymisation masks ("...", "…") of a decision with an LLM, for
comparison with the rule-based labels of mask_types.py.

The masks are found with the same code as mask_types.py and numbered
(⟦1⟧, ⟦2⟧, ...); the model only returns a label per number and never rewrites
the text, so both versions differ only in the labels.

Usage:
  python3 llm_fill.py 1176959900 [1096393800 ...]

Writes, per decision, into data/docs_llm_sample/:
  <id>.llm.txt      text with the LLM's placeholders
  <id>.regex.txt    text with mask_types.py placeholders
  <id>.compare.txt  every mask: context, regex label, LLM label
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from mask_types import PLACEHOLDER, classify, masks

DOCS_DIR = Path(__file__).parent / "data" / "docs"
OUT_DIR = Path(__file__).parent / "data" / "docs_llm_sample"
API = "http://10.150.96.44:8112/v1/chat/completions"
MODEL = "/model"   # vLLM serves Qwen/Qwen3.6-27B-FP8 under this id

LABELS = ["PER", "ORG", "LOC", "COU", "DAT", "REF", "MON", "ELLIPSIS", "MASK"]

SYSTEM = """Sen Türk idare hukuku kararları üzerinde çalışan bir anonimleştirme uzmanısın.
Danıştay kararlarında kişisel bilgiler "..." ile sansürlenmiştir. Sana verilen metinde
her sansürlü yer ⟦n⟧ şeklinde numaralandırıldı. Her numara için, bağlama bakarak
sansürlenen bilginin türünü belirle.

Etiketler:
- PER: kişi adı (davacı, avukat, tetkik hâkimi, mirasçı, tanık vb.)
- ORG: kurum, şirket, idare adı veya adının bir parçası (... Bakanlığı, ... A.Ş., ... Belediyesi)
- LOC: yer (il, ilçe, mahalle, köy, sokak, ada/parsel, kapı numarası)
- COU: mahkeme adı veya adının parçası (... İdare Mahkemesi, ... Bölge İdare Mahkemesi ... İdari Dava Dairesi)
- DAT: tarih
- REF: dosya, karar, işlem veya belge numarası (E:..., K:..., ... sayılı, ruhsat no, takip no)
- MON: para tutarı
- ELLIPSIS: sansür değil, alıntıda atlanan metin (örn. mevzuat alıntısında "(...)")
- MASK: bağlamdan tür çıkarılamıyor

Kurallar:
- Metni değiştirme veya yeniden yazma. Sadece etiket ver.
- Gerçek değeri tahmin etmeye çalışma; sadece türü ver.
- Her numara için tam olarak bir etiket ver, hiçbirini atlama.
- Cevabı sadece JSON olarak ver: {"1": "PER", "2": "DAT", ...}"""


def number_masks(text: str):
    """Return (text with ⟦n⟧ markers, [(start, end)] of masks in the original)."""
    spans = list(masks(text))
    out, pos = [], 0
    for i, (s, e) in enumerate(spans, 1):
        out.append(text[pos:s])
        out.append(f"⟦{i}⟧")
        pos = e
    out.append(text[pos:])
    return "".join(out), spans


def ask_llm(numbered: str, n: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Metinde {n} sansür var (⟦1⟧ - ⟦{n}⟧).\n\n{numbered}"},
        ],
        "temperature": 0,
        "max_tokens": 8000,
        "response_format": {"type": "json_object"},
        # Qwen3: answer directly, no <think> block
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    content = resp["choices"][0]["message"]["content"]
    content = re.sub(r"(?s)<think>.*?</think>", "", content).strip()
    m = re.search(r"(?s)\{.*\}", content)
    if not m:
        raise ValueError(f"no JSON in model answer: {content[:300]}")
    return json.loads(m.group(0)), resp.get("usage", {})


def render(text: str, spans, labels) -> str:
    """Same placeholder/spacing logic as mask_types.type_masks."""
    out, pos = [], 0
    for (s, e), label in zip(spans, labels):
        out.append(text[pos:s])
        rep = PLACEHOLDER[label]
        if label != "ELLIPSIS":
            if s and (text[s - 1].isalnum() or text[s - 1] == "."):
                rep = " " + rep
            if e < len(text) and text[e].isalnum():
                rep += " "
        out.append(rep)
        pos = e
    out.append(text[pos:])
    return "".join(out)


def process(doc_id: str):
    text = (DOCS_DIR / f"{doc_id}.txt").read_text(encoding="utf-8")
    numbered, spans = number_masks(text)
    if not spans:
        print(f"{doc_id}: sansür yok")
        return
    regex = [classify(text, s, e) for s, e in spans]

    t0 = time.time()
    answer, usage = ask_llm(numbered, len(spans))
    llm = []
    for i in range(1, len(spans) + 1):
        lab = str(answer.get(str(i), "")).strip().upper()
        llm.append(lab if lab in LABELS else "MASK")
    missing = [i for i in range(1, len(spans) + 1) if str(i) not in answer]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{doc_id}.llm.txt").write_text(render(text, spans, llm), encoding="utf-8")
    (OUT_DIR / f"{doc_id}.regex.txt").write_text(render(text, spans, regex), encoding="utf-8")

    same = sum(a == b for a, b in zip(regex, llm))
    lines = [f"{doc_id}: {len(spans)} sansür | aynı: {same} | farklı: {len(spans) - same}"
             f" | model süresi {time.time() - t0:.0f}s | token {usage}",
             f"modelin atladığı numaralar: {missing or 'yok'}", "",
             f"{'#':>3}  {'REGEX':<9} {'LLM':<9}  bağlam"]
    for i, ((s, e), a, b) in enumerate(zip(spans, regex, llm), 1):
        ctx = " ".join((text[max(0, s - 70):s] + "⟦" + text[s:e] + "⟧" + text[e:e + 50]).split())
        mark = "  " if a == b else "≠ "
        lines.append(f"{i:>3}{mark}{a:<9} {b:<9}  {ctx}")
    (OUT_DIR / f"{doc_id}.compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(lines[0])
    print("  regex:", dict(Counter(regex)))
    print("  llm:  ", dict(Counter(llm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="decision ids (data/docs/<id>.txt)")
    args = ap.parse_args()
    for doc_id in args.ids:
        try:
            process(doc_id)
        except Exception as e:
            print(f"{doc_id}: HATA {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
