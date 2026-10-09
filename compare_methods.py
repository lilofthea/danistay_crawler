#!/usr/bin/env python3
"""
Label the same decisions three ways and compare, mask by mask:
  regex   mask_types.py only (unknowns stay MASK)
  llm     the LLM labels every mask, without seeing the regex labels
  hybrid  regex first, LLM for the rest (read from label_sample.py's masks.jsonl)

Input/output folder: data/labeled_<n>/ made by label_sample.py.
Writes there:
  regex_only/<id>.txt, llm_only/<id>.txt
  compare.jsonl     per mask: regex, llm, hybrid labels + context
  compare.txt       agreement, confusion matrix, all regex/llm disagreements

Usage:
  python3 compare_methods.py            # data/labeled_100
  python3 compare_methods.py --dir data/labeled_100 --workers 4
"""

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mask_types import PLACEHOLDER, classify, masks
from label_sample import API, MODEL, LLM_LABELS, render

LABELS = ["PER", "ORG", "LOC", "COU", "DAT", "REF", "MON", "ELLIPSIS", "MASK"]

# Same label definitions as label_sample.SYSTEM, but nothing is pre-filled:
# every mask is numbered and the model labels all of them.
SYSTEM = """Sen Türk idare hukuku kararları üzerinde çalışan bir anonimleştirme uzmanısın.
Danıştay kararlarında kişisel bilgiler "..." ile sansürlenmiştir. Sana verilen metinde
her sansürlü yer ⟦n⟧ şeklinde numaralandırıldı. Her numara için, bağlama bakarak
sansürlenen bilginin türünü belirle.

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
- Cevabı sadece JSON olarak ver: {"1": "PER", "2": "DAT", ...}"""


def ask_all(numbered: str, n: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Metinde {n} sansür var (⟦1⟧ - ⟦{n}⟧).\n\n{numbered}"},
        ],
        "temperature": 0,
        "max_tokens": 8000,
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
        raise ValueError(f"no JSON in model answer: {content[:200]}")
    return json.loads(m.group(0))


def label_doc(path: Path):
    text = path.read_text(encoding="utf-8")
    spans = list(masks(text))
    regex = [classify(text, s, e) for s, e in spans]
    llm, error = ["MASK"] * len(spans), None
    if spans:
        numbered = render(text, spans, [f"⟦{i}⟧" for i in range(1, len(spans) + 1)])
        for attempt in range(3):   # the model occasionally emits broken JSON
            try:
                answer = ask_all(numbered, len(spans))
                missing = 0
                for i in range(len(spans)):
                    lab = str(answer.get(str(i + 1), "")).strip().upper()
                    if lab in LLM_LABELS:
                        llm[i] = lab
                    else:
                        missing += 1
                if missing:
                    error = f"{missing} numara cevapsız/geçersiz"
                break
            except Exception as e:
                error = str(e)
    return path.stem, text, spans, regex, llm, error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent / "data" / "labeled_100")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    files = sorted((args.dir / "original").glob("*.txt"))
    if not files:
        sys.exit(f"no decisions in {args.dir / 'original'}")
    hybrid = {}
    with open(args.dir / "masks.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hybrid[(r["id"], r["start"])] = r["label"]
    for sub in ("regex_only", "llm_only"):
        (args.dir / sub).mkdir(exist_ok=True)

    rows, errors = [], []
    with ThreadPoolExecutor(args.workers) as pool:
        for k, (doc_id, text, spans, regex, llm, error) in enumerate(
                pool.map(label_doc, files), 1):
            (args.dir / "regex_only" / f"{doc_id}.txt").write_text(
                render(text, spans, [PLACEHOLDER[l] for l in regex]), encoding="utf-8")
            (args.dir / "llm_only" / f"{doc_id}.txt").write_text(
                render(text, spans, [PLACEHOLDER[l] for l in llm]), encoding="utf-8")
            for (s, e), a, b in zip(spans, regex, llm):
                rows.append({"id": doc_id, "start": s, "end": e, "regex": a, "llm": b,
                             "hybrid": hybrid.get((doc_id, s), "?"),
                             "context": " ".join((text[max(0, s - 60):s] + "⟦" + text[s:e]
                                                  + "⟧" + text[e:e + 40]).split())})
            if error:
                errors.append(f"{doc_id}: {error}")
            print(f"[{k}/{len(files)}] {doc_id}: {len(spans)} sansür"
                  + (f"  HATA: {error}" if error else ""), flush=True)

    with open(args.dir / "compare.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    agree = lambda a, b: sum(r[a] == r[b] for r in rows)
    known = [r for r in rows if r["regex"] != "MASK"]
    lines = [f"{len(files)} karar, {n} sansür", "",
             "Uyum (aynı etiketi veren sansür oranı):",
             f"  regex  - llm    : {agree('regex', 'llm'):>5} / {n}  ({agree('regex', 'llm') / n:.1%})",
             f"  regex  - hybrid : {agree('regex', 'hybrid'):>5} / {n}  ({agree('regex', 'hybrid') / n:.1%})",
             f"  llm    - hybrid : {agree('llm', 'hybrid'):>5} / {n}  ({agree('llm', 'hybrid') / n:.1%})",
             f"  regex'in etiketlediklerinde (MASK hariç) regex - llm: "
             f"{sum(r['regex'] == r['llm'] for r in known)} / {len(known)} "
             f"({sum(r['regex'] == r['llm'] for r in known) / len(known):.1%})",
             "", "Etiket dağılımı:", f"  {'':<10}{'regex':>8}{'llm':>8}{'hybrid':>8}"]
    for lab in LABELS:
        lines.append(f"  {PLACEHOLDER[lab]:<10}" + "".join(
            f"{sum(r[m] == lab for r in rows):>8}" for m in ("regex", "llm", "hybrid")))

    conf = Counter((r["regex"], r["llm"]) for r in rows)
    lines += ["", "Karışıklık matrisi (satır: regex, sütun: llm):",
              "  " + " " * 9 + "".join(f"{l[:4]:>6}" for l in LABELS)]
    for a in LABELS:
        lines.append(f"  {a:<9}" + "".join(f"{conf[(a, b)] or '.':>6}" for b in LABELS))

    by_pair = defaultdict(list)
    for r in rows:
        if r["regex"] != r["llm"]:
            by_pair[(r["regex"], r["llm"])].append(r)
    lines += ["", "Regex ile llm'in ayrıştığı yerler (en sık çiftten başlayarak):"]
    for (a, b), rs in sorted(by_pair.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"\n## regex {a} / llm {b}: {len(rs)}")
        for r in rs:
            lines.append(f"  {r['id']}  [hybrid {r['hybrid']}]  {r['context']}")
    if errors:
        lines += ["", "Model hataları:"] + errors
    (args.dir / "compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:40]))


if __name__ == "__main__":
    main()
