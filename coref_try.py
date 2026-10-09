#!/usr/bin/env python3
"""
Experiment: number the persons and institutions in a labelled decision, so
masks that refer to the same one share a number: [KİŞİ-1] ... [KİŞİ-1].

The names are masked, so "same person" can only come from context (roles
such as davacı / tetkik hâkimi, "anılan şirket", repeated pairs). The LLM is
asked to group only the masks it is sure about; the rest stay [KİŞİ]/[KURUM].
Answers that break the rules (a number in two groups, a group mixing labels,
unknown numbers) are dropped and reported.

Input: labels from the pipeline output (data/pipeline_50/masks.jsonl).
Usage:  python3 coref_try.py 1036821400
Output: data/coref_try/<id>.txt (numbered text) and <id>.groups.txt (report)
"""

import json
import sys
from pathlib import Path

from labeling_pipeline import chat, render
from mask_types import PLACEHOLDER, masks

ROOT = Path(__file__).parent
LABELS_FILE = ROOT / "data" / "pipeline_50" / "masks.jsonl"
OUT = ROOT / "data" / "coref_try"

SYSTEM = """Sen Türk idare hukuku kararları üzerinde çalışan bir uzmansın. Kararda kişi ve
kurum adları "..." ile sansürlenmiş ve türleri etiketlenmiş. Kişi ve kurum sansürleri
⟦n:PER⟧ ve ⟦n:ORG⟧ şeklinde numaralı. Görevin: AYNI kişiyi veya AYNI kurumu gösteren
sansürleri gruplamak.

Kullanabileceğin ipuçları: rol (davacı, davalı, vekil, tetkik hâkimi, tanık, şirket
temsilcisi), "anılan/söz konusu şirket" gibi göndermeler, aynı sırayla tekrar eden isim
listeleri, ek ve bağlam. Gerçek isimleri tahmin etme.

Kurallar:
- Sadece EMİN olduğun bağları kur. Emin değilsen o sansürü hiçbir gruba koyma.
- Tek başına ama başka hiçbir sansürle aynı olmadığından emin olduğun bir kişi/kurum
  tek elemanlı grup olabilir.
- Bir numara en fazla bir grupta olur. PER ve ORG ayrı gruplanır.
- Her gruba kısa bir rol yaz (ör. "davacı", "davacı şirketin temsilcisi", "tanık 1").

Cevabı sadece JSON olarak ver:
{"PER": [{"masks": [1, 7], "rol": "davacı"}, ...], "ORG": [{"masks": [3, 9], "rol": "..."}]}"""


def main():
    doc_id = sys.argv[1]
    rows = sorted((json.loads(l) for l in open(LABELS_FILE, encoding="utf-8")
                   if f'"{doc_id}"' in l), key=lambda r: r["n"])
    if not rows:
        sys.exit(f"{doc_id} not in {LABELS_FILE}")
    text = (ROOT / "data" / "docs" / f"{doc_id}.txt").read_text(encoding="utf-8")
    spans = list(masks(text))
    labels = [r["final"] for r in rows]
    assert len(spans) == len(labels), "mask positions changed since labelling"

    prompt = render(text, spans, [f"⟦{i}:{l}⟧" if l in ("PER", "ORG") else PLACEHOLDER[l]
                                  for i, l in enumerate(labels, 1)])
    ans = chat(SYSTEM, prompt)

    seen, groups, dropped = set(), {"PER": [], "ORG": []}, []
    for kind in ("PER", "ORG"):
        for g in ans.get(kind, []) or []:
            ms = [int(m) for m in g.get("masks", []) if str(m).isdigit()]
            bad = [m for m in ms if not (1 <= m <= len(labels)) or labels[m - 1] != kind
                   or m in seen]
            if bad or not ms:
                dropped.append(f"{kind} {ms} ({g.get('rol')}): geçersiz numara {bad}")
                continue
            seen.update(ms)
            groups[kind].append((ms, g.get("rol", "")))

    number = {}
    for kind in ("PER", "ORG"):
        for k, (ms, _) in enumerate(groups[kind], 1):
            for m in ms:
                number[m] = k
    reps = []
    for i, l in enumerate(labels, 1):
        p = PLACEHOLDER[l]
        reps.append(p[:-1] + f"-{number[i]}]" if i in number else p)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{doc_id}.txt").write_text(render(text, spans, reps), encoding="utf-8")

    ctx = lambda m: " ".join((text[max(0, spans[m - 1][0] - 60):spans[m - 1][0]] + "⟦"
                              + text[spans[m - 1][0]:spans[m - 1][1]] + "⟧"
                              + text[spans[m - 1][1]:spans[m - 1][1] + 40]).split())
    lines = []
    for kind in ("PER", "ORG"):
        total = sum(l == kind for l in labels)
        grouped = sum(len(ms) for ms, _ in groups[kind])
        lines.append(f"===== {kind}: {total} sansür, {grouped} gruplandı, "
                     f"{len(groups[kind])} farklı varlık, {total - grouped} gruplanmadı")
        for k, (ms, rol) in enumerate(groups[kind], 1):
            lines.append(f"\n{PLACEHOLDER[kind][:-1]}-{k}] ({rol}) -> #{ms}")
            lines += [f"    #{m}: {ctx(m)}" for m in ms]
        rest = [i for i, l in enumerate(labels, 1) if l == kind and i not in seen]
        if rest:
            lines.append(f"\ngruplanmayan {kind}:")
            lines += [f"    #{m}: {ctx(m)}" for m in rest]
        lines.append("")
    if dropped:
        lines += ["kurallara uymadığı için atılan gruplar:"] + [f"  {d}" for d in dropped]
    (OUT / f"{doc_id}.groups.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
