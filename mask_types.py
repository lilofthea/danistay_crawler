#!/usr/bin/env python3
"""
Replace the anonymisation masks ("...", "…") in Danıştay decisions with typed
placeholders, decided from the surrounding words.

Label set follows TurkishLegalNER (koc-lab/turkishlegalner):
  PER kişi, ORG kurum, LOC yer, COU mahkeme, DAT tarih, REF dosya/karar atfı
plus MON (tutar, not in TurkishLegalNER) and MASK for masks no rule could type. Masks inside quotations are left as an
ellipsis ("…"), since there "..." usually means omitted text, not a name.

The crawled files are never modified; use this as a preprocessing step.

Usage:
  python3 mask_types.py                 # coverage report over data/docs
  python3 mask_types.py --samples MASK  # show contexts still untyped
  python3 mask_types.py --out data/docs_typed
"""

import argparse
import random
import re
from collections import Counter
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "data" / "docs"

PLACEHOLDER = {
    "PER": "[KİŞİ]", "ORG": "[KURUM]", "LOC": "[YER]", "COU": "[MAHKEME]",
    "DAT": "[TARİH]", "REF": "[ATIF]", "MON": "[TUTAR]", "MASK": "[SANSÜR]", "ELLIPSIS": "…",
}

MASK = re.compile(r"…+(?:\.+)?|\.{2,}")
_M = r"(?:…+|\.{2,})"                       # a mask, inside other patterns
_CHAIN = rf"(?:\s*(?:,|ve|ile|ila|ilâ|-)\s*{_M})*"   # "..., ... ve ..." list

# Conventions (kept in sync with the LLM prompt):
#  - a city/district inside an institution or court name belongs to that name:
#    "... Vergi Dairesi" -> ORG, "... İdare Mahkemesi" -> COU
#  - ada/parsel/pafta/blok/kat numbers are part of an address -> LOC

# Rules on the text right AFTER the mask, run on the full window (they may look
# past further masks, e.g. "..., ..., ... sayılı").
AFTER_FULL = [
    ("DAT", r"(?:tarih|gün)"),
    # "... için 5.000,00 TL maddi" - the claimant, before the amount rule
    ("PER", r"için\s+[\d.,]+\s*(?:TL|-TL)"),
    ("MON", r"(?:-\s*)?(?:TL|Türk\s+Lirası|lira|Avro|Euro|ABD\s+Doları)\b"),
    # parcel numbers, also in lists: "... ada, ... sayılı parsel"
    ("LOC", rf"{_CHAIN}\s*(?:sayılı\s+|numaralı\s+|nolu\s+)?"
            r"(?:parsel|ada\b|pafta|[Bb]lok\b|[Kk]at\b)"),
    # numbered lists / ranges: "..., ... ve ... sayılı", "... ila ... sayılı"
    ("REF", rf"{_CHAIN}\s*(?:sayılı|sayı\b|Esas|E\.|Karar\b|K\.|nolu|no'lu|"
            r"(?:\w+\s+){0,2}numaralı|(?:\w+\s+){0,2}numarası|plakalı|plaka\b|"
            r"ID\b|İMEİ|IMEI|GSM\b|takip\s+n)"),
    ("PER", r"(?:['’]?\w{0,4}\s+)?DÜŞÜNCESİ"),
]
# Rules on the text AFTER the mask, cut at the next mask, line end or ":;()",
# so a keyword of the next party ("Av. ... 2- ... Üniversitesi") can't leak in.
AFTER_CUT = [
    ("COU", r"(?:\d+\.\s*)?(?:Bölge\s+İdare|İdare\s+Mahkeme|Vergi\s+Mahkeme|"
            r"İdari\s+Dava|Vergi\s+Dava|İdare\s+Dava|Asliye|Ağır\s+Ceza|Sulh|"
            r"İş\s+Mahkeme|Kadastro|Ticaret\s+Mahkeme|Aile\s+Mahkeme|İcra|"
            r"Cumhuriyet\s+Başsavcılığı|Mahkeme)"),
    ("ORG", r"(?:isimli\s+(?:bir\s+)?(?:\w+\s+)?)?(?:program|yazılım|otomasyon)"),
    ("ORG", r"(?:[\wÇĞİÖŞÜçğıöşü.\-,&' ]{0,80}?)(?:Bakanlığı|Valiliği|Belediye|"
            r"Müdürlüğü|Başkanlığı|Kurumu|Kurumları|Kurum\b|Kurulu|Üniversitesi|"
            r"Vergi\s+Dairesi|Şirketi|A\.\s*Ş|Ltd|Kaymakamlığı|Rektörlüğü|"
            r"Noterliği|Komutanlığı|Bankası|Odası|Birliği|Kooperatifi|Hastanesi|"
            r"Okulu|Lisesi|Fakültesi|Derneği|Vakfı|Merkezi|Limited|Şti|"
            r"Sanayi\s+ve\s+Ticaret|Holding|Otel|Barosu|Sendikası|[Ff]irması|"
            r"Grubu|Partisi|Ajansı|Enstitüsü|Kulübü|Federasyonu|Gümrük|"
            r"Koleji|Kolej\b|Kursu|Kursunda|Corporat|GmbH|B\.\s*V|Inc\b|LLC)"),
    # company name: first word after the mask is a sector word
    ("ORG", r"(?:İnşaat|Turizm|Taahhüt|Nakliyat|Gıda|Tekstil|Akaryakıt|Petrol|"
            r"Elektrik|Elektronik|Otomotiv|Ambalaj|Reklam|Grup|Ticaret|Sanayi|"
            r"Yayıncılık|Enerji|Madencilik|Maden|Lojistik|Danışmanlık|Kimya|"
            r"Market|Mühendislik|Endüstri|Sistem|Makina|Makine|Mermer|Tarım|"
            r"Medikal|Yazılım|Bilişim|Gayrimenkul|Emlak|Yapı|Beton|Plastik|"
            r"Metal|Demir|Çelik|Mobilya|İlaç|Kağıt|Matbaa|Hırdavat|Oto\b|"
            r"Park\b|Kargo|Taşımacılık|Telekom|Organizasyon|Kuyumculuk)"),
    # the party that has a lawyer: "... vekili Av. ..."
    ("PER", r"vekil(?:i|leri)\b"),
    ("LOC", r"(?:İli\b|ili\b|İlçesi|ilçesi|Mahallesi|mahallesi|Mah\.|Köyü|köyü|"
            r"Sokak|sokak|Caddesi|caddesi|Bulvarı|Meydanı|Mevkii|mevkii|"
            r"Mezra|mezra|Beldesi|Garajı|OSB)"),
    # name with a glued case suffix and no apostrophe: "...nin", "...nın"
    ("PER", r"(?<![\s])n[ıiuü]n\b"),
]
# A name inside a quotation: "... adında bir polis memurunun". Checked before
# the quotation rule, which would otherwise call it omitted text.
NAMED = re.compile(r"^\s*(?:['’]\w{1,4}\s*)?(?:adında|adlı|isimli|isminde)\s+"
                   r"(?:bir\s+)?(?:şahıs|şahsın|kişi|polis|memur|kadın|erkek|"
                   r"bayan|bay|tanık|öğrenci|hasta|çocuk|asker|personel|doktor)")
# Rules on the text right BEFORE the mask.
BEFORE = [
    # door number in an address, before the generic "No:" reference
    ("LOC", r"(?:Sokak|Sokağı|Caddesi|Bulvarı|Cad\.|Sok\.|Sk\.)[,\s]*"
            r"(?:No|Kapı\s+No)\s*:?\s*$"),
    ("REF", r"\bNo\s*:\s*$"),
    ("REF", r"(?:\b[EK]\s*[:.]\s*|Esas\s*No\s*:?\s*|Karar\s*No\s*:?\s*|"
            r"\d{4}/\s*)$"),
    ("DAT", r"(?:tarihli|tarihinde|tarihi)\s*$"),
    # government lawyers: "Hukuk Müşaviri ...", "1. Huk. Müş. Yrd. V. ..."
    ("PER", r"(?:Müşaviri|Müş\.)(?:\s*Yrd\.?)?(?:\s*V\.)?\s*:?\s*$"),
    ("PER", r"(?:Müdür(?:ü)?|Yrd\.)\s+V\.\s*$"),
    ("PER", r"(?:\bAv\.?|Avukat|Hâkimi|Hakimi|Savcısı|Savcı|Üye|Başkan|"
            r"[Dd]avacı(?:lar)?|[Dd]avalı|[Ss]anık|[Mm]üşteki|[Mm]irasçıları|"
            r"[Mm]üteveffa|isimli|adlı|[Vv]ekil(?:i|leri)?|[Mm]üdafi(?:i)?|"
            r"dava\s+dışı|\d+\s*-)\s*\)?\s*:?\s*$"),
    # party headers without a colon line: "2- (DAVACI) ...", "III- DAVACI ...",
    # "DAVACI YANINDA MÜDAHİL : ..."
    ("PER", r"(?:\((?:DAVACI|DAVALI|İTİRAZ\s+EDEN)[^)\n]*\)|"
            r"\b(?:DAVACI|DAVALI)\w*(?:\s+YANINDA\s+MÜDAHİL)?)\s*:?\s*$"),
    # "... Bakanlığı / ...", "Belediye Başkanlığı - ...", "Şirketi - ...",
    # "(... Vergi Dairesi Müdürlüğü) - ..." - the city
    ("LOC", r"(?:\w+(?:lığı|liği|luğu|lüğü|lğü)|Belediyesi|Şirketi|Valiliği)"
            r"\s*\)?\s*[/-]\s*$"),
    # header line "DAVACI : ..." / "VEKİLİ : ..." etc.
    ("PER", r"(?m)^\s*(?:DAVACI|DAVALI|VEKİL\w*|TEMYİZ\s+EDEN|KARŞI\s+TARAF|"
            r"İSTEMDE\s+BULUNAN|MÜDAHİL|DAVAYA\s+KATILAN|Temyiz\s+İsteminde\s+Bulunan|Karşı\s+Taraf)"
            r"[^\n:]*:\s*(?:Av\.\s*)?$"),
]
# Person if a possessive/case suffix follows the mask: "...'ın", "...'nın"
PER_SUFFIX = re.compile(r"^['’][a-zçğıöşüâîûA-ZÇĞİÖŞÜÂÎÛ]{1,6}\b")

def masks(text: str):
    """Yield (start, end) of each mask. In "Av...." the first dot belongs to
    the abbreviation, so it is left out of the mask."""
    for m in MASK.finditer(text):
        start = m.start()
        if (text[start] == "." and m.end() - start >= 4 and start
                and text[start - 1].isalpha()):
            start += 1
        yield start, m.end()


AFTER_FULL_RE = [(t, re.compile(r"^\s*" + p)) for t, p in AFTER_FULL]
AFTER_CUT_RE = [(t, re.compile(r"^\s*" + p)) for t, p in AFTER_CUT]
BEFORE_RE = [(t, re.compile(p)) for t, p in BEFORE]
QUOTE = re.compile(r"(?:''|``|[\"“”])\s*$")
QUOTE_AFTER = re.compile(r"^\s*(?:''|[\"“”])")
CUT = re.compile(rf"{_M}|[\n:;()]")


def classify(text: str, start: int, end: int) -> str:
    after = text[end:end + 120]
    m = CUT.search(after)
    after_cut = after[:m.start()] if m else after
    line_start = text.rfind("\n", 0, start) + 1
    before = text[max(line_start, start - 80):start]
    if not before.strip() and line_start:
        # mask opens the line: the label is on the line above ("I- DAVACI\n...")
        prev_start = text.rfind("\n", 0, line_start - 1) + 1
        before = text[max(prev_start, line_start - 81):line_start - 1].rstrip() + " "
    if NAMED.match(after):
        return "PER"
    if (QUOTE.search(before) or QUOTE_AFTER.match(after)
            or (before.endswith("(") and after.startswith(")"))):
        return "ELLIPSIS"
    for t, rx in AFTER_FULL_RE:
        if rx.match(after):
            return t
    # "...'in", "...'nın": the mask is a whole name, so a keyword further on
    # ("...'in Manavgat İlçe Merkezi ile") is not part of it
    if PER_SUFFIX.match(after):
        return "PER"
    for t, rx in AFTER_CUT_RE:
        if rx.match(after_cut):
            return t
    for t, rx in BEFORE_RE:
        if rx.search(before):
            return t
    return "MASK"


def type_masks(text: str) -> tuple[str, Counter]:
    """Return (text with typed placeholders, label counts)."""
    out, counts, pos = [], Counter(), 0
    for start, end in masks(text):
        label = classify(text, start, end)
        counts[label] += 1
        out.append(text[pos:start])
        rep = PLACEHOLDER[label]
        # keep words apart when the mask was glued to them: "ilişkin...tarih"
        if label != "ELLIPSIS":
            if start and (text[start - 1].isalnum() or text[start - 1] == "."):
                rep = " " + rep
            if end < len(text) and text[end].isalnum():
                rep += " "
        out.append(rep)
        pos = end
    out.append(text[pos:])
    return "".join(out), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=DOCS_DIR)
    ap.add_argument("--out", type=Path, help="write typed copies here")
    ap.add_argument("--samples", help="print random contexts for this label")
    ap.add_argument("-n", type=int, default=25)
    args = ap.parse_args()

    files = [f for f in sorted(args.docs.glob("*.txt"))
             if "Ben Robot Değilim" not in f.read_text(encoding="utf-8")]
    total, samples = Counter(), []
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    for f in files:
        text = f.read_text(encoding="utf-8")
        typed, counts = type_masks(text)
        total += counts
        if args.out:
            (args.out / f.name).write_text(typed, encoding="utf-8")
        if args.samples:
            for start, end in masks(text):
                if classify(text, start, end) == args.samples:
                    ctx = text[max(0, start - 60):end + 60]
                    samples.append(f"{f.stem}: " + " ".join(ctx.split()))

    n = sum(total.values())
    print(f"{len(files)} karar, {n:,} sansür")
    for label, c in total.most_common():
        print(f"  {PLACEHOLDER[label]:<10} {label:<9} {c:>7,}  {c / n:6.1%}")
    if samples:
        random.seed(0)
        print()
        for s in random.sample(samples, min(args.n, len(samples))):
            print(" ", s)


if __name__ == "__main__":
    main()
