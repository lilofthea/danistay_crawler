#!/usr/bin/env python3
"""
Average length of downloaded Danıştay decisions (data/docs/*.txt).

The header (court, parties, lawyers) is skipped: the body is taken from the
"İSTEMİN KONUSU" heading onwards. Spelling of that heading varies a lot
(İSTEMİN_KONUSU, İstemin Özeti, İSTEMLERİN KONUSU, DAVANIN KONUSU ...).
Decisions without any such heading fall back to the text after "Karar No".

Usage:
  python3 avg_length.py            # data/docs
  python3 avg_length.py <docs_dir>
"""

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

DOCS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "docs"

BODY_START = re.compile(
    r"(?:İSTEM(?:İN|LERİN|I|İ)?|DAVANIN)[\s_]*(?:KONUSU|ÖZETİ)", re.IGNORECASE)
# Captcha error page saved instead of a decision - not a real document.
CAPTCHA_PAGE = "Ben Robot Değilim"
KARAR_NO = re.compile(r"Karar\s*No\s*:?\s*\S+[^\n]*\n", re.IGNORECASE)


def body(text: str) -> tuple[str, str]:
    """Return (body, how_it_was_found)."""
    m = BODY_START.search(text)
    if m:
        return text[m.start():], "heading"
    m = KARAR_NO.search(text)
    if m:
        return text[m.end():], "karar_no"
    return text, "full"


def main():
    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        sys.exit(f"no .txt files in {DOCS_DIR}")

    chars, words, how = [], [], Counter()
    captcha = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        if CAPTCHA_PAGE in text:
            captcha.append(f.stem)
            continue
        b, src = body(text)
        b = b.strip()
        chars.append(len(b))
        words.append(len(b.split()))
        how[src] += 1

    def show(name, xs):
        print(f"{name:<10} ortalama={statistics.mean(xs):>9,.0f}  "
              f"medyan={statistics.median(xs):>9,.0f}  "
              f"min={min(xs):>7,}  max={max(xs):>9,}")

    print(f"{len(chars)} karar  (başlangıç: {dict(how)})")
    if captcha:
        print(f"{len(captcha)} dosya captcha hata sayfası, hesaba katılmadı")
    show("karakter", chars)
    show("kelime", words)


if __name__ == "__main__":
    main()
