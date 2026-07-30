"""Write paper/highlights.txt: the Highlights as a standalone file.

Elsevier requires Highlights to be uploaded as a separate item in Editorial
Manager, not only embedded in the manuscript. Extracted from manuscript.tex so
the two cannot disagree, and the character limit is re-checked here.

Run:  python -X utf8 scripts/build_highlights.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "manuscript.tex"
OUT = ROOT / "paper" / "highlights.txt"

CAP = 85
BS = chr(92)


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    block = tex.split(BS + "begin{highlights}", 1)[1] \
               .split(BS + "end{highlights}", 1)[0]
    items = [re.sub(r"\s+", " ", i).strip()
             for i in re.findall(re.escape(BS) + r"item\s+(.+)", block)]

    if not 3 <= len(items) <= 5:
        raise SystemExit(f"Elsevier allows 3-5 highlights; found {len(items)}")

    bad = [i for i in items if len(i) > CAP]
    acro = [a for i in items
            for a in re.findall(r"\b(?:[A-Z]{2,}|[A-Z]+-?\d{2,})\b", i)]

    OUT.write_text("\n".join(items) + "\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}  ({len(items)} highlights)\n")
    for i in items:
        print(f"  [{len(i):2d}/{CAP}] {i}")
    print()
    if bad:
        raise SystemExit(f"over the {CAP}-character limit: {bad}")
    if acro:
        raise SystemExit(f"Elsevier forbids acronyms in highlights: {acro}")
    print(f"all within {CAP} characters, no acronyms")


if __name__ == "__main__":
    main()
