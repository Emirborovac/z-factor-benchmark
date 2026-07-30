r"""Repair the 40-key citation list into a single unbroken \cite command.

An earlier attempt wrapped the key list across source lines with '%'
continuations. That does not work: the '%' and the newline are absorbed into the
adjacent key, producing keys like "ben2017%\njarne2011" that BibTeX cannot
resolve. A citation key list must be one unbroken line; LaTeX does not care how
long a source line is.

Run:  python -X utf8 scripts/repair_citelist.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "manuscript.tex"
SRC = ROOT / "paper" / "refs_sources.bib"

BS = chr(92)


def main() -> None:
    s = TEX.read_text(encoding="utf-8")
    src = re.findall(r"@\w+\{([^,]+),", SRC.read_text(encoding="utf-8"))
    if len(src) != 40:
        raise SystemExit(f"expected 40 source keys, found {len(src)}")

    pat = re.escape(BS) + r"cite\{[^{}]*\}"
    broken = [m for m in re.finditer(pat, s, re.S) if "%" in m.group(0)]
    if not broken:
        print("no broken citation list found (already repaired?)")
        return
    if len(broken) > 1:
        raise SystemExit(f"{len(broken)} broken lists; expected 1")

    m = broken[0]
    s = s[:m.start()] + BS + "cite{" + ",".join(src) + "}" + s[m.end():]
    TEX.write_text(s, encoding="utf-8")
    print(f"rewrote the citation list as one line with {len(src)} keys")


if __name__ == "__main__":
    main()
