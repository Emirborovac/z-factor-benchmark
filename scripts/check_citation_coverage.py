"""Will every bibliography entry actually reach the compiled reference list?

BibTeX prints only entries that are cited. check_manuscript.py verified the
converse -- that every \\cite key resolves -- and therefore missed the case that
matters here: 40 experimental source studies sitting in refs.bib, never cited,
and silently dropped from the reference list, while the manuscript states three
times that they "are cited individually in the reference list".

Run:  python -X utf8 scripts/check_citation_coverage.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

BS = chr(92)


def keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", text))


def main() -> None:
    tex = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    bib = keys((PAPER / "refs.bib").read_text(encoding="utf-8"))
    core = keys((PAPER / "refs_core.bib").read_text(encoding="utf-8"))
    src = keys((PAPER / "refs_sources.bib").read_text(encoding="utf-8"))

    cited: set[str] = set()
    for m in re.findall(re.escape(BS) + r"cite[tp]?\*?\{([^}]*)\}", tex):
        cited |= {k.strip() for k in m.split(",") if k.strip()}
    nocited: set[str] = set()
    for m in re.findall(re.escape(BS) + r"nocite\{([^}]*)\}", tex):
        nocited |= {k.strip() for k in m.split(",") if k.strip()}

    printed = cited | nocited
    if "*" in nocited:
        printed = bib

    dropped = bib - printed
    print("Citation coverage\n" + "=" * 62)
    print(f"  bibliography entries      {len(bib)}")
    print(f"  cited with \\cite          {len(cited)}")
    print(f"  added with \\nocite        {len(nocited)}")
    print(f"  will be PRINTED           {len(printed & bib)}")
    print(f"  will be DROPPED           {len(dropped)}")
    print(f"    of which source studies {len(dropped & src)} of {len(src)}")
    print(f"    of which core refs      {len(dropped & core)} of {len(core)}")

    claims = [
        ("manuscript, Section 2.1",
         "are cited individually in the reference list"),
        ("manuscript, Data availability",
         "cited individually in the reference list"),
    ]
    print()
    asserts_all_cited = any(c[1] in tex for c in claims)
    if asserts_all_cited:
        print("  The manuscript CLAIMS the source studies are individually")
        print("  cited. That claim is "
              f"{'TRUE' if not (dropped & src) else 'FALSE'} as written.")

    missing = sorted(cited - bib)
    if missing:
        print(f"\n  unresolved \\cite keys: {missing}")

    bad = bool(dropped & src) or bool(missing)
    print()
    if bad:
        print(f"FAIL: {len(dropped & src)} source studies would not appear in "
              f"the reference list.")
        print("      Fix with \\nocite{...} for those keys, or reword the "
              "claim.")
    else:
        print("OK: every bibliography entry reaches the reference list.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
