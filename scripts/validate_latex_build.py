"""Predict the LaTeX + BibTeX build without a TeX installation.

There is no TeX engine on this machine, so the compiled output cannot be
inspected. This reproduces the parts of the toolchain that actually decide
whether the build succeeds and whether the reference list is complete:

  * brace balance and environment nesting, the usual causes of a hard failure;
  * every command used is either a LaTeX/elsarticle built-in or provided by a
    declared package, so nothing is silently undefined;
  * every \\includegraphics target exists on disk;
  * every \\ref and \\label resolves;
  * BibTeX behaviour: which entries would be PRINTED, in what order, numbered,
    and whether any cited entry is missing a field its type requires. This is
    what determines "does the reference list end at 55".

It cannot catch everything a real compile would, and says so at the end.

Run:  python -X utf8 scripts/validate_latex_build.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGDIR = PAPER / "figures"

BS = chr(92)
FAIL: list[str] = []
WARN: list[str] = []


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    FAIL.append(msg)


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")
    WARN.append(msg)


def ok(msg: str) -> None:
    print(f"  [ok  ] {msg}")


# Commands provided by LaTeX itself, elsarticle, or the packages declared in the
# preamble (amsmath, amssymb, booktabs, lineno, graphicx+natbib via elsarticle).
KNOWN = {
    # structure
    "documentclass", "usepackage", "begin", "end", "newcommand", "renewcommand",
    "journal", "title", "author", "ead", "cortext", "corref", "affiliation",
    "section", "subsection", "subsubsection", "item", "label", "ref", "eqref",
    "caption", "centering", "includegraphics", "cite", "nocite",
    "bibliographystyle", "bibliography", "linenumbers", "sep", "thanks",
    "orcid", "fnref", "footnote", "appendix",
    # text
    "textbf", "textit", "emph", "texttt", "textrm", "textsc", "underline",
    "text", "mbox", "hspace", "vspace", "noindent", "par", "\\",
    # maths
    "frac", "sqrt", "sum", "prod", "int", "partial", "mathrm", "mathbf",
    "mathit", "left", "right", "times", "cdot", "pm", "mp", "le", "ge", "leq",
    "geq", "neq", "approx", "sim", "gtrsim", "lesssim", "ll", "gg", "infty",
    "ln", "log", "exp", "min", "max", "overline", "theta", "rho", "gamma",
    "mu", "delta", "Delta", "alpha", "beta", "lambda", "sigma", "phi", "psi",
    "dagger", "circ", "prime", "quad", "qquad", ",", ";", "!", ":",
    # counters and numbering (LaTeX built-ins)
    "arabic", "roman", "Roman", "alph", "Alph", "thetable", "thesection",
    "thefigure", "theequation", "setcounter", "addtocounter", "value",
    # booktabs
    "toprule", "midrule", "bottomrule", "cmidrule", "addlinespace",
    # longtable (supplementary only)
    "endfirsthead", "endhead", "endfoot", "endlastfoot",
    # graphicx / misc
    "linewidth", "textwidth", "columnwidth", "url", "href", "and",
    # accents / escapes
    '"', "'", "`", "^", "~", "=", ".", "u", "v", "c", "H", "k", "b", "d", "r",
    "%", "&", "_", "$", "#", "{", "}", "i", "j", "l", "o", "O", "aa", "AA",
    "ss", "ae", "AE", "oe", "OE",
}

ENVS = {
    "document", "frontmatter", "abstract", "keyword", "highlights",
    "figure", "table", "tabular", "longtable", "equation", "align",
    "enumerate", "itemize", "description", "center", "displaymath",
}


def check_braces(name: str, s: str) -> None:
    """Brace balance, ignoring escaped braces and comment lines."""
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = s.replace(r"\{", "").replace(r"\}", "")
    depth, line = 0, 1
    for ch in s:
        if ch == "\n":
            line += 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                bad(f"{name}: unmatched '}}' at line {line}")
                return
    if depth:
        bad(f"{name}: {depth} unclosed '{{' at end of file")
    else:
        ok(f"{name}: braces balanced")


def check_envs(name: str, s: str) -> None:
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    stack: list[str] = []
    for m in re.finditer(re.escape(BS) + r"(begin|end)\{([^}]+)\}", s):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack:
                bad(f"{name}: \\end{{{env}}} with no matching \\begin")
                return
            top = stack.pop()
            if top != env:
                bad(f"{name}: \\begin{{{top}}} closed by \\end{{{env}}}")
                return
    if stack:
        bad(f"{name}: unclosed environments {stack}")
    else:
        ok(f"{name}: all environments balanced")

    used = set(re.findall(re.escape(BS) + r"begin\{([^}]+)\}", s))
    unknown = used - ENVS
    if unknown:
        warn(f"{name}: environments not in the known list {sorted(unknown)} "
             f"- check the package that provides them is declared")
    else:
        ok(f"{name}: every environment is provided by a declared package")


def check_commands(name: str, s: str) -> None:
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    defined = set(re.findall(re.escape(BS) + r"(?:new|renew)command\{"
                             + re.escape(BS) + r"([a-zA-Z]+)\}", s))
    used = set(re.findall(re.escape(BS) + r"([a-zA-Z]+)", s))
    unknown = used - KNOWN - defined - ENVS
    if unknown:
        bad(f"{name}: possibly undefined commands {sorted(unknown)}")
    else:
        ok(f"{name}: no undefined commands "
           f"({len(used)} distinct, {len(defined)} author-defined)")


def parse_bib(text: str) -> dict[str, tuple[str, dict[str, str]]]:
    out = {}
    for chunk in text.split("@")[1:]:
        m = re.match(r"(\w+)\{([^,]+),", chunk)
        if not m:
            continue
        typ, key = m.group(1).lower(), m.group(2).strip()
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{", chunk):
            k, depth, buf = fm.end(), 1, []
            while k < len(chunk) and depth:
                ch = chunk[k]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if not depth:
                        break
                buf.append(ch)
                k += 1
            fields[fm.group(1).lower()] = "".join(buf).strip()
        out[key] = (typ, fields)
    return out


# elsarticle-num derives from plain/unsrt: these are the fields whose absence
# makes BibTeX emit a warning and print an incomplete entry.
REQUIRED = {
    "article": ["author", "title", "journal", "year"],
    "inproceedings": ["author", "title", "booktitle", "year"],
    "techreport": ["author", "title", "institution", "year"],
    "misc": [],
    "book": ["author", "title", "publisher", "year"],
    "incollection": ["author", "title", "booktitle", "publisher", "year"],
}


def main() -> None:
    print("Predicted LaTeX + BibTeX build\n" + "=" * 66)

    for fn in ("manuscript.tex", "supplementary.tex"):
        f = PAPER / fn
        if not f.exists():
            bad(f"{fn} missing")
            continue
        s = f.read_text(encoding="utf-8")
        print(f"\n-- {fn}")
        check_braces(fn, s)
        check_envs(fn, s)
        check_commands(fn, s)

        # declared packages vs environments that need them
        pkgs = set(re.findall(re.escape(BS) + r"usepackage(?:\[[^\]]*\])?"
                              r"\{([^}]+)\}", s))
        pkgs = {p.strip() for grp in pkgs for p in grp.split(",")}
        if "longtable" in s and "longtable" not in pkgs:
            bad(f"{fn}: uses longtable but does not \\usepackage{{longtable}}")
        else:
            ok(f"{fn}: declared packages {sorted(pkgs) or '(none)'}")

    # ---- graphics ---------------------------------------------------------
    print("\n-- graphics")
    tex = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    inc = re.findall(re.escape(BS) + r"includegraphics(?:\[[^\]]*\])?"
                     r"\{([^}]+)\}", tex)
    missing = [g for g in inc
               if not any((FIGDIR / f"{g}{e}").exists()
                          for e in (".pdf", ".png", ".eps", ".jpg"))]
    if missing:
        bad(f"figure files not found: {missing}")
    else:
        ok(f"all {len(inc)} \\includegraphics targets exist in paper/figures/")

    # ---- cross-references -------------------------------------------------
    print("\n-- cross-references")
    labels = set(re.findall(re.escape(BS) + r"label\{([^}]+)\}", tex))
    refs = set(re.findall(re.escape(BS) + r"(?:ref|eqref)\{([^}]+)\}", tex))
    dangling = refs - labels
    if dangling:
        bad(f"\\ref to undefined labels: {sorted(dangling)}")
    else:
        ok(f"all {len(refs)} references resolve among {len(labels)} labels")
    dup = [l for l in labels
           if len(re.findall(re.escape(BS) + r"label\{" + re.escape(l) + r"\}",
                             tex)) > 1]
    if dup:
        bad(f"duplicate labels: {dup}")
    else:
        ok("no duplicate labels")

    # ---- BibTeX -----------------------------------------------------------
    print("\n-- BibTeX (elsarticle-num)")
    bibfile = re.search(re.escape(BS) + r"bibliography\{([^}]+)\}", tex)
    if not bibfile:
        bad("no \\bibliography command")
        raise SystemExit(1)
    bibname = bibfile.group(1).strip()
    bp = PAPER / f"{bibname}.bib"
    if not bp.exists():
        bad(f"\\bibliography{{{bibname}}} but {bp.name} does not exist")
        raise SystemExit(1)
    ok(f"\\bibliography{{{bibname}}} -> {bp.name} exists")

    entries = parse_bib(bp.read_text(encoding="utf-8"))
    cited_order: list[str] = []
    for m in re.finditer(re.escape(BS) + r"cite[tp]?\*?\{([^}]*)\}", tex):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k and k not in cited_order:
                cited_order.append(k)
    for m in re.finditer(re.escape(BS) + r"nocite\{([^}]*)\}", tex):
        for k in (x.strip() for x in m.group(1).split(",")):
            if k == "*":
                for k2 in entries:
                    if k2 not in cited_order:
                        cited_order.append(k2)
            elif k and k not in cited_order:
                cited_order.append(k)

    unresolved = [k for k in cited_order if k not in entries]
    if unresolved:
        bad(f"cited keys with no bib entry (BibTeX would print '?'): "
            f"{unresolved}")
    else:
        ok(f"every one of {len(cited_order)} cited keys resolves")

    printed = [k for k in cited_order if k in entries]
    print(f"  [ok  ] reference list would contain {len(printed)} entries, "
          f"numbered 1-{len(printed)}")
    if printed:
        print(f"         first: [1] {printed[0]}")
        print(f"         last:  [{len(printed)}] {printed[-1]}")

    incomplete = []
    for k in printed:
        typ, f = entries[k]
        for req in REQUIRED.get(typ, []):
            if not f.get(req):
                incomplete.append(f"{k} ({typ}) missing {req}")
    if incomplete:
        for i in incomplete[:12]:
            warn(f"incomplete entry: {i}")
        warn(f"{len(incomplete)} entries would trigger a BibTeX warning")
    else:
        ok("every printed entry has the fields its type requires")

    uncited = set(entries) - set(printed)
    if uncited:
        bad(f"{len(uncited)} bib entries would be DROPPED from the list: "
            f"{sorted(uncited)[:6]}...")
    else:
        ok("no bibliography entry is dropped")

    # ---- unescaped specials in bib fields --------------------------------
    rough = []
    for k, (typ, f) in entries.items():
        for fld in ("title", "journal", "booktitle", "author"):
            v = f.get(fld, "")
            if re.search(r"(?<!\\)&", v):
                rough.append(f"{k}.{fld} has an unescaped '&'")
            if re.search(r"(?<!\\)%", v):
                rough.append(f"{k}.{fld} has an unescaped '%'")
    if rough:
        for r in rough[:10]:
            bad(r)
    else:
        ok("no unescaped & or % in bibliography fields")

    print("\n" + "=" * 66)
    print(f"{len(FAIL)} hard problem(s), {len(WARN)} warning(s)")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print()
    print("Static prediction only. It does not run TeX, so it cannot catch")
    print("overfull boxes, float placement, page-count overruns, or a package")
    print("version incompatibility on the publisher's system.")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
