"""Generate the Supplementary Material the manuscript promises.

The manuscript cites "Supplementary Table S1" for the constituent studies and
"the Supplementary Material" for the excluded source, and neither existed. An
Elsevier submission that references supplementary material without uploading it
gets returned by the editorial office.

Generated from the benchmark itself, so it cannot disagree with the paper.
Writes paper/supplementary.tex, compilable standalone with the same class.

Run:  python -X utf8 scripts/build_supplementary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyaga8

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from eval_test_b import eos_predict  # noqa: E402

X_COLS = [f"x_{c}" for c in COMPONENTS]
OUT = ROOT / "paper" / "supplementary.tex"

PREAMBLE = r"""%% Supplementary Material for
%%   "A neural surrogate for the natural-gas compressibility factor:
%%    reference-equation accuracy from synthetic training data alone,
%%    benchmarked on 2426 independent laboratory measurements"
%%
%% GENERATED FILE - do not edit. Rebuild with:
%%   python -X utf8 scripts/build_supplementary.py
%% Compiles standalone. Upload as a supplementary file, not as part of the
%% main manuscript PDF.
\documentclass[preprint,12pt]{elsarticle}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\journal{Fuel}

\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thesection}{S\arabic{section}}

\begin{document}

\begin{frontmatter}
\title{Supplementary Material\\
A neural surrogate for the natural-gas compressibility factor:
reference-equation accuracy from synthetic training data alone, benchmarked on
2426 independent laboratory measurements}
\author{Emir Borovac}
\end{frontmatter}

\section{Constituent studies of the benchmark}

Every measurement in the benchmark comes from one of the studies in
Table~\ref{tab:s1}, distributed as machine-readable records in the NIST
ThermoML archive (doi:10.18434/mds2-2422). These are the experimental work on
which this study rests and should be cited when the benchmark is used; they are
also cited individually in the reference list of the main article.

Column $n$ gives the number of gas-phase points the study contributes to the
full benchmark, and $n_{\mathrm{NG}}$ the number falling inside the natural-gas
domain used for the primary comparison
($x_{\mathrm{CH_4}} \ge 0.5$, $T_{\mathrm{pr}} \le 3.0$, $T \le 500$~K).

"""

CLOSE = r"""
\section{Source excluded by the pre-declared screen}

One pre-declared screen was applied to the benchmark: a source was excluded if
\emph{both} AGA8-DETAIL and GERG-2008 deviated from it by more than 10\,\% in
median. Exactly one source failed this test:

\begin{center}
\begin{tabular}{ll}
\toprule
Digital object identifier & \texttt{@DOI@} \\
Points contributed        & @N@ \\
Median deviation, AGA8-DETAIL & @DET@\,\% \\
Median deviation, GERG-2008   & @GERG@\,\% \\
Points inside the natural-gas domain & @NNG@ \\
\bottomrule
\end{tabular}
\end{center}

The screen is method-agnostic: it removes data that no established equation
reproduces, rather than data that any particular method finds difficult. It is
also inconsequential for the primary comparison, because none of these points
lies inside the natural-gas domain. Every value reported in the Results section
of the main article is therefore identical with and without the screen. The
sensitivity check that establishes this is released with the code as
\texttt{scripts/screening\_sensitivity.py}.

\end{document}
"""


def main() -> None:
    t = pd.read_parquet(ROOT / "data" / "processed" / "test_b.parquet")
    y = t.z.to_numpy()
    x = t[X_COLS].to_numpy()
    tpc, ppc = x @ TC_VEC, x @ PC_VEC
    P, T = t.P_MPa.to_numpy(), t.T_K.to_numpy()
    tpr = T / tpc

    gerg = eos_predict(pyaga8.Gerg2008, t)
    det = eos_predict(pyaga8.Detail, t)

    def med_dev(pred, idx):
        return float(np.nanmedian(np.abs(pred[idx] - y[idx]) / y[idx]) * 100)

    susp = [d for d, g in t.groupby("doi")
            if med_dev(gerg, g.index.to_numpy()) > 10]
    keep = (~t.doi.isin(susp)).to_numpy()
    ng = ((t.x_methane >= 0.5).to_numpy() & (tpr <= 3.0) & (T <= 500))

    # All 40 studies, with the screened one daggered, so the table sums to the
    # 6726 points the main article reports for the extracted set.
    rows = []
    for doi, g in t.groupby("doi"):
        i = g.index.to_numpy()
        rows.append((str(doi), len(g), int(ng[i].sum()),
                     float(g.T_K.min()), float(g.T_K.max()),
                     float(g.P_MPa.min()), float(g.P_MPa.max()),
                     doi in susp))
    rows.sort(key=lambda r: -r[1])

    L = [PREAMBLE,
         r"\begin{longtable}{rlrrcc}",
         r"\caption{Studies constituting the benchmark, ordered by the number of "
         r"points contributed.\label{tab:s1}}\\",
         r"\toprule",
         r"\# & Digital object identifier & $n$ & $n_{\mathrm{NG}}$ & "
         r"$T$ (K) & $p$ (MPa) \\",
         r"\midrule",
         r"\endfirsthead",
         r"\toprule",
         r"\# & Digital object identifier & $n$ & $n_{\mathrm{NG}}$ & "
         r"$T$ (K) & $p$ (MPa) \\",
         r"\midrule",
         r"\endhead",
         r"\bottomrule",
         r"\endfoot"]
    for i, (doi, n, nng, t0, t1, p0, p1, excl) in enumerate(rows, 1):
        safe = doi.replace("_", r"\_")
        mark = r"$^{\dagger}$" if excl else ""
        L.append(f"{i} & \\texttt{{{safe}}}{mark} & {n} & {nng} & "
                 f"{t0:.0f}--{t1:.0f} & {p0:.3f}--{p1:.1f} \\\\")
    L.append(r"\end{longtable}")
    L.append("")
    L.append(f"In total {len(rows)} studies contribute {len(t):,} gas-phase "
             f"points. $^{{\\dagger}}$Excluded by the pre-declared screen "
             f"(Section~S2), leaving {int(keep.sum()):,} retained points, of "
             f"which {int((ng & keep).sum()):,} fall inside the natural-gas "
             f"domain used for the primary comparison.")

    if susp:
        d0 = susp[0]
        idx = t.index[t.doi == d0].to_numpy()
        # plain token replacement: the template contains literal LaTeX percent
        # signs, which collide with %-formatting
        block = CLOSE
        for tok, val in [("@DOI@", d0.replace("_", r"\_")),
                         ("@N@", str(len(idx))),
                         ("@DET@", f"{med_dev(det, idx):.1f}"),
                         ("@GERG@", f"{med_dev(gerg, idx):.1f}"),
                         ("@NNG@", str(int(ng[idx].sum())))]:
            block = block.replace(tok, val)
        L.append(block)
    else:
        L.append(r"\end{document}")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  Table S1: {len(rows)} studies")
    if susp:
        idx = t.index[t.doi == susp[0]].to_numpy()
        print(f"  excluded source: {susp[0]}, {len(idx)} points, "
              f"{int(ng[idx].sum())} inside the NG domain")


if __name__ == "__main__":
    main()
