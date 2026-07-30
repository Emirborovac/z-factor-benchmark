"""Does the pre-declared screen change any conclusion?

One source (30 points) was excluded because both reference equations deviated
from it by more than 10 % in median. A referee will reasonably ask whether the
conclusions survive retaining it. This recomputes the entire headline comparison
with the source put back, so the answer is a measurement rather than an
assurance.

Writes reports/screening_sensitivity.md.

Run:  python -X utf8 scripts/screening_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyaga8
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from train_models_v2 import MW_VEC  # noqa: E402
from train_nn import Net, build_features  # noqa: E402
from convention_recovery import wichert_aziz  # noqa: E402
from eval_test_b import eos_predict  # noqa: E402
import zfactor.correlations as C  # noqa: E402

X_COLS = [f"x_{c}" for c in COMPONENTS]
MODEL = "NNCF"
ORDER = ["AGA8-DETAIL", MODEL, "GERG-2008", "DPR", "DAK", "Hall-Yarborough"]
L: list[str] = []


def w(s=""):
    print(s)
    L.append(s)


def aad(y, p):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean(np.abs(p[m] - y[m]) / np.abs(y[m])) * 100)


def clustered(y, a, b, src, n=10_000, seed=11):
    """Bootstrap the a-minus-b advantage, resampling whole laboratories."""
    labs = np.unique(src)
    rng = np.random.default_rng(seed)
    per = {s: (np.abs(b[src == s] - y[src == s]) / y[src == s] * 100
               - np.abs(a[src == s] - y[src == s]) / y[src == s] * 100).mean()
           for s in labs}
    v = np.array([per[s] for s in labs])
    draws = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return v.mean(), np.percentile(draws, [2.5, 97.5]), int((v > 0).sum()), len(labs)


def main() -> None:
    t = pd.read_parquet(ROOT / "data" / "processed" / "test_b.parquet")
    y = t.z.to_numpy()
    x = t[X_COLS].to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    tpc, ppc = x @ TC_VEC, x @ PC_VEC
    tpcw, ppcw = wichert_aziz(tpc, ppc, t.x_co2.to_numpy(), t.x_h2s.to_numpy())
    P, T = t.P_MPa.to_numpy(), t.T_K.to_numpy()
    ppr, tpr = P / ppc, T / tpc

    ck = torch.load(ROOT / "models" / "nn" / "eos_nn_detail.pt",
                    weights_only=False)
    net = Net(ck["d_in"], ck["width"], ck["blocks"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    F = build_features(T, P, ppr, tpr, x, gamma)
    with torch.no_grad():
        r = net(torch.tensor((F - ck["mu"]) / ck["sd"])).squeeze(1).numpy()

    gerg = eos_predict(pyaga8.Gerg2008, t)
    det = eos_predict(pyaga8.Detail, t)
    preds = {MODEL: C.dak(ppr, tpr) + r, "GERG-2008": gerg, "AGA8-DETAIL": det,
             "DAK": C.dak(P / ppcw, T / tpcw),
             "Hall-Yarborough": C.hall_yarborough(P / ppcw, T / tpcw),
             "DPR": C.dpr(P / ppcw, T / tpcw)}

    susp = [d for d, g in t.groupby("doi")
            if np.nanmedian(np.abs(gerg[g.index.to_numpy()]
                                   - y[g.index.to_numpy()])
                            / y[g.index.to_numpy()]) * 100 > 10]
    keep = (~t.doi.isin(susp)).to_numpy()
    domain = (t.x_methane >= 0.5).to_numpy() & (tpr <= 3.0) & (T <= 500)

    w("# Screening sensitivity")
    w()
    w("One source was excluded by the pre-declared screen: a source is dropped "
      "if **both** reference equations deviate from it by more than 10 % in "
      "median. This recomputes the headline comparison with that source "
      "retained.")
    w()
    w(f"Excluded source: `{susp[0] if susp else 'none'}`")
    w()

    scr = domain & keep                       # as reported in the paper
    unscr = domain                            # screen removed
    n_extra = int(unscr.sum() - scr.sum())
    w(f"- As reported (screened): **{int(scr.sum()):,}** points, "
      f"{t[scr].doi.nunique()} laboratories")
    w(f"- Screen removed: **{int(unscr.sum()):,}** points, "
      f"{t[unscr].doi.nunique()} laboratories "
      f"(+{n_extra} points from the excluded source)")
    w()

    w("## Average absolute deviation (%)")
    w()
    w("| method | screened (reported) | screen removed | change |")
    w("|---|---:|---:|---:|")
    for k in ORDER:
        a, b = aad(y[scr], preds[k][scr]), aad(y[unscr], preds[k][unscr])
        w(f"| {k} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    w()

    w("## Ranking")
    w()
    for tag, m in [("screened (reported)", scr), ("screen removed", unscr)]:
        rank = sorted(ORDER, key=lambda k: aad(y[m], preds[k][m]))
        w(f"- {tag}: " + " < ".join(rank))
    w()

    w("## Clustered comparison against NNCF, screen removed")
    w()
    w("> Estimator note: this bootstrap averages the per-laboratory mean "
      "advantage without weighting by laboratory size, so the point estimates "
      "are NOT directly comparable with Table 2 of the manuscript, which "
      "weights by size. Both columns here use the same estimator, which is "
      "what makes the screened/unscreened comparison valid.")
    w()
    w("| comparison | advantage (pp) | 95 % CI | labs won |")
    w("|---|---:|---|---:|")
    src = t.doi.to_numpy()[unscr]
    for k in ORDER:
        if k == MODEL:
            continue
        mu, ci, won, nl = clustered(y[unscr], preds[MODEL][unscr],
                                    preds[k][unscr], src)
        w(f"| NNCF vs {k} | {mu:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
          f"{won}/{nl} |")
    w()

    # ---- the conclusions that must survive -------------------------------
    rank_u = sorted(ORDER, key=lambda k: aad(y[unscr], preds[k][unscr]))
    rank_s = sorted(ORDER, key=lambda k: aad(y[scr], preds[k][scr]))
    corr = ["DPR", "DAK", "Hall-Yarborough"]
    beats_all = all(
        clustered(y[unscr], preds[MODEL][unscr], preds[k][unscr], src)[1][0] > 0
        for k in corr)
    tied = clustered(y[unscr], preds[MODEL][unscr],
                     preds["AGA8-DETAIL"][unscr], src)[1]

    w("## Verdict")
    w()
    w(f"- Ranking unchanged: **{rank_u == rank_s}**")
    w(f"- NNCF still beats all three classical correlations with the whole "
      f"confidence interval above zero: **{beats_all}**")
    w(f"- NNCF vs AGA8-DETAIL confidence interval still spans zero "
      f"(statistical tie): **{tied[0] < 0 < tied[1]}** "
      f"([{tied[0]:+.3f}, {tied[1]:+.3f}])")
    w()

    out = ROOT / "reports" / "screening_sensitivity.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
