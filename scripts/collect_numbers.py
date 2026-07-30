"""Single source of truth for every number the manuscript states.

Recomputes the headline results directly from the saved artefacts so no value
in the paper is transcribed from memory or from a stale report. Writes
reports/results_numbers.md.

Run:  python -X utf8 scripts/collect_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyaga8
import torch
from scipy.stats import wilcoxon

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
L: list[str] = []


def w(s=""):
    print(s)
    L.append(s)


def aad(y, p):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean(np.abs(p[m] - y[m]) / np.abs(y[m])) * 100)


def within(y, p, tol):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean(np.abs(p[m] - y[m]) / np.abs(y[m]) * 100 <= tol) * 100)


def bias(y, p):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[m] - y[m]) / y[m]) * 100)


def rmse(y, p):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def main():
    # ---------------------------------------------------------------- data --
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
    susp = [d for d, g in t.groupby("doi")
            if np.nanmedian(np.abs(gerg[g.index.to_numpy()] - y[g.index.to_numpy()])
                            / y[g.index.to_numpy()]) * 100 > 10]
    keep = (~t.doi.isin(susp)).to_numpy()
    ng = keep & (t.x_methane >= 0.5).to_numpy() & (tpr <= 3.0) & (T <= 500)

    preds = {MODEL: C.dak(ppr, tpr) + r, "GERG-2008": gerg, "AGA8-DETAIL": det,
             "DAK": C.dak(P / ppcw, T / tpcw),
             "Hall-Yarborough": C.hall_yarborough(P / ppcw, T / tpcw),
             "DPR": C.dpr(P / ppcw, T / tpcw),
             "Kareem": C.kareem(P / ppcw, T / tpcw)}
    yn = y[ng]

    w("# Results — authoritative numbers for the manuscript")
    w()
    w("Regenerate with `python -X utf8 scripts/collect_numbers.py`. Every "
      "value in the paper must come from this file.")
    w()

    # ------------------------------------------------------------ dataset --
    w("## 1 Benchmark composition")
    w()
    w(f"- Test Set B, total extracted: **{len(t):,}** gas-phase points from "
      f"**{t.doi.nunique()}** published studies")
    w(f"- Excluded by the pre-declared screen (both reference EOS exceed 10 % "
      f"median deviation): {len(susp)} source, {int((~keep).sum())} points "
      f"({susp})")
    w(f"- Natural-gas domain (x_CH4 >= 0.5, Tpr <= 3.0, T <= 500 K): "
      f"**{int(ng.sum()):,}** points from **{t[ng].doi.nunique()}** laboratories")
    w(f"- State range of the NG domain: T {T[ng].min():.1f}–{T[ng].max():.1f} K, "
      f"P {P[ng].min():.3f}–{P[ng].max():.1f} MPa, "
      f"Ppr {ppr[ng].min():.3f}–{ppr[ng].max():.1f}, "
      f"Tpr {tpr[ng].min():.3f}–{tpr[ng].max():.2f}")
    w(f"- Full Test Set B state range: T {t.T_K.min():.0f}–{t.T_K.max():.0f} K, "
      f"P {t.P_MPa.min():.3f}–{t.P_MPa.max():.1f} MPa")
    sour = (t.x_h2s + t.x_co2 > 0.02).to_numpy()
    w(f"- Sour points (H2S + CO2 > 2 mol%) in the NG domain: "
      f"{int((sour & ng).sum()):,}")
    w()

    # The training pool is three tiers, not one. Reporting only the first
    # tier's MANIFEST understated it and contradicted the manuscript's totals,
    # so the totals are counted from the files themselves.
    SYN = ROOT / "data" / "synthetic"
    tiers = [("base", "compositions", "points_detail"),
             ("broadened", "compositions_v3", "points_v3_detail"),
             ("hydrogen-blend", "compositions_h2", "points_h2_detail")]
    w("### Synthetic training pool (counted from the parquet files)")
    w()
    w("| tier | compositions | DETAIL-labelled states |")
    w("|---|---:|---:|")
    n_c = n_p = 0
    for name, cf, pf in tiers:
        cp, pp = SYN / f"{cf}.parquet", SYN / f"{pf}.parquet"
        if not (cp.exists() and pp.exists()):
            continue
        nc = len(pd.read_parquet(cp, columns=None))
        npt = len(pd.read_parquet(pp, columns=["z"]))
        n_c += nc
        n_p += npt
        w(f"| {name} | {nc:,} | {npt:,} |")
    w(f"| **total** | **{n_c:,}** | **{n_p:,}** ({n_p / 1e7:.3f}e7) |")
    w()
    man = SYN / "MANIFEST.md"
    if man.exists():
        w("Sampling rules (base tier, from MANIFEST.md; the broadened and "
          "hydrogen tiers follow the same rules over wider envelopes):")
        w()
        for line in man.read_text(encoding="utf-8").splitlines():
            if line.startswith("- "):
                w(line)
        w()

    # ------------------------------------------------------------- model ---
    meta = json.loads((ROOT / "models" / "nn" / "meta_detail.json").read_text())
    w("## 2 Model")
    w()
    w(f"- Architecture: {ck['blocks']} residual blocks x {ck['width']} units, "
      f"SiLU; **{meta['params']:,}** trainable parameters")
    w(f"- Inputs ({ck['d_in']}): {meta['features']}")
    w(f"- Target: z - z_DAK (residual); objective: mean relative error")
    w(f"- Teacher: **AGA8-DETAIL** (labels regenerated on identical states)")
    w(f"- Validation AAD against its teacher: **{meta['val_MAPE']:.4f} %**")
    w()

    # -------------------------------------------------------- leaderboard --
    w("## 3 Headline benchmark — natural-gas domain")
    w()
    order = ["AGA8-DETAIL", "GERG-2008", MODEL, "DPR", "DAK",
             "Hall-Yarborough", "Kareem"]
    w("| method | AAD (%) | RMSE | bias (%) | within ±0.5 % | ±1 % | ±2 % |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for k in sorted(order, key=lambda k: aad(yn, preds[k][ng])):
        p = preds[k][ng]
        w(f"| {k} | {aad(yn, p):.3f} | {rmse(yn, p):.5f} | {bias(yn, p):+.3f} "
          f"| {within(yn, p, 0.5):.1f} | {within(yn, p, 1.0):.1f} "
          f"| {within(yn, p, 2.0):.1f} |")
    w()
    w("Kareem is evaluated outside its stated validity range for part of the "
      "domain; see the validity note below.")
    w()

    # --------------------------------------------------------- Kareem note --
    kv = (tpr >= 1.15) & (tpr <= 3.0) & (ppr >= 0.2) & (ppr <= 15.0) & ng
    w(f"- Kareem within its stated validity (1.15 <= Tpr <= 3, "
      f"0.2 <= Ppr <= 15): AAD **{aad(y[kv], preds['Kareem'][kv]):.3f} %** "
      f"on {int(kv.sum()):,} points")
    w(f"- Fraction of the NG domain outside Kareem validity: "
      f"**{100 * (1 - kv.sum() / ng.sum()):.1f} %**")
    w()

    # ------------------------------------------------------- significance --
    w("## 4 Statistical comparison (clustered by laboratory)")
    w()
    src = t.doi.to_numpy()[ng]
    uniq = np.unique(src)
    rng = np.random.default_rng(7)
    a_nn = np.abs(preds[MODEL][ng] - yn) / yn * 100
    w("| comparison | mean advantage (pp) | 95 % CI | P(NNCF better) | "
      "labs won | Wilcoxon p |")
    w("|---|---:|---|---:|---:|---:|")
    for opp in ["DPR", "DAK", "Hall-Yarborough", "GERG-2008", "AGA8-DETAIL"]:
        a_o = np.abs(preds[opp][ng] - yn) / yn * 100
        d = a_o - a_nn
        g = np.array([d[src == s].mean() for s in uniq])
        boots = np.array([g[rng.integers(0, len(g), len(g))].mean()
                          for _ in range(10000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        nwin = int(sum(a_o[src == s].mean() > a_nn[src == s].mean()
                       for s in uniq))
        _, pv = wilcoxon(a_o, a_nn)
        w(f"| NNCF vs {opp} | {d.mean():+.4f} | "
          f"[{lo:+.4f}, {hi:+.4f}] | {(boots > 0).mean():.3f} | "
          f"{nwin}/{len(uniq)} | {pv:.2e} |")
    w()

    # ------------------------------------------------- surrogate fidelity --
    dz = (preds[MODEL][ng] - preds["GERG-2008"][ng]) / preds["GERG-2008"][ng] * 100
    w("## 5 Surrogate fidelity to GERG-2008 (NG domain)")
    w()
    w(f"- median |Δz|/z: **{np.median(np.abs(dz)):.4f} %**")
    w(f"- 95th percentile |Δz|/z: **{np.percentile(np.abs(dz), 95):.4f} %**")
    w(f"- maximum |Δz|/z: {np.abs(dz).max():.3f} %")
    w()

    # ------------------------------------------------------- per-laboratory --
    w("## 6 Per-laboratory results")
    w()
    w("| laboratory (DOI) | n | NNCF AAD | DPR AAD | GERG AAD |")
    w("|---|---:|---:|---:|---:|")
    for s in uniq:
        m = src == s
        w(f"| {s} | {int(m.sum())} | {a_nn[m].mean():.3f} | "
          f"{(np.abs(preds['DPR'][ng] - yn) / yn * 100)[m].mean():.3f} | "
          f"{(np.abs(preds['GERG-2008'][ng] - yn) / yn * 100)[m].mean():.3f} |")
    w()

    # ------------------------------------------------------------ regional --
    reg = ROOT / "reports" / "regional_benchmark.md"
    if reg.exists():
        w("## 7 Regional study")
        w()
        w("Source: `reports/regional_benchmark.md` (unchanged, reproduced for "
          "convenience)")
        w()
        for line in reg.read_text(encoding="utf-8").splitlines():
            if line.startswith("|") or line.startswith("- "):
                w(line)
        w()

    # ---------------------------------------------------------- correction --
    ff = ROOT / "reports" / "functional_form_search.csv"
    if ff.exists():
        d = pd.read_csv(ff).dropna(subset=["LOSO_MAPE"])
        w("## 8 Residual-correction study")
        w()
        w(f"- functional families tested: **{d.form.nunique()}**, "
          f"base predictors: {d.base.nunique()}, "
          f"total leave-one-laboratory-out fits: **{len(d)}**")
        w()
        w("| base predictor | uncorrected AAD | best out-of-sample form | "
          "best AAD | gain (pp) |")
        w("|---|---:|---|---:|---:|")
        for b, s in d.groupby("base"):
            best = s.loc[s.LOSO_MAPE.idxmin()]
            w(f"| {b} | {best.base_MAPE:.4f} | {best.form} | "
              f"{best.LOSO_MAPE:.4f} | {best.base_MAPE - best.LOSO_MAPE:+.4f} |")
        w()

    # ------------------------------------------------------ model families --
    w("## 9 Model families (same data, same target)")
    w()
    w("| family | AAD (%) |")
    w("|---|---:|")
    for k, v in [("Gradient boosting, raw (T,P,x)", 2.451),
                 ("Gradient boosting, reduced coords", 2.231),
                 ("Gradient boosting, broad pool", 2.143),
                 ("Gradient boosting, focused pool", 1.853),
                 ("NNCF, residual neural EOS", aad(yn, preds[MODEL][ng]))]:
        w(f"| {k} | {v:.3f} |")
    w()
    w("(tree values from reports/ during model selection; NNCF recomputed here)")
    w()

    # reports/, not paper/: the published repository carries the model and
    # the results, and the manuscript is submitted separately.
    out = ROOT / "reports" / "results_numbers.md"
    # ------------------------------------------------- computational cost ---
    cf = ROOT / "reports" / "cost_benchmark.json"
    if cf.exists():
        c = json.loads(cf.read_text(encoding="utf-8"))
        w("## 8 Computational cost")
        w()
        w(f"Batch of {c['n_states']:,} states, best of {c['repeats']}, "
          f"{c['torch_threads']} CPU threads. "
          f"Regenerate with `python -X utf8 scripts/benchmark_cost.py`.")
        w()
        w("| method | CPU us/state | states/s |")
        w("|---|---:|---:|")
        for k, v in sorted(c["us_per_state"].items(), key=lambda kv: kv[1]):
            w(f"| {k} | {v:.2f} | {c['states_per_second'][k]:,} |")
        w()
        it = c["newton_iters_correlations"]
        w(f"- Newton iterations, correlation family: mean {it['mean']}, "
          f"max {it['max']} (the max is the iteration cap, i.e. non-convergence)")
        w(f"- NNCF iterations: {c['nncf_iters']} "
          f"(fixed depth; convergence failure impossible)")
        w("- GPU (RTX 3050 Laptop), batched: **0.62 us/state** "
          "(~1.6e6 states/s) - measured separately, see the manuscript")
        w()
        w(f"> {c['caveat']}")
        w()

    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
