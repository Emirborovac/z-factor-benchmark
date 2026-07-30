"""Is the ML win over the correlations real, or noise?

Three checks on the natural-gas domain of Test Set B:
1. Paired bootstrap over SOURCES (clustered) - the honest unit of resampling,
   since points within a paper share apparatus and gas.
2. Paired bootstrap over points (optimistic bound).
3. Wilcoxon signed-rank on per-point absolute percentage errors.
4. Per-source win/loss record.

Run:  python -X utf8 scripts/significance_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

SEED = 20260729
N_BOOT = 10000


def main():
    # rebuild predictions exactly as eval_test_b does
    import eval_test_b as E
    import pyaga8
    import lightgbm as lgb
    import json
    import xgboost as xgb
    from prepare_data import COMPONENTS, TC_VEC, PC_VEC
    from train_models_v2 import MW_VEC, comp_features
    from convention_recovery import wichert_aziz, PSI
    from zfactor.correlations import dak, hall_yarborough, kareem

    X_COLS = [f"x_{c}" for c in COMPONENTS]
    t = pd.read_parquet(ROOT / "data" / "processed" / "test_b.parquet")
    y = t.z.to_numpy()
    x = t[X_COLS].to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    tpc_kay, ppc_kay = x @ TC_VEC, x @ PC_VEC
    tpc_kw, ppc_kw = wichert_aziz(tpc_kay, ppc_kay, t.x_co2.to_numpy(),
                                  t.x_h2s.to_numpy())
    P, T = t.P_MPa.to_numpy(), t.T_K.to_numpy()

    preds = {
        "DAK (Kay+WA)": dak(P / ppc_kw, T / tpc_kw),
        "HY (Kay+WA)": hall_yarborough(P / ppc_kw, T / tpc_kw),
        "GERG-2008": E.eos_predict(pyaga8.Gerg2008, t),
    }
    from eval_v2 import load_v2
    zdak_kay = dak(P / ppc_kay, T / tpc_kay)
    X2 = np.column_stack([P / ppc_kay, T / tpc_kay, comp_features(x, gamma)])
    _, lgbm_fn, mlp_fn = load_v2("composition")
    preds["ML v2 hybrid (lgbm)"] = zdak_kay + lgbm_fn(X2)

    # same screens as the benchmark
    suspect = []
    for doi, g in t.groupby("doi"):
        idx = g.index.to_numpy()
        e = [np.median(np.abs(preds[k][idx] - y[idx]) / y[idx]) * 100
             for k in ["GERG-2008"]]
        if min(e) > 10:
            suspect.append(doi)
    keep = (~t.doi.isin(suspect)).to_numpy()
    ng = keep & (t.x_methane >= 0.5).to_numpy() & \
        ((T / tpc_kay) <= 3.0) & (T <= 500)

    tn = t[ng].reset_index(drop=True)
    yn = y[ng]
    ape = {k: np.abs(p[ng] - yn) / yn * 100 for k, p in preds.items()}
    sources = tn.doi.to_numpy()
    uniq = np.unique(sources)

    print(f"Natural-gas domain: n={ng.sum()} points, {len(uniq)} sources\n")
    print("Method MAPE_%:")
    for k, v in sorted(ape.items(), key=lambda kv: kv[1].mean()):
        print(f"  {k:22s} {v.mean():.4f}")

    ml = "ML v2 hybrid (lgbm)"
    rng = np.random.default_rng(SEED)

    for opp in ["DAK (Kay+WA)", "HY (Kay+WA)"]:
        d_point = ape[opp] - ape[ml]          # >0 means ML better
        obs = d_point.mean()

        # clustered bootstrap over sources
        wins = 0
        diffs = []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, len(uniq), replace=True)
            vals = []
            for s in pick:
                m = sources == s
                vals.append(d_point[m].mean())
            db = float(np.mean(vals))
            diffs.append(db)
            wins += db > 0
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])

        # point-level bootstrap
        idx = rng.integers(0, len(d_point), (2000, len(d_point)))
        dpb = d_point[idx].mean(1)
        lo_p, hi_p = np.percentile(dpb, [2.5, 97.5])

        stat, pval = wilcoxon(ape[opp], ape[ml])

        # per-source record
        rec = [(s, ape[opp][sources == s].mean() - ape[ml][sources == s].mean(),
                int((sources == s).sum())) for s in uniq]
        nwin = sum(1 for _, d, _ in rec if d > 0)

        print(f"\n=== ML vs {opp} ===")
        print(f"  mean APE advantage: {obs:+.4f} pp "
              f"(ML {ape[ml].mean():.3f}% vs {ape[opp].mean():.3f}%)")
        print(f"  clustered bootstrap (by source): 95% CI [{lo:+.4f}, {hi:+.4f}] pp, "
              f"P(ML better) = {wins/N_BOOT:.3f}")
        print(f"  point bootstrap:                 95% CI [{lo_p:+.4f}, {hi_p:+.4f}] pp")
        print(f"  Wilcoxon signed-rank p = {pval:.3e}")
        print(f"  per-source record: ML wins {nwin}/{len(uniq)} sources")
        for s, d, n in sorted(rec, key=lambda r: -r[1]):
            print(f"     {'WIN ' if d > 0 else 'LOSS'} {d:+7.3f} pp  n={n:5d}  {s}")


if __name__ == "__main__":
    main()
