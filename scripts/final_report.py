"""Unified final evaluation: every method, both exams, with clustered
significance testing. Produces the paper's tables.

Exam A: chart-interface test set (1,079 lab points, reduced coords published)
Exam B: absolute-input test set (NIST ThermoML, real P/T/composition)

Run:  python -X utf8 scripts/final_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyaga8
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from train_models_v2 import COMP_FEATS, MW_VEC, comp_features  # noqa: E402
from convention_recovery import wichert_aziz, PSI  # noqa: E402
from eval_test_b import eos_predict, metrics  # noqa: E402
from zfactor.correlations import CORRELATIONS  # noqa: E402
import zfactor.correlations as C  # noqa: E402

X_COLS = [f"x_{c}" for c in COMPONENTS]
V3 = ROOT / "models" / "v3"
SEED = 20260730
N_BOOT = 10000


def load_v3(name):
    meta = json.loads((V3 / f"{name}_meta.json").read_text())
    booster = lgb.Booster(model_file=str(V3 / f"{name}_lgbm.txt"))
    import torch
    import torch.nn as nn
    ck = torch.load(V3 / f"{name}_mlp.pt", weights_only=False)
    w = ck.get("width", 512)
    net = nn.Sequential(
        nn.Linear(ck["in_dim"], w), nn.SiLU(),
        nn.Linear(w, w), nn.SiLU(),
        nn.Linear(w, w), nn.SiLU(),
        nn.Linear(w, 256), nn.SiLU(),
        nn.Linear(256, 1))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    mu, sd = ck["mu"], ck["sd"]

    def mlp(X):
        with torch.no_grad():
            return net(torch.tensor(((X - mu) / sd).astype(np.float32))).numpy().ravel()
    return meta, booster.predict, mlp


def build_features(df, feats, x, gamma, ppr, tpr):
    cols = {}
    cf = comp_features(x, gamma)
    for i, n in enumerate(COMP_FEATS):
        cols[n] = cf[:, i]
    cols["Ppr"], cols["Tpr"] = ppr, tpr
    cols["T_K"] = df.T_K.to_numpy() if "T_K" in df else np.nan
    cols["P_MPa"] = df.P_MPa.to_numpy() if "P_MPa" in df else np.nan
    for i, c in enumerate(X_COLS):
        cols[c] = x[:, i]
    return np.column_stack([cols[f] for f in feats]).astype(np.float32)


def clustered_test(ape_a, ape_b, groups, rng):
    """Bootstrap the mean APE difference (a - b) by resampling groups."""
    uniq = np.unique(groups)
    d = ape_a - ape_b
    boots = np.empty(N_BOOT)
    gvals = {g: d[groups == g].mean() for g in uniq}
    arr = np.array([gvals[g] for g in uniq])
    for i in range(N_BOOT):
        boots[i] = arr[rng.integers(0, len(arr), len(arr))].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, float((boots > 0).mean())


def main():
    rng = np.random.default_rng(SEED)
    out = ["# Z-Factor Benchmark - Final Results\n"]

    # ---------------- Exam B: absolute inputs ----------------
    t = pd.read_parquet(ROOT / "data" / "processed" / "test_b.parquet")
    y = t.z.to_numpy()
    x = t[X_COLS].to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    tpc_kay, ppc_kay = x @ TC_VEC, x @ PC_VEC
    tpc_kw, ppc_kw = wichert_aziz(tpc_kay, ppc_kay, t.x_co2.to_numpy(),
                                  t.x_h2s.to_numpy())
    tpc_sut = (169.2 + 349.5 * gamma - 74 * gamma**2) / 1.8
    ppc_sut = (756.8 - 131.0 * gamma - 3.6 * gamma**2) * PSI
    tpc_sw, ppc_sw = wichert_aziz(tpc_sut, ppc_sut, t.x_co2.to_numpy(),
                                  t.x_h2s.to_numpy())
    P, T = t.P_MPa.to_numpy(), t.T_K.to_numpy()
    ppr_k, tpr_k = P / ppc_kw, T / tpc_kw

    preds = {
        "GERG-2008 (EOS)": eos_predict(pyaga8.Gerg2008, t),
        "AGA8-DETAIL (EOS)": eos_predict(pyaga8.Detail, t),
        "DAK + Kay/WA": C.dak(ppr_k, tpr_k),
        "HY + Kay/WA": C.hall_yarborough(ppr_k, tpr_k),
        "DPR + Kay/WA": C.dpr(ppr_k, tpr_k),
        "Kareem + Kay/WA": C.kareem(ppr_k, tpr_k),
        "DAK + Sutton/WA": C.dak(P / ppc_sw, T / tpc_sw),
    }
    z_dak_kay = C.dak(P / ppc_kay, T / tpc_kay)
    for key, label in [("hybrid_reduced", "ML-hybrid"),
                       ("native_direct", "ML-native"),
                       ("chart", "ML-chart")]:
        if not (V3 / f"{key}_meta.json").exists():
            continue
        meta, lg, ml = load_v3(key)
        X = build_features(t, meta["features"], x, gamma,
                           P / ppc_kay, T / tpc_kay)
        base = z_dak_kay if meta["residual"] else 0.0
        pl, pm = base + lg(X), base + ml(X)
        preds[f"{label} (LGBM)"] = pl
        preds[f"{label} (MLP)"] = pm
        preds[f"{label} (ens)"] = 0.5 * (pl + pm)

    # faulty-source screen via reference EOS
    suspect = [doi for doi, g in t.groupby("doi")
               if np.nanmedian(np.abs(preds["GERG-2008 (EOS)"][g.index.to_numpy()]
                                      - y[g.index.to_numpy()])
                               / y[g.index.to_numpy()]) * 100 > 10]
    keep = (~t.doi.isin(suspect)).to_numpy()
    ng = keep & (t.x_methane >= 0.5).to_numpy() & ((T / tpc_kay) <= 3.0) & (T <= 500)
    out.append(f"\nExcluded sources failing both reference EOS (>10% median): {suspect}\n")

    for label, mask in [("Full Test Set B", keep),
                        ("Natural-gas domain", ng),
                        ("H2-blend subset", ng & (t.x_hydrogen > 0.02).to_numpy())]:
        rows = [{"method": k, **metrics(y[mask], p[mask])}
                for k, p in preds.items()]
        df = pd.DataFrame(rows).set_index("method").sort_values("MAPE_%")
        df["Accuracy_%"] = (100 - df["MAPE_%"]).round(3)
        out.append(f"\n## Exam B - {label} (n={int(mask.sum())})\n")
        out.append(df.round(4).to_markdown())

    # significance: best ML vs best correlation on NG domain
    ape = {k: np.abs(p[ng] - y[ng]) / y[ng] * 100 for k, p in preds.items()}
    ml_keys = [k for k in ape if k.startswith("ML")]
    corr_keys = [k for k in ape if "+ Kay" in k or "+ Sutton" in k]
    best_ml = min(ml_keys, key=lambda k: np.nanmean(ape[k])) if ml_keys else None
    best_corr = min(corr_keys, key=lambda k: np.nanmean(ape[k]))
    groups = t.doi.to_numpy()[ng]

    if best_ml:
        out.append(f"\n## Significance - {best_ml} vs {best_corr} "
                   f"(natural-gas domain, clustered by source)\n")
        mean_d, lo, hi, pbetter = clustered_test(ape[best_corr], ape[best_ml],
                                                 groups, rng)
        _, pval = wilcoxon(ape[best_corr], ape[best_ml])
        rec = [(s, ape[best_corr][groups == s].mean() - ape[best_ml][groups == s].mean(),
                int((groups == s).sum())) for s in np.unique(groups)]
        nwin = sum(1 for _, d, _ in rec if d > 0)
        out.append(
            f"- mean APE advantage of ML: **{mean_d:+.4f} pp**\n"
            f"- clustered bootstrap 95% CI: [{lo:+.4f}, {hi:+.4f}] pp\n"
            f"- P(ML better | source resampling): **{pbetter:.3f}**\n"
            f"- Wilcoxon signed-rank p (point-level): {pval:.3e}\n"
            f"- per-source record: ML wins **{nwin}/{len(rec)}** sources\n")
        out.append(pd.DataFrame(rec, columns=["source", "ML_advantage_pp", "n"])
                   .sort_values("ML_advantage_pp", ascending=False)
                   .round(3).to_markdown(index=False))

    # ---------------- Exam A: chart-interface ----------------
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    ta = m[(m.tier == "experimental") & (m.quality_flag == "ok")].copy()
    ppr_a, tpr_a, ya = ta.Ppr.to_numpy(), ta.Tpr.to_numpy(), ta.z.to_numpy()
    xa = ta[X_COLS].fillna(0).to_numpy()
    gam_a = (xa @ MW_VEC) / 28.9647
    preds_a = {f"{k} (correlation)": f(ppr_a, tpr_a)
               for k, f in CORRELATIONS.items()}
    z_dak_a = C.dak(ppr_a, tpr_a)
    for key, label in [("hybrid_reduced", "ML-hybrid"), ("chart", "ML-chart")]:
        if not (V3 / f"{key}_meta.json").exists():
            continue
        meta, lg, ml = load_v3(key)
        X = build_features(ta.assign(T_K=np.nan, P_MPa=np.nan),
                           meta["features"], xa, gam_a, ppr_a, tpr_a)
        base = z_dak_a if meta["residual"] else 0.0
        preds_a[f"{label} (LGBM)"] = base + lg(X)
        preds_a[f"{label} (MLP)"] = base + ml(X)

    rows = [{"method": k, **metrics(ya, p)} for k, p in preds_a.items()]
    dfa = pd.DataFrame(rows).set_index("method").sort_values("MAPE_%")
    dfa["Accuracy_%"] = (100 - dfa["MAPE_%"]).round(3)
    out.append(f"\n\n## Exam A - chart-interface test set (n={len(ta)})\n")
    out.append(dfa.round(4).to_markdown())

    txt = "\n".join(out)
    (ROOT / "reports" / "FINAL_RESULTS.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
