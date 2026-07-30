"""Lab-calibrated correction layer, evaluated leave-one-source-out.

Idea (standard engineering practice): a base predictor gives z_base; a small
model learns the systematic residual (z_lab - z_base) as a function of state
and composition. Deployed, this is "calibrate the tool to measured data".

Honest protocol:
  - Leave-One-Source-Out: the correction never sees the source it is scored on,
    so every reported number is out-of-sample for a new lab AND a new gas.
  - The SAME correction machinery is applied to every base method (ML, DAK,
    HY, GERG). If correcting DAK gains as much as correcting ML, the credit
    belongs to the correction, not the model - this is the control.
  - Low-capacity learner (shallow trees, heavy regularization): with only ~10
    sources, capacity is the enemy.

Run:  python -X utf8 scripts/calibrated_correction.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyaga8

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from train_models_v2 import COMP_FEATS, MW_VEC, comp_features  # noqa: E402
from convention_recovery import wichert_aziz  # noqa: E402
from eval_test_b import eos_predict  # noqa: E402
import zfactor.correlations as C  # noqa: E402

X_COLS = [f"x_{c}" for c in COMPONENTS]
V3 = ROOT / "models" / "v3"
V2 = ROOT / "models" / "v2"
SEED = 20260730


def load_lgbm_v3(name):
    meta = json.loads((V3 / f"{name}_meta.json").read_text())
    b = lgb.Booster(model_file=str(V3 / f"{name}_lgbm.txt"))
    return meta, b.predict


def mape(y, p):
    return float(np.mean(np.abs(p - y) / np.abs(y)) * 100)


def main():
    t = pd.read_parquet(ROOT / "data" / "processed" / "test_b.parquet")
    y = t.z.to_numpy()
    x = t[X_COLS].to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    tpc_kay, ppc_kay = x @ TC_VEC, x @ PC_VEC
    tpc_kw, ppc_kw = wichert_aziz(tpc_kay, ppc_kay, t.x_co2.to_numpy(),
                                  t.x_h2s.to_numpy())
    P, T = t.P_MPa.to_numpy(), t.T_K.to_numpy()
    ppr, tpr = P / ppc_kay, T / tpc_kay

    base = {
        "DAK": C.dak(P / ppc_kw, T / tpc_kw),
        "HY": C.hall_yarborough(P / ppc_kw, T / tpc_kw),
        "GERG-2008": eos_predict(pyaga8.Gerg2008, t),
    }
    # best available ML base
    for key, label in [("native_direct", "ML-native"),
                       ("hybrid_reduced", "ML-hybrid")]:
        if (V3 / f"{key}_meta.json").exists():
            meta, fn = load_lgbm_v3(key)
            cols = {"Ppr": ppr, "Tpr": tpr, "T_K": T, "P_MPa": P}
            cf = comp_features(x, gamma)
            for i, n in enumerate(COMP_FEATS):
                cols[n] = cf[:, i]
            for i, c in enumerate(X_COLS):
                cols[c] = x[:, i]
            X = np.column_stack([cols[f] for f in meta["features"]]).astype(np.float32)
            pred = fn(X)
            if meta["residual"]:
                pred = C.dak(ppr, tpr) + pred
            base[label] = pred
    if "ML-hybrid" not in base and (V2 / "composition_lgbm.txt").exists():
        b = lgb.Booster(model_file=str(V2 / "composition_lgbm.txt"))
        X = np.column_stack([ppr, tpr, comp_features(x, gamma)])
        base["ML-hybrid(v2)"] = C.dak(ppr, tpr) + b.predict(X)

    # screens: faulty sources + natural-gas domain
    gerg = base["GERG-2008"]
    suspect = [doi for doi, g in t.groupby("doi")
               if np.nanmedian(np.abs(gerg[g.index.to_numpy()] - y[g.index.to_numpy()])
                               / y[g.index.to_numpy()]) * 100 > 10]
    ng = ((~t.doi.isin(suspect)).to_numpy() & (t.x_methane >= 0.5).to_numpy()
          & (tpr <= 3.0) & (T <= 500))

    tn = t[ng].reset_index(drop=True)
    yn, xn = y[ng], x[ng]
    pprn, tprn = ppr[ng], tpr[ng]
    cfn = comp_features(xn, gamma[ng])
    src = tn.doi.to_numpy()
    uniq = np.unique(src)
    print(f"natural-gas domain: n={len(tn)}, sources={len(uniq)}\n")

    feat = np.column_stack([pprn, tprn, cfn])
    results = {}

    for name, pred_all in base.items():
        p0 = pred_all[ng]
        corrected = np.empty_like(p0)
        for s in uniq:
            te = src == s
            tr = ~te
            model = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.03, num_leaves=15,
                min_child_samples=60, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=5.0,
                random_state=SEED, n_jobs=-1, verbose=-1)
            Xtr = np.column_stack([feat[tr], p0[tr]])
            model.fit(Xtr, yn[tr] - p0[tr])
            corrected[te] = p0[te] + model.predict(
                np.column_stack([feat[te], p0[te]]))
        results[name] = {"base_MAPE": mape(yn, p0),
                         "calibrated_MAPE": mape(yn, corrected),
                         "per_source_base": {s: mape(yn[src == s], p0[src == s])
                                             for s in uniq},
                         "per_source_cal": {s: mape(yn[src == s], corrected[src == s])
                                            for s in uniq}}
        print(f"{name:16s} base {results[name]['base_MAPE']:.4f}%  ->  "
              f"LOSO-calibrated {results[name]['calibrated_MAPE']:.4f}%  "
              f"({results[name]['base_MAPE'] - results[name]['calibrated_MAPE']:+.4f} pp)")

    rows = []
    for k, v in results.items():
        rows.append({"method": k, "base_MAPE_%": round(v["base_MAPE"], 4),
                     "calibrated_MAPE_%": round(v["calibrated_MAPE"], 4),
                     "gain_pp": round(v["base_MAPE"] - v["calibrated_MAPE"], 4),
                     "base_acc_%": round(100 - v["base_MAPE"], 3),
                     "cal_acc_%": round(100 - v["calibrated_MAPE"], 3)})
    df = pd.DataFrame(rows).sort_values("calibrated_MAPE_%")

    per = pd.DataFrame({k: v["per_source_cal"] for k, v in results.items()})
    per["n"] = [int((src == s).sum()) for s in per.index]

    out = ["# Lab-calibrated correction layer (leave-one-source-out)\n",
           "Every number is out-of-sample: the correction never saw the source "
           "it is scored on. The same correction is applied to every base "
           "method as a control.\n",
           f"\nNatural-gas domain: n={len(tn)}, {len(uniq)} independent sources\n",
           "\n## Base vs calibrated\n", df.round(4).to_markdown(index=False),
           "\n\n## Calibrated MAPE_% per held-out source\n",
           per.round(3).to_markdown()]
    txt = "\n".join(out)
    (ROOT / "reports" / "calibrated_correction.md").write_text(txt, encoding="utf-8")
    print("\n" + per.round(3).to_string())
    print("\nwrote reports/calibrated_correction.md")


if __name__ == "__main__":
    main()
