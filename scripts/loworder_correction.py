"""Low-order, physically-motivated correction factors, tested leave-one-
source-out. The controlled counterpart to the high-capacity experiment:
fewer parameters should generalize better when only ~10 sources exist.

Forms tested (theta fitted by least squares on held-in sources only):
  F0 scale        z' = a*z
  F1 affine       z' = a*z + b
  F2 departure    z' = z + c*(1-z)            (error scales with non-ideality)
  F3 dep-linear   z' = z + (a + b*Ppr)*(1-z)
  F4 dep-temp     z' = z + (a + b*Ppr + c/Tpr)*(1-z)

Applied identically to every base method as a control.

Run:  python -X utf8 scripts/loworder_correction.py
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


def mape(y, p):
    return float(np.mean(np.abs(p - y) / np.abs(y)) * 100)


def design(form, z, ppr, tpr):
    """Return (A, apply) so that z_corrected = apply(theta) with A @ theta."""
    d = 1.0 - z
    if form == "F0_scale":
        return np.column_stack([z]), lambda th, Z, A: A @ th
    if form == "F1_affine":
        return np.column_stack([z, np.ones_like(z)]), lambda th, Z, A: A @ th
    if form == "F2_departure":
        return np.column_stack([d]), lambda th, Z, A: Z + A @ th
    if form == "F3_dep_linear":
        return np.column_stack([d, d * ppr]), lambda th, Z, A: Z + A @ th
    if form == "F4_dep_temp":
        return (np.column_stack([d, d * ppr, d / tpr]),
                lambda th, Z, A: Z + A @ th)
    raise ValueError(form)


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
    for key, label in [("native_direct", "ML-native"),
                       ("hybrid_reduced", "ML-hybrid")]:
        if (V3 / f"{key}_meta.json").exists():
            meta = json.loads((V3 / f"{key}_meta.json").read_text())
            fn = lgb.Booster(model_file=str(V3 / f"{key}_lgbm.txt")).predict
            cols = {"Ppr": ppr, "Tpr": tpr, "T_K": T, "P_MPa": P}
            cf = comp_features(x, gamma)
            for i, n in enumerate(COMP_FEATS):
                cols[n] = cf[:, i]
            for i, c in enumerate(X_COLS):
                cols[c] = x[:, i]
            X = np.column_stack([cols[f] for f in meta["features"]]).astype(np.float32)
            p = fn(X)
            base[label] = C.dak(ppr, tpr) + p if meta["residual"] else p
    if not any(k.startswith("ML") for k in base) and (V2 / "composition_lgbm.txt").exists():
        b = lgb.Booster(model_file=str(V2 / "composition_lgbm.txt"))
        base["ML-hybrid(v2)"] = C.dak(ppr, tpr) + b.predict(
            np.column_stack([ppr, tpr, comp_features(x, gamma)]))

    gerg = base["GERG-2008"]
    suspect = [doi for doi, g in t.groupby("doi")
               if np.nanmedian(np.abs(gerg[g.index.to_numpy()] - y[g.index.to_numpy()])
                               / y[g.index.to_numpy()]) * 100 > 10]
    ng = ((~t.doi.isin(suspect)).to_numpy() & (t.x_methane >= 0.5).to_numpy()
          & (tpr <= 3.0) & (T <= 500))
    tn = t[ng].reset_index(drop=True)
    yn, pprn, tprn = y[ng], ppr[ng], tpr[ng]
    src = tn.doi.to_numpy()
    uniq = np.unique(src)
    print(f"natural-gas domain: n={len(tn)}, {len(uniq)} sources\n")

    forms = ["F0_scale", "F1_affine", "F2_departure", "F3_dep_linear",
             "F4_dep_temp"]
    rows = []
    for name, pall in base.items():
        z0 = pall[ng]
        rec = {"method": name, "uncorrected": round(mape(yn, z0), 4)}
        for form in forms:
            A, apply = design(form, z0, pprn, tprn)
            zc = np.empty_like(z0)
            for s in uniq:
                te = src == s
                tr = ~te
                target = yn[tr] if form in ("F0_scale", "F1_affine") \
                    else yn[tr] - z0[tr]
                th, *_ = np.linalg.lstsq(A[tr], target, rcond=None)
                zc[te] = apply(th, z0[te], A[te])
            rec[form] = round(mape(yn, zc), 4)
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("method")
    best = df.idxmin(axis=1)
    df["best_form"] = best
    df["best_MAPE"] = df[forms + ["uncorrected"]].min(axis=1)
    df["gain_pp"] = (df["uncorrected"] - df["best_MAPE"]).round(4)
    print(df.to_string())

    out = ["# Low-order correction factors (leave-one-source-out)\n",
           "Coefficients fitted by least squares on held-in sources only; "
           "every score is out-of-sample for a new lab.\n",
           f"\nNatural-gas domain: n={len(tn)}, {len(uniq)} sources\n\n",
           df.to_markdown()]
    (ROOT / "reports" / "loworder_correction.md").write_text(
        "\n".join(out), encoding="utf-8")
    print("\nwrote reports/loworder_correction.md")


if __name__ == "__main__":
    main()
