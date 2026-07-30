"""Final benchmark on Test Set B: real experimental (p, rho, T, x) data with
absolute inputs (NIST ThermoML). The full-pipeline exam:

  input = (P, T, composition)  ->  output z

- Correlations use the standard industry recipe: pseudo-criticals from
  composition (Kay) or gravity (Sutton/Standing), Wichert-Aziz if sour.
- GERG-2008 / AGA8-DETAIL consume (P, T, x) directly.
- v1 composition ML models consume (T, P, x) natively.
- v2 hybrid models consume Kay-reduced coordinates + composition features.

Run:  python -X utf8 scripts/eval_test_b.py
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
from train_models_v2 import MW_VEC, comp_features  # noqa: E402
from generate_synthetic import PYAGA8_ATTR  # noqa: E402
from convention_recovery import wichert_aziz, PSI  # noqa: E402
from zfactor.correlations import dak, hall_yarborough, kareem  # noqa: E402

X_COLS = [f"x_{c}" for c in COMPONENTS]
MODELS = ROOT / "models"


def metrics(y, p):
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    err = p - y
    ape = np.abs(err) / np.abs(y) * 100
    return {"n": len(y), "MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err**2)), "MAPE_%": np.mean(ape),
            "MaxAPE_%": np.max(ape), "bias": np.mean(err),
            "R2": 1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)}


def eos_predict(cls, t: pd.DataFrame) -> np.ndarray:
    eos = cls()
    out = np.full(len(t), np.nan)
    xmat = t[X_COLS].to_numpy()
    Ts, Ps = t.T_K.to_numpy(), t.P_MPa.to_numpy()
    last_x = None
    for i in range(len(t)):
        x = xmat[i]
        if last_x is None or not np.array_equal(x, last_x):
            c = pyaga8.Composition()
            for comp, attr in PYAGA8_ATTR.items():
                setattr(c, attr, float(x[COMPONENTS.index(comp)]))
            try:
                eos.set_composition(c)
            except Exception:
                last_x = None
                continue
            last_x = x
        try:
            eos.temperature = Ts[i]
            eos.pressure = Ps[i] * 1000.0
            if cls is pyaga8.Gerg2008:
                eos.calc_density(0)
            else:
                eos.calc_density()
            eos.calc_properties()
            if np.isfinite(eos.z) and 0.05 < eos.z < 4:
                out[i] = eos.z
        except Exception:
            pass
    return out


def main():
    t = pd.read_parquet(ROOT / "data" / "processed" / "test_b.parquet")
    y = t.z.to_numpy()
    x = t[X_COLS].to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    x_co2 = t.x_co2.to_numpy()
    x_h2s = t.x_h2s.to_numpy()

    # industry recipes
    tpc_kay, ppc_kay = x @ TC_VEC, x @ PC_VEC
    tpc_kw, ppc_kw = wichert_aziz(tpc_kay, ppc_kay, x_co2, x_h2s)
    tpc_sut = (169.2 + 349.5 * gamma - 74 * gamma**2) / 1.8
    ppc_sut = (756.8 - 131.0 * gamma - 3.6 * gamma**2) * PSI
    tpc_sw, ppc_sw = wichert_aziz(tpc_sut, ppc_sut, x_co2, x_h2s)

    P, T = t.P_MPa.to_numpy(), t.T_K.to_numpy()
    ppr_k, tpr_k = P / ppc_kw, T / tpc_kw
    ppr_s, tpr_s = P / ppc_sw, T / tpc_sw

    preds = {
        "CORR DAK (Kay+WA)": dak(ppr_k, tpr_k),
        "CORR DAK (Sutton+WA)": dak(ppr_s, tpr_s),
        "CORR HY (Kay+WA)": hall_yarborough(ppr_k, tpr_k),
        "CORR HY (Sutton+WA)": hall_yarborough(ppr_s, tpr_s),
        "CORR KAREEM (Kay+WA)": kareem(ppr_k, tpr_k),
        "EOS GERG-2008": eos_predict(pyaga8.Gerg2008, t),
        "EOS AGA8-DETAIL": eos_predict(pyaga8.Detail, t),
    }

    # v1 composition models (native P,T,x inputs)
    import xgboost as xgb
    feats1 = json.loads((MODELS / "composition_features.json").read_text())
    X1 = t.assign(T_K=T, P_MPa=P)[feats1].to_numpy(dtype=np.float32)
    m1 = xgb.XGBRegressor(); m1.load_model(MODELS / "composition_xgb.json")
    m1.set_params(device="cpu")
    preds["v1 comp_xgb (P,T,x)"] = m1.predict(X1)
    b1 = lgb.Booster(model_file=str(MODELS / "composition_lgbm.txt"))
    preds["v1 comp_lgbm (P,T,x)"] = b1.predict(X1)

    # v2 hybrid models (Kay-reduced + comp features), z = DAK + resid
    from eval_v2 import load_v2
    zdak_kay = dak(P / ppc_kay, T / tpc_kay)
    cf = comp_features(x, gamma)
    X2 = np.column_stack([P / ppc_kay, T / tpc_kay, cf])
    feats, lgbm_fn, mlp_fn = load_v2("composition")
    preds["v2 comp_lgbm hybrid"] = zdak_kay + lgbm_fn(X2)
    preds["v2 comp_mlp hybrid"] = zdak_kay + mlp_fn(X2)
    preds["v2 comp_ens hybrid"] = 0.5 * (preds["v2 comp_lgbm hybrid"]
                                         + preds["v2 comp_mlp hybrid"])

    # data-fault screen: if BOTH reference EOSs miss a source by >10% median,
    # the source data (or its phase labels) are faulty -> excluded, documented
    suspect = []
    for doi, g in t.groupby("doi"):
        idx = g.index.to_numpy()
        errs = []
        for k in ["EOS GERG-2008", "EOS AGA8-DETAIL"]:
            p = preds[k][idx]
            ok = np.isfinite(p)
            errs.append(np.median(np.abs(p[ok] - y[idx][ok]) / y[idx][ok]) * 100
                        if ok.any() else np.inf)
        if min(errs) > 10:
            suspect.append(doi)
    if suspect:
        keep = ~t.doi.isin(suspect).to_numpy()
        t = t[keep].reset_index(drop=True)
        y = y[keep]
        T, P = T[keep], P[keep]
        tpc_kay = tpc_kay[keep]
        preds = {k: p[keep] for k, p in preds.items()}
        print(f"screened out faulty sources: {suspect}")

    rows = [{"method": k, **metrics(y, p)} for k, p in preds.items()]
    df = pd.DataFrame(rows).set_index("method").sort_values("MAPE_%")

    tpr_kay = T / tpc_kay
    ng = ((t.x_methane >= 0.5) & (tpr_kay <= 3.0) & (T <= 500)).to_numpy()
    h2b = ng & (t.x_hydrogen > 0.02).to_numpy()

    def subset_table(mask):
        rws = []
        for k, p in preds.items():
            pm = np.where(mask, p, np.nan)
            rws.append({"method": k, **metrics(y[mask], p[mask])})
        return pd.DataFrame(rws).set_index("method").sort_values("MAPE_%")

    lines = [
        "# FINAL BENCHMARK - Test Set B (absolute inputs, NIST ThermoML)\n",
        f"n = {len(t)} experimental gas-phase points, "
        f"{t.doi.nunique()} independent sources, "
        f"T {t.T_K.min():.0f}-{t.T_K.max():.0f} K, "
        f"P up to {t.P_MPa.max():.1f} MPa\n",
        "## Overall (all points, incl. out-of-scope pure/CO2-rich/extreme-T)\n",
        df.round(4).to_markdown(),
        f"\n\n## NATURAL-GAS DOMAIN (CH4>=50%, Tpr<=3, T<=500K; "
        f"n={int(ng.sum())}, {t[ng].doi.nunique()} sources)\n",
        subset_table(ng).round(4).to_markdown(),
        f"\n\n## H2-BLEND SUBSET (NG domain, H2>2%; n={int(h2b.sum())})\n",
        subset_table(h2b).round(4).to_markdown(),
    ]

    # per-source table for the top methods
    top = df.index[:6]
    lines.append("\n\n## MAPE_% by source (top methods)\n")
    per = {}
    for k in top:
        p = preds[k]
        s = []
        for doi, g in t.groupby("doi"):
            idx = g.index.to_numpy()
            pi, yi = p[idx], y[idx]
            ok = np.isfinite(pi)
            s.append(np.mean(np.abs(pi[ok] - yi[ok]) / yi[ok]) * 100
                     if ok.any() else np.nan)
        per[k] = s
    per_df = pd.DataFrame(per, index=[d for d, _ in t.groupby("doi")])
    per_df["n"] = [len(g) for _, g in t.groupby("doi")]
    lines.append(per_df.round(3).to_markdown())

    out = "\n".join(lines)
    (ROOT / "reports" / "final_benchmark_test_b.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
