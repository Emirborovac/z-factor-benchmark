"""Step 4: one-shot final evaluation on the experimental test set.

Every classical correlation and every trained model predicts the 1,079
lab-measured Z-factors (first and only use of the test tier), scored overall,
per region, and on the Ppr<=15 interpolation subset vs Ppr>15 extrapolation.
Also renders the paper figures.

Composition-track caveat: experimental records publish only reduced
coordinates, so absolute P,T are reconstructed via Kay's rule pseudo-criticals
(flagged in the report).

Run:  python -X utf8 scripts/eval_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from zfactor.correlations import CORRELATIONS  # noqa: E402

MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"
X_COLS = [f"x_{c}" for c in COMPONENTS]


# --------------------------------------------------------------------------- #
def load_test():
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    t = m[(m.tier == "experimental") & (m.quality_flag == "ok")].copy()
    x = t[X_COLS].fillna(0.0).to_numpy()
    t["tpc_kay"] = x @ TC_VEC
    t["ppc_kay"] = x @ PC_VEC
    t["T_K_rec"] = t.Tpr * t.tpc_kay
    t["P_MPa_rec"] = t.Ppr * t.ppc_kay
    return t


def load_models():
    chart_feats = json.loads((MODELS / "chart_features.json").read_text())
    comp_feats = json.loads((MODELS / "composition_features.json").read_text())

    def xgb_load(name):
        m = xgb.XGBRegressor()
        m.load_model(MODELS / f"{name}_xgb.json")
        m.set_params(device="cpu")
        return lambda X: m.predict(X)

    def lgb_load(name):
        b = lgb.Booster(model_file=str(MODELS / f"{name}_lgbm.txt"))
        return lambda X: b.predict(X)

    def sk_load(name):
        p = joblib.load(MODELS / f"{name}_mlp.joblib")
        return lambda X: p.predict(X)

    def torch_load(name):
        import torch
        import torch.nn as nn
        ck = torch.load(MODELS / f"{name}_mlp.pt", weights_only=False)
        net = nn.Sequential(
            nn.Linear(ck["in_dim"], 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        mu, sd = ck["mu"], ck["sd"]

        def pred(X):
            with torch.no_grad():
                Xs = torch.tensor(((X - mu) / sd).astype(np.float32))
                return net(Xs).numpy().ravel()
        return pred

    return {
        "ML chart_xgb": ("chart", chart_feats, xgb_load("chart")),
        "ML chart_lgbm": ("chart", chart_feats, lgb_load("chart")),
        "ML chart_mlp": ("chart", chart_feats, sk_load("chart")),
        "ML comp_xgb": ("composition", comp_feats, xgb_load("composition")),
        "ML comp_lgbm": ("composition", comp_feats, lgb_load("composition")),
        "ML comp_mlp": ("composition", comp_feats, torch_load("composition")),
    }


def metrics(y, p):
    err = p - y
    ape = np.abs(err) / np.abs(y) * 100
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err**2)),
            "MAPE_%": np.mean(ape), "MaxAPE_%": np.max(ape),
            "bias": np.mean(err),
            "R2": 1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)}


# --------------------------------------------------------------------------- #
def predict_all(t: pd.DataFrame) -> pd.DataFrame:
    """Return test table + one prediction column per method."""
    preds = pd.DataFrame(index=t.index)
    ppr, tpr = t.Ppr.to_numpy(), t.Tpr.to_numpy()

    for name, func in CORRELATIONS.items():
        preds[f"CORR {name}"] = func(ppr, tpr)

    models = load_models()
    Xchart = t[["Ppr", "Tpr"]].to_numpy()
    comp_feats = json.loads((MODELS / "composition_features.json").read_text())
    Xcomp = t.assign(T_K=t.T_K_rec, P_MPa=t.P_MPa_rec)[comp_feats] \
        .fillna(0.0).to_numpy(dtype=np.float32)
    for name, (track, feats, fn) in models.items():
        preds[name] = fn(Xchart if track == "chart" else Xcomp)
    return preds


def score_table(t, preds, mask=None, label="all"):
    idx = t.index if mask is None else t.index[mask]
    y = t.loc[idx, "z"].to_numpy()
    rows = []
    for col in preds.columns:
        p = preds.loc[idx, col].to_numpy()
        ok = np.isfinite(p)
        rows.append({"method": col, "subset": label, "n": int(ok.sum()),
                     **metrics(y[ok], p[ok])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def figures(t, preds):
    FIGS.mkdir(parents=True, exist_ok=True)
    y = t.z.to_numpy()

    # 1. cross-plots for key methods
    keys = ["CORR DAK", "CORR HY", "ML chart_lgbm", "ML comp_lgbm"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    for ax, k in zip(axes, keys):
        p = preds[k].to_numpy()
        mape = np.mean(np.abs(p - y) / y) * 100
        ax.scatter(y, p, s=6, alpha=0.35)
        lim = [0.3, 2.3]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(title=f"{k}  (MAPE {mape:.2f}%)", xlabel="measured z",
               xlim=lim, ylim=lim)
    axes[0].set_ylabel("predicted z")
    fig.suptitle("Predicted vs experimental Z (n=1079, unseen test set)")
    fig.tight_layout()
    fig.savefig(FIGS / "crossplots_test.png", dpi=150)

    # 2. error vs Ppr
    fig, ax = plt.subplots(figsize=(9, 5))
    order = np.argsort(t.Ppr.to_numpy())
    for k in keys:
        ape = (np.abs(preds[k] - t.z) / t.z * 100).to_numpy()[order]
        ax.plot(t.Ppr.to_numpy()[order], pd.Series(ape).rolling(75, center=True,
                min_periods=25).mean(), label=k, lw=1.6)
    ax.axvline(15, color="gray", ls=":", lw=1)
    ax.text(15.2, ax.get_ylim()[1] * 0.9, "chart-data limit", fontsize=8)
    ax.set(xlabel="Ppr", ylabel="rolling mean APE (%)",
           title="Error vs pseudo-reduced pressure (experimental test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "error_vs_ppr.png", dpi=150)

    # 3. SK isotherm overlay (chart domain)
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    sk = m[(m.tier == "chart_digitized")]
    models = load_models()
    chart_lgbm = models["ML chart_lgbm"][2]
    from zfactor.correlations import dak, hall_yarborough
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for tpr_v, color in [(1.05, "tab:red"), (1.3, "tab:orange"),
                         (1.5, "tab:green"), (2.0, "tab:blue")]:
        c = sk[np.isclose(sk.Tpr, tpr_v)].sort_values("Ppr")
        ax.plot(c.Ppr, c.z, ".", ms=2.5, color=color, alpha=0.5,
                label=f"SK chart Tpr={tpr_v}")
        g = np.linspace(0.2, 15, 300)
        ax.plot(g, chart_lgbm(np.column_stack([g, np.full_like(g, tpr_v)])),
                "-", color=color, lw=1.4)
        ax.plot(g, dak(g, np.full_like(g, tpr_v)), "--", color=color, lw=1.0,
                alpha=0.7)
    ax.set(xlabel="Ppr", ylabel="z",
           title="Standing-Katz isotherms: chart data (dots), ML LightGBM (solid), DAK (dashed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "sk_isotherm_overlay.png", dpi=150)

    # 4. per-region MAPE bars
    fig, ax = plt.subplots(figsize=(10, 5))
    regs = ["near_critical_trough", "moderate", "high_pressure",
            "ultra_high_pressure"]
    width = 0.2
    xpos = np.arange(len(regs))
    for i, k in enumerate(keys):
        vals = []
        for r in regs:
            mask = (t.region == r).to_numpy()
            if mask.sum() == 0:
                vals.append(np.nan)
                continue
            p = preds[k].to_numpy()[mask]
            vals.append(np.mean(np.abs(p - y[mask]) / y[mask]) * 100)
        ax.bar(xpos + i * width, vals, width, label=k)
    ax.set_xticks(xpos + 1.5 * width)
    ax.set_xticklabels([r.replace("_", "\n") for r in regs], fontsize=9)
    ax.set(ylabel="MAPE (%)", title="Accuracy by region (experimental test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "mape_by_region.png", dpi=150)
    plt.close("all")


# --------------------------------------------------------------------------- #
def main():
    t = load_test()
    preds = predict_all(t)
    preds.to_parquet(ROOT / "reports" / "test_predictions.parquet")

    full = score_table(t, preds, label="all_1079")
    interp = score_table(t, preds, (t.Ppr <= 15).to_numpy(), "Ppr<=15")
    extrap = score_table(t, preds, (t.Ppr > 15).to_numpy(), "Ppr>15")
    allscores = pd.concat([full, interp, extrap], ignore_index=True)
    allscores.to_csv(ROOT / "reports" / "final_scores.csv", index=False)

    def fmt(df):
        return (df.drop(columns="subset").set_index("method")
                .sort_values("MAPE_%").round(4).to_markdown())

    lines = [
        "# Final benchmark - experimental test set (n=1079, one-shot)\n",
        "Composition-track inputs use P,T reconstructed from published "
        "Ppr/Tpr via Kay's rule (see DATASHEET).\n",
        f"\n## All 1,079 points\n\n{fmt(full)}",
        f"\n\n## Interpolation regime (Ppr <= 15, n={int((t.Ppr<=15).sum())})\n\n{fmt(interp)}",
        f"\n\n## Extrapolation regime (Ppr > 15, n={int((t.Ppr>15).sum())})\n\n{fmt(extrap)}",
        "\n\nFigures: reports/figures/*.png",
    ]
    (ROOT / "reports" / "final_benchmark.md").write_text("\n".join(lines),
                                                         encoding="utf-8")
    print("\n".join(lines[:3]))
    figures(t, preds)
    print("\nWrote reports/final_benchmark.md, final_scores.csv, figures/")


if __name__ == "__main__":
    main()
