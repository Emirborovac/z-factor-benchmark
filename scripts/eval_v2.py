"""Evaluate v2 hybrid models on the experimental test set, side by side with
the classical correlations and the v1 leaders.

Run:  python -X utf8 scripts/eval_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS  # noqa: E402
from train_models_v2 import COMP_FEATS, MW_VEC, comp_features  # noqa: E402
from zfactor.correlations import dak, hall_yarborough, dpr, kareem  # noqa: E402

MODELS = ROOT / "models" / "v2"
X_COLS = [f"x_{c}" for c in COMPONENTS]


def metrics(y, p):
    err = p - y
    ape = np.abs(err) / np.abs(y) * 100
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err**2)),
            "MAPE_%": np.mean(ape), "MaxAPE_%": np.max(ape),
            "bias": np.mean(err),
            "R2": 1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)}


def load_v2(track):
    feats = json.loads((MODELS / f"{track}_features.json").read_text())
    booster = lgb.Booster(model_file=str(MODELS / f"{track}_lgbm.txt"))

    import torch
    import torch.nn as nn
    ck = torch.load(MODELS / f"{track}_mlp.pt", weights_only=False)
    net = nn.Sequential(
        nn.Linear(ck["in_dim"], 512), nn.SiLU(),
        nn.Linear(512, 512), nn.SiLU(),
        nn.Linear(512, 512), nn.SiLU(),
        nn.Linear(512, 256), nn.SiLU(),
        nn.Linear(256, 1))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    mu, sd = ck["mu"], ck["sd"]

    def mlp_pred(X):
        with torch.no_grad():
            return net(torch.tensor(((X - mu) / sd).astype(np.float32))) \
                .numpy().ravel()

    return feats, booster.predict, mlp_pred


def main():
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    t = m[(m.tier == "experimental") & (m.quality_flag == "ok")].copy()
    ppr, tpr, y = t.Ppr.to_numpy(), t.Tpr.to_numpy(), t.z.to_numpy()
    x = t[X_COLS].fillna(0.0).to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    cf = comp_features(x, gamma)

    zdak = dak(ppr, tpr)
    preds = {
        "CORR DAK": zdak,
        "CORR HY": hall_yarborough(ppr, tpr),
        "CORR DPR": dpr(ppr, tpr),
        "CORR KAREEM": kareem(ppr, tpr),
    }

    Xchart = np.column_stack([ppr, tpr])
    Xcomp = np.column_stack([ppr, tpr, cf])
    for track, X in [("chart", Xchart), ("composition", Xcomp)]:
        feats, lgbm_fn, mlp_fn = load_v2(track)
        assert len(feats) == X.shape[1], (track, feats)
        pl = zdak + lgbm_fn(X)
        pm = zdak + mlp_fn(X)
        preds[f"v2 {track}_lgbm"] = pl
        preds[f"v2 {track}_mlp"] = pm
        preds[f"v2 {track}_ens"] = 0.5 * (pl + pm)

    subsets = {
        "all_1079": np.ones(len(t), bool),
        "Ppr<=15": (ppr <= 15),
        "Ppr>15": (ppr > 15),
    }
    lines = ["# v2 final benchmark - experimental test set (one-shot for v2)\n"]
    for label, mask in subsets.items():
        rows = []
        for name, p in preds.items():
            ok = np.isfinite(p) & mask
            rows.append({"method": name, "n": int(ok.sum()),
                         **metrics(y[ok], p[ok])})
        df = pd.DataFrame(rows).set_index("method").sort_values("MAPE_%")
        lines.append(f"\n## {label} (n={int(mask.sum())})\n")
        lines.append(df.round(4).to_markdown())

    out = "\n".join(lines)
    (ROOT / "reports" / "final_benchmark_v2.md").write_text(out, encoding="utf-8")
    pd.DataFrame({k: v for k, v in preds.items()},
                 index=t.index).to_parquet(ROOT / "reports" / "test_predictions_v2.parquet")
    print(out)


if __name__ == "__main__":
    main()
