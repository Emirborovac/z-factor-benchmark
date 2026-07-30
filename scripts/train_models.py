"""Step 3: train ML models for both tracks.

Track "chart":       (Ppr, Tpr) -> z          | train/val = SK chart split
Track "composition": (T, P, x_1..x_21) -> z   | train/val = synthetic tier,
                                                split at composition level
Models per track: XGBoost, LightGBM, MLP (sklearn for chart, torch/GPU for
composition). The experimental test set is NOT touched here - final scoring
happens once, in the evaluation step.

Run:  python -X utf8 scripts/train_models.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "models"
SEED = 20260728

X_COLS_COMP = None  # filled in main


def metrics(y, p):
    err = p - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / np.abs(y)) * 100),
        "R2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
    }


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
def load_chart_track():
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    sk = m[(m.tier == "chart_digitized") & (m.quality_flag == "ok")]
    tr, va = sk[sk.split == "train"], sk[sk.split == "val"]
    F = ["Ppr", "Tpr"]
    return (tr[F].to_numpy(), tr.z.to_numpy(),
            va[F].to_numpy(), va.z.to_numpy(), F)


def load_composition_track():
    pts = pd.read_parquet(ROOT / "data" / "synthetic" / "points.parquet")
    comps = pd.read_parquet(ROOT / "data" / "synthetic" / "compositions.parquet")
    xcols = [c for c in comps.columns if c.startswith("x_")]
    F = ["T_K", "P_MPa"] + xcols
    df = pts.merge(comps[["comp_id", "split"] + xcols], on="comp_id", how="left")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    return (tr[F].to_numpy(dtype=np.float32), tr.z.to_numpy(dtype=np.float32),
            va[F].to_numpy(dtype=np.float32), va.z.to_numpy(dtype=np.float32), F)


# --------------------------------------------------------------------------- #
def train_xgb(Xtr, ytr, Xva, yva, name):
    # with few features, column subsampling floors to 1 feature/tree and
    # cripples the fit -> use all columns for small feature spaces
    csb = 1.0 if Xtr.shape[1] <= 4 else 0.9
    model = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=9,
        subsample=0.9, colsample_bytree=csb, min_child_weight=5,
        tree_method="hist", device="cuda", random_state=SEED,
        early_stopping_rounds=50, eval_metric="rmse")
    try:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    except Exception as e:                      # no GPU -> CPU fallback
        log(f"  xgb cuda failed ({type(e).__name__}), retrying on CPU")
        model.set_params(device="cpu")
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    model.save_model(MODELS / f"{name}_xgb.json")
    return lambda X: model.predict(X)


def train_lgbm(Xtr, ytr, Xva, yva, name):
    model = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.05, num_leaves=255,
        min_child_samples=20, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    model.booster_.save_model(str(MODELS / f"{name}_lgbm.txt"))
    return lambda X: model.predict(X)


def train_mlp_sklearn(Xtr, ytr, Xva, yva, name):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="tanh",
                     solver="adam", max_iter=800, random_state=SEED,
                     early_stopping=True, n_iter_no_change=30, tol=1e-7))
    pipe.fit(Xtr, ytr)
    joblib.dump(pipe, MODELS / f"{name}_mlp.joblib")
    return lambda X: pipe.predict(X)


def train_mlp_torch(Xtr, ytr, Xva, yva, name, epochs=8, batch=16384):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xt = torch.tensor((Xtr - mu) / sd)
    yt = torch.tensor(ytr).unsqueeze(1)
    Xv = torch.tensor((Xva - mu) / sd).to(dev)
    yv = torch.tensor(yva).unsqueeze(1).to(dev)

    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.SiLU(),
        nn.Linear(256, 256), nn.SiLU(),
        nn.Linear(256, 256), nn.SiLU(),
        nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     generator=torch.Generator().manual_seed(SEED))
    lossf = nn.MSELoss()
    best = float("inf")
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xv), yv).item()
        log(f"  mlp epoch {ep+1}/{epochs} val_mse={vl:.3e}")
        if vl < best:
            best = vl
            torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd,
                        "in_dim": Xtr.shape[1]}, MODELS / f"{name}_mlp.pt")

    ck = torch.load(MODELS / f"{name}_mlp.pt", weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    def predict(X):
        with torch.no_grad():
            Xs = torch.tensor(((X - mu) / sd).astype(np.float32)).to(dev)
            out = []
            for i in range(0, len(Xs), 262144):
                out.append(net(Xs[i:i + 262144]).cpu().numpy().ravel())
        return np.concatenate(out)
    return predict


# --------------------------------------------------------------------------- #
def run_track(track, loader, mlp_trainer):
    Xtr, ytr, Xva, yva, feats = loader()
    log(f"track={track}: train {Xtr.shape}, val {Xva.shape}")
    results = {}
    for algo, trainer in [("xgb", train_xgb), ("lgbm", train_lgbm),
                          ("mlp", mlp_trainer)]:
        t0 = time.perf_counter()
        predict = trainer(Xtr, ytr, Xva, yva, track)
        res = {"train": metrics(ytr, predict(Xtr)),
               "val": metrics(yva, predict(Xva)),
               "fit_seconds": round(time.perf_counter() - t0, 1)}
        results[algo] = res
        log(f"  {track}/{algo}: val MAPE {res['val']['MAPE_%']:.4f}% "
            f"RMSE {res['val']['RMSE']:.5f} ({res['fit_seconds']}s)")
    (MODELS / f"{track}_features.json").write_text(json.dumps(feats))
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["chart", "composition", "all"],
                    default="all")
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    mfile = MODELS / "training_metrics.json"
    all_res = json.loads(mfile.read_text()) if mfile.exists() else {}
    if args.track in ("chart", "all"):
        all_res["chart"] = run_track("chart", load_chart_track,
                                     train_mlp_sklearn)
    if args.track in ("composition", "all"):
        all_res["composition"] = run_track("composition",
                                           load_composition_track,
                                           train_mlp_torch)
    mfile.write_text(json.dumps(all_res, indent=2))

    lines = ["# Training metrics (train/val only - test untouched)\n"]
    for track, algos in all_res.items():
        lines.append(f"\n## {track} track\n")
        rows = []
        for algo, r in algos.items():
            rows.append({"model": algo, **{f"val_{k}": round(v, 5) for k, v in r["val"].items()},
                         "fit_s": r["fit_seconds"]})
        lines.append(pd.DataFrame(rows).set_index("model").to_markdown())
    (ROOT / "reports" / "training_metrics.md").write_text("\n".join(lines),
                                                          encoding="utf-8")
    log("wrote models/ and reports/training_metrics.md")


if __name__ == "__main__":
    main()
