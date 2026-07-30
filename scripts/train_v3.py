"""v3 training on the domain-matched synthetic pool.

Three model families, trained on the same data, decided by the test set:

  A. hybrid_reduced : target = z - DAK(Ppr,Tpr); inputs (Ppr, Tpr, comp feats)
                      -> physics backbone + corresponding-states scaling
  B. native_direct  : target = z;                inputs (T, P, x_1..x_21)
                      -> pure GERG emulator on absolute inputs
  C. chart          : target = z - DAK;          inputs (Ppr, Tpr)
                      -> the 2-input drop-in replacement for correlations

Run:  python -X utf8 scripts/train_v3.py [--models A,B,C] [--epochs 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from train_models_v2 import COMP_FEATS, MW_VEC, comp_features, pseudo_criticals  # noqa: E402
from zfactor.correlations import dak  # noqa: E402

MODELS = ROOT / "models" / "v3"
SEED = 20260730
X_COLS = [f"x_{c}" for c in COMPONENTS]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_pool(rng):
    """v3 pool + legacy pools, in absolute and reduced coordinates."""
    frames = []
    syn = ROOT / "data" / "synthetic"
    for pf, cf in [("points_v3.parquet", "compositions_v3.parquet"),
                   ("points.parquet", "compositions.parquet"),
                   ("points_h2.parquet", "compositions_h2.parquet")]:
        if not (syn / pf).exists():
            continue
        p = pd.read_parquet(syn / pf)
        c = pd.read_parquet(syn / cf)
        frames.append(p.merge(c[["comp_id", "split"] + X_COLS], on="comp_id"))
        log(f"  {pf}: {len(p):,} points")
    df = pd.concat(frames, ignore_index=True)
    del frames

    x = df[X_COLS].to_numpy(dtype=np.float64)
    df = df.drop(columns=X_COLS)

    # convention-augmented reduced coordinates (robustness to pc recipe)
    tpc, ppc, gamma = pseudo_criticals(x, rng)
    T = df.T_K.to_numpy()
    P = df.P_MPa.to_numpy()
    ppr = P / ppc
    tpr = T / tpc
    keep = (tpr >= 1.0) & (tpr <= 3.6) & (ppr > 0.01) & (ppr <= 45)
    log(f"  reduced-space keep: {keep.sum():,} of {len(df):,}")

    out = pd.DataFrame({
        "T_K": T[keep].astype(np.float32), "P_MPa": P[keep].astype(np.float32),
        "Ppr": ppr[keep].astype(np.float32), "Tpr": tpr[keep].astype(np.float32),
        "z": df.z.to_numpy()[keep].astype(np.float32),
        "split": df.split.to_numpy()[keep]})
    cfeat = comp_features(x[keep], gamma[keep])
    for i, n in enumerate(COMP_FEATS):
        out[n] = cfeat[:, i].astype(np.float32)
    for i, c in enumerate(X_COLS):
        out[c] = x[keep][:, i].astype(np.float32)

    log("computing DAK baseline (chunked)...")
    z_dak = np.empty(len(out))
    pa, ta = out.Ppr.to_numpy(np.float64), out.Tpr.to_numpy(np.float64)
    for i in range(0, len(out), 500_000):
        z_dak[i:i + 500_000] = dak(pa[i:i + 500_000], ta[i:i + 500_000])
    out["z_dak"] = z_dak.astype(np.float32)
    out["resid"] = (out.z - out.z_dak).astype(np.float32)
    ok = np.isfinite(out.resid) & (out.resid.abs() < 0.6)
    out = out[ok]
    log(f"training pool: {len(out):,} rows")
    return out


def fit_lgbm(Xtr, ytr, Xva, yva, path, leaves=511, n_est=5000):
    m = lgb.LGBMRegressor(
        n_estimators=n_est, learning_rate=0.05, num_leaves=leaves,
        min_child_samples=40, subsample=0.9, subsample_freq=1,
        colsample_bytree=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    m.booster_.save_model(str(path))
    return lambda X: m.predict(X)


def fit_mlp(Xtr, ytr, Xva, yva, path, epochs=10, batch=32768, width=512):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xt = torch.tensor(((Xtr - mu) / sd).astype(np.float32))
    yt = torch.tensor(ytr.astype(np.float32)).unsqueeze(1)
    Xv = torch.tensor(((Xva - mu) / sd).astype(np.float32)).to(dev)
    yv = torch.tensor(yva.astype(np.float32)).unsqueeze(1).to(dev)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], width), nn.SiLU(),
        nn.Linear(width, width), nn.SiLU(),
        nn.Linear(width, width), nn.SiLU(),
        nn.Linear(width, 256), nn.SiLU(),
        nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=batch, shuffle=True,
        generator=torch.Generator().manual_seed(SEED))
    lossf = nn.SmoothL1Loss(beta=0.01)
    best = float("inf")
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            lossf(net(xb), yb).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = torch.mean(torch.abs(net(Xv) - yv)).item()
        log(f"    epoch {ep+1}/{epochs} val_mae={vl:.3e}")
        if vl < best:
            best = vl
            torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd,
                        "in_dim": Xtr.shape[1], "width": width}, path)
    ck = torch.load(path, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    def pred(X):
        with torch.no_grad():
            Xs = torch.tensor(((X - mu) / sd).astype(np.float32)).to(dev)
            return np.concatenate([net(Xs[i:i + 262144]).cpu().numpy().ravel()
                                   for i in range(0, len(Xs), 262144)])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="A,B,C")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    want = set(args.models.split(","))

    MODELS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pool = build_pool(rng)
    tr, va = pool[pool.split == "train"], pool[pool.split == "val"]
    res = {}

    specs = {
        "A": ("hybrid_reduced", ["Ppr", "Tpr"] + COMP_FEATS, True),
        "B": ("native_direct", ["T_K", "P_MPa"] + X_COLS, False),
        "C": ("chart", ["Ppr", "Tpr"], True),
    }
    for key in ["A", "B", "C"]:
        if key not in want:
            continue
        name, feats, residual = specs[key]
        Xtr = tr[feats].to_numpy(np.float32)
        Xva = va[feats].to_numpy(np.float32)
        ytr = (tr.resid if residual else tr.z).to_numpy(np.float32)
        yva = (va.resid if residual else va.z).to_numpy(np.float32)
        base_va = va.z_dak.to_numpy() if residual else 0.0
        zva = va.z.to_numpy()
        log(f"[{name}] train {Xtr.shape}, val {Xva.shape}, residual={residual}")

        p_lgbm = fit_lgbm(Xtr, ytr, Xva, yva, MODELS / f"{name}_lgbm.txt")
        z_l = base_va + p_lgbm(Xva)
        m_l = np.mean(np.abs(z_l - zva) / zva) * 100
        log(f"  {name}/lgbm val MAPE(z) {m_l:.4f}%")

        p_mlp = fit_mlp(Xtr, ytr, Xva, yva, MODELS / f"{name}_mlp.pt",
                        epochs=args.epochs)
        z_m = base_va + p_mlp(Xva)
        m_m = np.mean(np.abs(z_m - zva) / zva) * 100
        log(f"  {name}/mlp  val MAPE(z) {m_m:.4f}%")

        z_e = 0.5 * (z_l + z_m)
        m_e = np.mean(np.abs(z_e - zva) / zva) * 100
        log(f"  {name}/ens  val MAPE(z) {m_e:.4f}%")

        res[name] = {"lgbm": round(float(m_l), 4), "mlp": round(float(m_m), 4),
                     "ens": round(float(m_e), 4), "features": feats,
                     "residual": residual}
        (MODELS / f"{name}_meta.json").write_text(
            json.dumps({"features": feats, "residual": residual}))

    (MODELS / "training_metrics_v3.json").write_text(json.dumps(res, indent=2))
    log("done -> models/v3/")


if __name__ == "__main__":
    main()
