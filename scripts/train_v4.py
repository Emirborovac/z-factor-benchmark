"""v4: domain-focused training.

v3's lesson: a broader pool (binaries at all ratios, pures, T to 540 K,
Ppr to 45) diluted capacity and made natural-gas accuracy WORSE (2.14% vs
v2's 1.85%). v4 keeps the extra composition diversity but restricts the
STATE ENVELOPE to the application domain, and weights natural-gas-like
compositions up.

Domain (declared a priori from the application, not tuned on test labels):
    Tpr in [1.00, 3.05], T in [200, 520] K, Ppr in [0.02, 16], CH4 >= 30%

Models: LightGBM on the DAK residual (hybrid) and on z directly (native).

Run:  python -X utf8 scripts/train_v4.py
"""
from __future__ import annotations

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

MODELS = ROOT / "models" / "v4"
SEED = 20260731
X_COLS = [f"x_{c}" for c in COMPONENTS]

TPR_LO, TPR_HI = 1.00, 3.05
T_LO, T_HI = 200.0, 520.0
PPR_LO, PPR_HI = 0.02, 16.0
CH4_MIN = 0.30


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build():
    rng = np.random.default_rng(SEED)
    syn = ROOT / "data" / "synthetic"
    frames = []
    for pf, cf in [("points_v3.parquet", "compositions_v3.parquet"),
                   ("points.parquet", "compositions.parquet"),
                   ("points_h2.parquet", "compositions_h2.parquet")]:
        if (syn / pf).exists():
            p = pd.read_parquet(syn / pf)
            c = pd.read_parquet(syn / cf)
            frames.append(p.merge(c[["comp_id", "split"] + X_COLS], on="comp_id"))
    df = pd.concat(frames, ignore_index=True)
    del frames
    log(f"raw pool: {len(df):,}")

    x = df[X_COLS].to_numpy(np.float64)
    df = df.drop(columns=X_COLS)
    T = df.T_K.to_numpy()
    P = df.P_MPa.to_numpy()

    # composition filter first (cheap)
    keep_c = x[:, COMPONENTS.index("methane")] >= CH4_MIN
    x, T, P = x[keep_c], T[keep_c], P[keep_c]
    df = df[keep_c]
    log(f"after CH4>={CH4_MIN}: {len(df):,}")

    tpc, ppc, gamma = pseudo_criticals(x, rng)
    ppr, tpr = P / ppc, T / tpc
    keep = ((tpr >= TPR_LO) & (tpr <= TPR_HI) & (T >= T_LO) & (T <= T_HI)
            & (ppr >= PPR_LO) & (ppr <= PPR_HI))
    x, gamma, ppr, tpr = x[keep], gamma[keep], ppr[keep], tpr[keep]
    df = df[keep]
    T, P = T[keep], P[keep]
    log(f"after state-envelope filter: {len(df):,}")

    out = pd.DataFrame({
        "Ppr": ppr.astype(np.float32), "Tpr": tpr.astype(np.float32),
        "T_K": T.astype(np.float32), "P_MPa": P.astype(np.float32),
        "z": df.z.to_numpy(np.float32), "split": df.split.to_numpy()})
    cf = comp_features(x, gamma)
    for i, n in enumerate(COMP_FEATS):
        out[n] = cf[:, i].astype(np.float32)
    for i, c in enumerate(X_COLS):
        out[c] = x[:, i].astype(np.float32)

    log("DAK baseline (chunked)...")
    zd = np.empty(len(out))
    pa, ta = out.Ppr.to_numpy(np.float64), out.Tpr.to_numpy(np.float64)
    for i in range(0, len(out), 500_000):
        zd[i:i + 500_000] = dak(pa[i:i + 500_000], ta[i:i + 500_000])
    out["z_dak"] = zd.astype(np.float32)
    out["resid"] = (out.z - out.z_dak).astype(np.float32)
    out = out[np.isfinite(out.resid) & (out.resid.abs() < 0.6)]
    log(f"training pool: {len(out):,}")
    return out


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    pool = build()
    tr, va = pool[pool.split == "train"], pool[pool.split == "val"]

    specs = [("hybrid", ["Ppr", "Tpr"] + COMP_FEATS, True),
             ("native", ["T_K", "P_MPa"] + X_COLS, False)]
    res = {}
    for name, feats, residual in specs:
        Xtr = tr[feats].to_numpy(np.float32)
        Xva = va[feats].to_numpy(np.float32)
        ytr = (tr.resid if residual else tr.z).to_numpy(np.float32)
        yva = (va.resid if residual else va.z).to_numpy(np.float32)
        log(f"[{name}] train {Xtr.shape}")
        m = lgb.LGBMRegressor(
            n_estimators=6000, learning_rate=0.04, num_leaves=511,
            min_child_samples=30, subsample=0.9, subsample_freq=1,
            colsample_bytree=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        m.booster_.save_model(str(MODELS / f"{name}_lgbm.txt"))
        base = va.z_dak.to_numpy() if residual else 0.0
        z = base + m.predict(Xva)
        mp = float(np.mean(np.abs(z - va.z.to_numpy()) / va.z.to_numpy()) * 100)
        log(f"  {name} val MAPE(z) {mp:.4f}%  (best_iter {m.best_iteration_})")
        res[name] = mp
        (MODELS / f"{name}_meta.json").write_text(
            json.dumps({"features": feats, "residual": residual}))

    (MODELS / "training_metrics_v4.json").write_text(json.dumps(res, indent=2))
    log("done -> models/v4/")


if __name__ == "__main__":
    main()
