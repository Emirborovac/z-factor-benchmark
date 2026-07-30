"""v2 training: physics-informed hybrid models designed to beat DAK/HY on
real measurements.

Design (from expert practice + our diagnostics):
1. RESIDUAL TARGET: models learn r = z - DAK(Ppr,Tpr). DAK supplies physics
   (ideal-gas limit, sane extrapolation); ML only corrects its weaknesses.
2. TRAIN IN REDUCED SPACE ON REAL-GAS PHYSICS: GERG-2008 synthetic points are
   mapped to (Ppr,Tpr) via pseudo-critical rules. This teaches the model the
   *real-gas scatter* around the corresponding-states surface instead of the
   idealized chart alone; chart data is kept as an additional source.
3. CONVENTION AUGMENTATION: each synthetic point is mapped with Kay, Sutton,
   and Standing pseudo-criticals (randomly chosen per point) because the test
   compilation's sources used different conventions. The model learns a
   surface robust to that irreducible input noise.
4. EXTENDED RANGE: synthetic reduced coords reach Ppr ~30, so the test set's
   Ppr>15 region is interpolation for v2 models (fixes tree collapse).
5. Tracks: chart = f(Ppr,Tpr); composition = f(Ppr,Tpr, composition features).
   Both: LightGBM + MLP + their ensemble.

Run:  python -X utf8 scripts/train_models_v2.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from zfactor.correlations import dak  # noqa: E402

MODELS = ROOT / "models" / "v2"
SEED = 20260728
X_COLS = [f"x_{c}" for c in COMPONENTS]
MW_VEC = np.array([16.043, 28.014, 44.01, 30.07, 44.097, 58.123, 58.123,
                   72.15, 72.15, 86.177, 100.204, 114.231, 128.258, 142.285,
                   2.016, 31.999, 28.01, 18.015, 34.081, 4.003, 39.948])

COMP_FEATS = ["gamma", "x_co2", "x_h2s", "x_n2", "x_c1", "x_c2", "x_c3",
              "x_c4p", "x_inert"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pseudo_criticals(x: np.ndarray, rng: np.random.Generator):
    """Per-row pseudo-critical (Tpc K, Ppc MPa) under a randomly assigned
    convention: 0=Kay, 1=Sutton, 2=Standing (gravity-based)."""
    gamma = (x @ MW_VEC) / 28.9647
    tpc_kay = x @ TC_VEC
    ppc_kay = x @ PC_VEC
    tpc_sut = (169.2 + 349.5 * gamma - 74 * gamma**2) / 1.8
    ppc_sut = (756.8 - 131.0 * gamma - 3.6 * gamma**2) * 0.00689476
    tpc_std = (168 + 325 * gamma - 12.5 * gamma**2) / 1.8
    ppc_std = (677 + 15 * gamma - 37.5 * gamma**2) * 0.00689476
    conv = rng.integers(0, 3, len(x))
    tpc = np.select([conv == 0, conv == 1, conv == 2],
                    [tpc_kay, tpc_sut, tpc_std])
    ppc = np.select([conv == 0, conv == 1, conv == 2],
                    [ppc_kay, ppc_sut, ppc_std])
    return tpc, ppc, gamma


def comp_features(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    i = {c: k for k, c in enumerate(COMPONENTS)}
    x_c4p = x[:, [i["isobutane"], i["n_butane"], i["isopentane"],
                  i["n_pentane"], i["hexane"], i["heptane"], i["octane"],
                  i["nonane"], i["decane"]]].sum(1)
    x_inert = x[:, [i["helium"], i["argon"], i["hydrogen"], i["oxygen"],
                    i["co"], i["water"]]].sum(1)
    return np.column_stack([
        gamma, x[:, i["co2"]], x[:, i["h2s"]], x[:, i["nitrogen"]],
        x[:, i["methane"]], x[:, i["ethane"]], x[:, i["propane"]],
        x_c4p, x_inert])


# --------------------------------------------------------------------------- #
def build_training_data():
    """Reduced-space dataset: SK chart + convention-augmented GERG synthetic."""
    rng = np.random.default_rng(SEED)

    # --- synthetic tier -> reduced space (base pool + H2-blend extension)
    pts = pd.read_parquet(ROOT / "data" / "synthetic" / "points.parquet")
    comps = pd.read_parquet(ROOT / "data" / "synthetic" / "compositions.parquet")
    h2p = ROOT / "data" / "synthetic" / "points_h2.parquet"
    if h2p.exists():
        pts = pd.concat([pts, pd.read_parquet(h2p)], ignore_index=True)
        comps = pd.concat(
            [comps, pd.read_parquet(ROOT / "data" / "synthetic"
                                    / "compositions_h2.parquet")],
            ignore_index=True)
        log("H2-blend extension included")
    df = pts.merge(comps[["comp_id", "split"] + X_COLS], on="comp_id")
    del pts
    x = df[X_COLS].to_numpy()
    df = df.drop(columns=X_COLS)
    tpc, ppc, gamma = pseudo_criticals(x, rng)
    df["Ppr"] = df.P_MPa / ppc
    df["Tpr"] = df.T_K / tpc
    keep = ((df.Tpr >= 1.0) & (df.Tpr <= 3.2)
            & (df.Ppr > 0.02) & (df.Ppr <= 32)).to_numpy()
    df, x, gamma = df[keep], x[keep], gamma[keep]
    log(f"synthetic reduced-space rows kept: {len(df):,}")

    cf = comp_features(x, gamma)
    del x
    syn = pd.DataFrame({
        "Ppr": df.Ppr.to_numpy(np.float32), "Tpr": df.Tpr.to_numpy(np.float32),
        "z": df.z.to_numpy(np.float32), "split": df.split.to_numpy(),
        "source": "gerg"})
    for k, name in enumerate(COMP_FEATS):
        syn[name] = cf[:, k].astype(np.float32)
    del cf, df

    # --- SK chart (canonical corresponding-states surface, avg natural gas)
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    sk = m[(m.tier == "chart_digitized") & (m.quality_flag == "ok")]
    chart = pd.DataFrame({
        "Ppr": sk.Ppr.to_numpy(), "Tpr": sk.Tpr.to_numpy(),
        "z": sk.z.to_numpy(), "split": sk.split.to_numpy(),
        "source": "chart"})
    # chart has no composition: use average-gas placeholders (gamma ~0.65 lean)
    defaults = dict(zip(COMP_FEATS, [0.65, 0.01, 0.0, 0.01, 0.92, 0.04,
                                     0.01, 0.005, 0.0]))
    for k, v in defaults.items():
        chart[k] = v

    full = pd.concat([syn, chart], ignore_index=True)

    # residual target vs DAK (chunked to bound memory)
    log("computing DAK baseline for all rows (chunked)...")
    ppr_all = full.Ppr.to_numpy(np.float64)
    tpr_all = full.Tpr.to_numpy(np.float64)
    zdak = np.empty(len(full))
    step = 500_000
    for i in range(0, len(full), step):
        zdak[i:i + step] = dak(ppr_all[i:i + step], tpr_all[i:i + step])
    full["z_dak"] = zdak.astype(np.float32)
    full["resid"] = full.z - zdak
    ok = np.isfinite(full.resid) & (np.abs(full.resid) < 0.5)
    full = full[ok]
    log(f"training pool: {len(full):,} rows "
        f"({(full.source=='chart').sum():,} chart + {(full.source=='gerg').sum():,} gerg)")
    return full


# --------------------------------------------------------------------------- #
def train_lgbm(Xtr, ytr, Xva, yva, name):
    model = lgb.LGBMRegressor(
        n_estimators=4000, learning_rate=0.04, num_leaves=255,
        min_child_samples=40, subsample=0.9, subsample_freq=1,
        colsample_bytree=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    model.booster_.save_model(str(MODELS / f"{name}_lgbm.txt"))
    return lambda X: model.predict(X)


def train_mlp(Xtr, ytr, Xva, yva, name, epochs=24, batch=32768):
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
        nn.Linear(Xtr.shape[1], 512), nn.SiLU(),
        nn.Linear(512, 512), nn.SiLU(),
        nn.Linear(512, 512), nn.SiLU(),
        nn.Linear(512, 256), nn.SiLU(),
        nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=batch, shuffle=True,
        generator=torch.Generator().manual_seed(SEED))
    lossf = nn.SmoothL1Loss(beta=0.01)          # robust to residual outliers
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
            vl = torch.mean(torch.abs(net(Xv) - yv)).item()
        if ep % 4 == 3 or ep == epochs - 1:
            log(f"  {name}_mlp epoch {ep+1}/{epochs} val_mae={vl:.3e}")
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
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["chart", "composition", "all"],
                    default="all")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    global train_mlp
    _orig_mlp = train_mlp
    train_mlp = lambda *a, **k: _orig_mlp(*a, epochs=args.epochs, **k)

    MODELS.mkdir(parents=True, exist_ok=True)
    full = build_training_data()
    tr = full[full.split == "train"]
    va = full[full.split == "val"]

    tracks = [("chart", ["Ppr", "Tpr"]),
              ("composition", ["Ppr", "Tpr"] + COMP_FEATS)]
    if args.track != "all":
        tracks = [t for t in tracks if t[0] == args.track]

    results = {}
    for track, feats in tracks:
        Xtr = tr[feats].to_numpy(dtype=np.float64)
        Xva = va[feats].to_numpy(dtype=np.float64)
        ytr, yva = tr.resid.to_numpy(), va.resid.to_numpy()
        zva, zdak_va = va.z.to_numpy(), va.z_dak.to_numpy()
        log(f"track={track}: train {Xtr.shape}, val {Xva.shape}")

        preds = {}
        for algo, trainer in [("lgbm", train_lgbm), ("mlp", train_mlp)]:
            t0 = time.perf_counter()
            fn = trainer(Xtr, ytr, Xva, yva, track)
            pv = zdak_va + fn(Xva)
            mape = np.mean(np.abs(pv - zva) / np.abs(zva)) * 100
            results[f"{track}_{algo}"] = {
                "val_MAPE_%": round(float(mape), 4),
                "fit_s": round(time.perf_counter() - t0, 1)}
            log(f"  {track}/{algo}: val MAPE(z) {mape:.4f}%")
            preds[algo] = pv
        ens = 0.5 * (preds["lgbm"] + preds["mlp"])
        mape = np.mean(np.abs(ens - zva) / np.abs(zva)) * 100
        results[f"{track}_ens"] = {"val_MAPE_%": round(float(mape), 4)}
        log(f"  {track}/ensemble: val MAPE(z) {mape:.4f}%")
        (MODELS / f"{track}_features.json").write_text(json.dumps(feats))

    (MODELS / "training_metrics_v2.json").write_text(json.dumps(results, indent=2))
    log("done; models in models/v2/")


if __name__ == "__main__":
    main()
