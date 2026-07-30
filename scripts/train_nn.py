"""Neural EOS surrogate - the architecturally correct model.

Fixes three defects of the tree-based attempts:
  1. SMOOTHNESS  : z(T,P,x) is a smooth surface; trees are piecewise-constant.
                   A residual MLP approximates it natively.
  2. OBJECTIVE   : we are scored on MAPE, so train on RELATIVE error, not
                   absolute. Low-z near-critical points stop being ignored.
  3. NO BOTTLENECK: inputs carry the full state - absolute (T,P), reduced
                   (Ppr,Tpr), and all 21 mole fractions - nothing compressed.

Speed: the whole training tensor lives on the GPU, so an epoch is seconds and
hundreds of epochs are affordable (previous runs died at 8 undertrained ones).

Target: z - DAK(Ppr,Tpr) (physics backbone; the net learns only the departure).

Run:  python -X utf8 scripts/train_nn.py [--rows 4000000] [--epochs 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from train_models_v2 import MW_VEC, pseudo_criticals  # noqa: E402
from zfactor.correlations import dak  # noqa: E402

MODELS = ROOT / "models" / "nn"
SEED = 20260731
X_COLS = [f"x_{c}" for c in COMPONENTS]

TPR_LO, TPR_HI = 1.00, 3.05
T_LO, T_HI = 200.0, 520.0
PPR_LO, PPR_HI = 0.02, 16.0
CH4_MIN = 0.30


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class ResBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(w, w), nn.SiLU(),
                               nn.Linear(w, w))
        self.act = nn.SiLU()

    def forward(self, h):
        return self.act(h + self.f(h))


class Net(nn.Module):
    def __init__(self, d_in, w=512, blocks=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, w), nn.SiLU())
        self.body = nn.Sequential(*[ResBlock(w) for _ in range(blocks)])
        self.out = nn.Linear(w, 1)

    def forward(self, x):
        return self.out(self.body(self.inp(x)))


def build_features(T, P, ppr, tpr, x, gamma):
    """Absolute + reduced + full composition + a few physical transforms."""
    return np.column_stack([
        T, P, ppr, tpr,
        1.0 / tpr, np.log(np.clip(ppr, 1e-6, None)),
        ppr / tpr, ppr ** 2 / tpr ** 3,
        gamma, x,
    ]).astype(np.float32)


def load_pool(max_rows: int, teacher: str = "gerg"):
    """teacher: 'gerg' (GERG-2008 labels) or 'detail' (AGA8-DETAIL labels).

    Both use IDENTICAL (composition, T, P) states — only the reference
    equation that produced z differs, so the comparison is controlled.
    """
    rng = np.random.default_rng(SEED)
    syn = ROOT / "data" / "synthetic"
    suffix = "_detail" if teacher == "detail" else ""
    frames = []
    for pf, cf in [("points_v3.parquet", "compositions_v3.parquet"),
                   ("points.parquet", "compositions.parquet"),
                   ("points_h2.parquet", "compositions_h2.parquet")]:
        pf = pf.replace(".parquet", f"{suffix}.parquet")
        if (syn / pf).exists():
            p = pd.read_parquet(syn / pf)
            c = pd.read_parquet(syn / cf)
            frames.append(p.merge(c[["comp_id", "split"] + X_COLS], on="comp_id"))
    if not frames:
        raise SystemExit(f"no pool files for teacher={teacher}")
    df = pd.concat(frames, ignore_index=True)
    del frames
    x = df[X_COLS].to_numpy(np.float64)
    keep = x[:, COMPONENTS.index("methane")] >= CH4_MIN
    df, x = df[keep], x[keep]
    T, P = df.T_K.to_numpy(), df.P_MPa.to_numpy()
    tpc, ppc, gamma = pseudo_criticals(x, rng)
    ppr, tpr = P / ppc, T / tpc
    keep = ((tpr >= TPR_LO) & (tpr <= TPR_HI) & (T >= T_LO) & (T <= T_HI)
            & (ppr >= PPR_LO) & (ppr <= PPR_HI))
    df, x, T, P = df[keep], x[keep], T[keep], P[keep]
    ppr, tpr, gamma = ppr[keep], tpr[keep], gamma[keep]
    log(f"pool after domain filter: {len(df):,}")

    if len(df) > max_rows:
        sel = rng.choice(len(df), max_rows, replace=False)
        df, x, T, P = df.iloc[sel], x[sel], T[sel], P[sel]
        ppr, tpr, gamma = ppr[sel], tpr[sel], gamma[sel]
        log(f"subsampled to {len(df):,} (smooth surface needs coverage, not volume)")

    log("DAK baseline...")
    zd = np.empty(len(df))
    for i in range(0, len(df), 500_000):
        zd[i:i + 500_000] = dak(ppr[i:i + 500_000], tpr[i:i + 500_000])
    z = df.z.to_numpy()
    ok = np.isfinite(zd) & np.isfinite(z) & (np.abs(z - zd) < 0.6) & (z > 0.15)
    F = build_features(T[ok], P[ok], ppr[ok], tpr[ok], x[ok], gamma[ok])
    return F, z[ok].astype(np.float32), zd[ok].astype(np.float32), \
        df.split.to_numpy()[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4_000_000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--teacher", choices=["gerg", "detail"],
                    default="gerg",
                    help="reference EOS that generated the training labels")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    F, z, zd, split = load_pool(args.rows, args.teacher)
    tag = "" if args.teacher == "gerg" else f"_{args.teacher}"
    tr, va = split == "train", split == "val"
    log(f"train {tr.sum():,} | val {va.sum():,} | features {F.shape[1]}")

    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
    Ftr = torch.tensor((F[tr] - mu) / sd, device=dev)
    Fva = torch.tensor((F[va] - mu) / sd, device=dev)
    ztr = torch.tensor(z[tr], device=dev)
    zva = torch.tensor(z[va], device=dev)
    dtr = torch.tensor(zd[tr], device=dev)
    dva = torch.tensor(zd[va], device=dev)
    rtr = (ztr - dtr).unsqueeze(1)
    log(f"tensors resident on {dev}: {Ftr.element_size()*Ftr.nelement()/1e9:.2f} GB")

    net = Net(F.shape[1], args.width, args.blocks).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=args.epochs, pct_start=0.15)
    log(f"net: {n_par:,} params, {args.blocks} residual blocks x {args.width}")

    n = Ftr.shape[0]
    best = float("inf")
    g = torch.Generator(device=dev).manual_seed(SEED)
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(n, device=dev, generator=g)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            pred = net(Ftr[idx])
            # RELATIVE error objective - matches the MAPE we are scored on
            loss = (torch.abs(pred - rtr[idx]) / ztr[idx].unsqueeze(1)).mean()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            # chunked: a single 290k-row forward pass thrashes VRAM on 4 GB cards
            outs = [net(Fva[i:i + 65536]).squeeze(1)
                    for i in range(0, Fva.shape[0], 65536)]
            zp = dva + torch.cat(outs)
            vmape = (torch.abs(zp - zva) / zva).mean().item() * 100
        if vmape < best:
            best = vmape
            torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd,
                        "d_in": F.shape[1], "width": args.width,
                        "blocks": args.blocks},
                       MODELS / f"eos_nn{tag}.pt")
        if ep < 3 or ep % 5 == 0 or ep == args.epochs - 1:
            log(f"  epoch {ep+1}/{args.epochs} val MAPE {vmape:.4f}% "
                f"(best {best:.4f}%) [{time.perf_counter()-t0:.0f}s]")

    (MODELS / f"meta{tag}.json").write_text(json.dumps(
        {"val_MAPE": best, "params": n_par, "features": "T,P,Ppr,Tpr,1/Tpr,"
         "logPpr,Ppr/Tpr,Ppr2/Tpr3,gamma,x1..x21", "residual_vs": "DAK"}))
    log(f"done: best val MAPE {best:.4f}% -> models/nn/eos_nn.pt")


if __name__ == "__main__":
    main()
