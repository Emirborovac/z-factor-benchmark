"""Regional agreement measured against the TEACHER equation (AGA8-DETAIL).

The surrogate is distilled from AGA8-DETAIL, so a surrogate-fidelity statement
must be made against AGA8-DETAIL, not GERG-2008. This recomputes the regional
table on that basis and prints values ready to paste into the figure.

Run:  python -X utf8 scripts/eval_regional_detail.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyaga8
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from train_models_v2 import MW_VEC  # noqa: E402
from train_nn import Net, build_features  # noqa: E402
from convention_recovery import wichert_aziz, PSI  # noqa: E402
from generate_synthetic import PYAGA8_ATTR  # noqa: E402
import zfactor.correlations as C  # noqa: E402

X_COLS = [f"x_{c}" for c in COMPONENTS]
T_GRID = np.array([283.15, 298.15, 323.15, 353.15, 393.15])
P_GRID = np.array([1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0, 40.0])


def eos_batch(cls, X, T, P):
    eos = cls()
    out = np.full(len(T), np.nan)
    last = None
    for i in range(len(T)):
        xr = X[i]
        if last is None or not np.array_equal(xr, last):
            c = pyaga8.Composition()
            for comp, attr in PYAGA8_ATTR.items():
                setattr(c, attr, float(xr[COMPONENTS.index(comp)]))
            try:
                eos.set_composition(c)
            except Exception:
                last = None
                continue
            last = xr
        try:
            eos.temperature = float(T[i])
            eos.pressure = float(P[i]) * 1000.0
            if cls is pyaga8.Gerg2008:
                eos.calc_density(0)
            else:
                eos.calc_density()
            eos.calc_properties()
            if np.isfinite(eos.z) and 0.1 < eos.z < 3.5:
                out[i] = eos.z
        except Exception:
            pass
    return out


def main():
    b = pd.read_parquet(ROOT / "data" / "processed"
                        / "regional_compositions.parquet")
    x = b[X_COLS].to_numpy()
    TT, PP = np.meshgrid(T_GRID, P_GRID, indexing="ij")
    grid = np.column_stack([TT.ravel(), PP.ravel()])
    n_pt = len(grid)

    X = np.repeat(x, n_pt, axis=0)
    T = np.tile(grid[:, 0], len(b))
    P = np.tile(grid[:, 1], len(b))
    region = np.repeat(b.region.to_numpy(), n_pt)

    gamma = (X @ MW_VEC) / 28.9647
    tpc0, ppc0 = X @ TC_VEC, X @ PC_VEC
    tpcw, ppcw = wichert_aziz(tpc0, ppc0, X[:, COMPONENTS.index("co2")],
                              X[:, COMPONENTS.index("h2s")])
    ppr_k, tpr_k = P / ppc0, T / tpc0

    ck = torch.load(ROOT / "models" / "nn" / "eos_nn_detail.pt",
                    weights_only=False)
    net = Net(ck["d_in"], ck["width"], ck["blocks"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    F = build_features(T, P, ppr_k, tpr_k, X, gamma)
    with torch.no_grad():
        r = np.concatenate([
            net(torch.tensor((F[i:i + 200000] - ck["mu"]) / ck["sd"]))
            .squeeze(1).numpy() for i in range(0, len(F), 200000)])
    nncf = C.dak(ppr_k, tpr_k) + r

    print(f"evaluating {len(b)} gases x {n_pt} states = {len(T):,} points ...")
    ref = eos_batch(pyaga8.Detail, X, T, P)          # the TEACHER
    dak = C.dak(P / ppcw, T / tpcw)

    ok = np.isfinite(ref)
    rows = {}
    for reg in np.unique(region):
        m = ok & (region == reg)
        rows[reg] = (
            float(np.mean(np.abs(nncf[m] - ref[m]) / ref[m]) * 100),
            float(np.mean(np.abs(dak[m] - ref[m]) / ref[m]) * 100),
            int((b.region == reg).sum()),
        )

    order = sorted(rows, key=lambda k: rows[k][0])
    print("\nagreement with AGA8-DETAIL (the teaching equation):")
    print(f"{'region':22s} {'NNCF':>9s} {'DAK':>9s} {'n':>6s}")
    for k in order:
        a, d, n = rows[k]
        print(f"{k:22s} {a:9.4f} {d:9.4f} {n:6d}")
    tot = ok
    print(f"\noverall  NNCF {np.mean(np.abs(nncf[tot]-ref[tot])/ref[tot])*100:.4f} %"
          f"   DAK {np.mean(np.abs(dak[tot]-ref[tot])/ref[tot])*100:.4f} %")

    out = {k: [round(rows[k][0], 4), round(rows[k][1], 4), rows[k][2]]
           for k in order}
    (ROOT / "reports" / "regional_vs_detail.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote reports/regional_vs_detail.json")
    print("\npaste-ready dict for make_figures.fig5_regional:")
    print("    vals = {" + ", ".join(
        f'"{k.split(" (")[0]}": ({rows[k][0]:.3f}, {rows[k][1]:.3f}, '
        f'{rows[k][2]})' for k in order) + "}")


if __name__ == "__main__":
    main()
