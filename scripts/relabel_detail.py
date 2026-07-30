"""Relabel the synthetic training pool with AGA8-DETAIL.

Controlled teacher experiment: the (composition, T, P) states are IDENTICAL to
the GERG-2008 pool; only the reference equation that produces z changes. Any
difference in downstream model accuracy is therefore attributable to the
teacher alone.

Writes points_v3_detail.parquet, points_detail.parquet, points_h2_detail.parquet
alongside the originals.

Run:  python -X utf8 scripts/relabel_detail.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyaga8

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS  # noqa: E402
from generate_synthetic import PYAGA8_ATTR  # noqa: E402

SYN = ROOT / "data" / "synthetic"
X_COLS = [f"x_{c}" for c in COMPONENTS]
Z_LO, Z_HI = 0.15, 3.0

PAIRS = [("points_v3.parquet", "compositions_v3.parquet"),
         ("points.parquet", "compositions.parquet"),
         ("points_h2.parquet", "compositions_h2.parquet")]


def relabel(points_file: str, comps_file: str) -> None:
    out = SYN / points_file.replace(".parquet", "_detail.parquet")
    pts = pd.read_parquet(SYN / points_file)
    comps = pd.read_parquet(SYN / comps_file).set_index("comp_id")
    xmat = comps[X_COLS].to_numpy()
    cid_pos = {c: i for i, c in enumerate(comps.index)}
    attrs = [PYAGA8_ATTR[c] for c in COMPONENTS]

    det = pyaga8.Detail()
    z = np.full(len(pts), np.nan)
    cid = pts.comp_id.to_numpy()
    T = pts.T_K.to_numpy()
    P = pts.P_MPa.to_numpy()

    t0 = time.perf_counter()
    last = -1
    for i in range(len(pts)):
        c_ = cid[i]
        if c_ != last:
            comp = pyaga8.Composition()
            xr = xmat[cid_pos[c_]]
            for attr, v in zip(attrs, xr):
                setattr(comp, attr, float(v))
            try:
                det.set_composition(comp)
            except Exception:
                last = -1
                continue
            last = c_
        try:
            det.temperature = T[i]
            det.pressure = P[i] * 1000.0
            det.calc_density()
            det.calc_properties()
            v = det.z
            if np.isfinite(v) and Z_LO < v < Z_HI and det.dp_dd > 0:
                z[i] = v
        except Exception:
            pass
        if i and i % 2_000_000 == 0:
            el = time.perf_counter() - t0
            print(f"    {i:,}/{len(pts):,}  {i/el:,.0f} pts/s", flush=True)

    keep = np.isfinite(z)
    res = pts.loc[keep, ["comp_id", "T_K", "P_MPa"]].copy()
    res["z"] = z[keep]
    res.to_parquet(out, index=False)
    dt = time.perf_counter() - t0
    print(f"  {points_file}: {keep.sum():,}/{len(pts):,} kept "
          f"({100*keep.mean():.1f} %) in {dt/60:.1f} min -> {out.name}")


def main():
    for pf, cf in PAIRS:
        if not (SYN / pf).exists():
            print(f"  skip {pf} (missing)")
            continue
        print(f"relabelling {pf} with AGA8-DETAIL ...")
        relabel(pf, cf)
    print("done")


if __name__ == "__main__":
    main()
