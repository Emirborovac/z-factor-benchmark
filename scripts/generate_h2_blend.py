"""Extend the synthetic tier with hydrogen-blended natural gas compositions
(H2 0-35%), the energy-transition case absent from the original pool.

Writes data/synthetic/compositions_h2.parquet + points_h2.parquet
(same schema as the originals; comp_id offset by 100000).

Run:  python -X utf8 scripts/generate_h2_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from generate_synthetic import (  # noqa: E402
    OUT, _norm, generate_points, detail_crosscheck)

SEED = 20260729
N_COMPS = 2500
PTS = 400


def main():
    rng = np.random.default_rng(SEED)
    idx = {c: i for i, c in enumerate(COMPONENTS)}
    ncomp = len(COMPONENTS)

    rows = []
    for _ in range(N_COMPS):
        x = np.zeros(ncomp)
        x[idx["hydrogen"]] = rng.uniform(0.0, 0.35)
        x[idx["nitrogen"]] = rng.uniform(0.0, 0.10)
        x[idx["co2"]] = rng.uniform(0.0, 0.08)
        x[idx["ethane"]] = rng.uniform(0.0, 0.10)
        x[idx["propane"]] = rng.uniform(0.0, 0.04)
        x[idx["n_butane"]] = rng.uniform(0.0, 0.01)
        x[idx["methane"]] += max(0.0, 1.0 - x.sum())
        rows.append(_norm(x))

    comps = pd.DataFrame(rows, columns=[f"x_{c}" for c in COMPONENTS])
    comps.insert(0, "comp_id", np.arange(100000, 100000 + len(comps)))
    comps.insert(1, "family", "h2_blend")
    comps["tpc_kay"] = comps[[f"x_{c}" for c in COMPONENTS]].to_numpy() @ TC_VEC
    comps["ppc_kay"] = comps[[f"x_{c}" for c in COMPONENTS]].to_numpy() @ PC_VEC
    comps["split"] = np.where(rng.random(len(comps)) < 0.10, "val", "train")

    print(f"generating {N_COMPS * PTS:,} H2-blend GERG points...")
    pts, rej, dt = generate_points(comps, PTS, rng)
    print(f"kept {len(pts):,} in {dt/60:.1f} min; rejected {rej}")
    print(detail_crosscheck(comps, pts, rng, n=5000))

    comps.to_parquet(OUT / "compositions_h2.parquet", index=False)
    pts.to_parquet(OUT / "points_h2.parquet", index=False)
    print("wrote compositions_h2.parquet, points_h2.parquet")


if __name__ == "__main__":
    main()
