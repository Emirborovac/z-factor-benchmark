"""v3 synthetic generation: coverage matched to the Test-B domain.

Adds what v1/v2 lacked:
  - binary mixtures across the full ratio range (CH4+N2/CO2/C2H6/H2/He/...)
  - pure components
  - wider state envelope: T 195-540 K, P to 215 MPa
  - keeps natural-gas + sour + H2-blend families

Writes data/synthetic/compositions_v3.parquet + points_v3.parquet.

Run:  python -X utf8 scripts/generate_v3.py [--n-comps 12000] [--pts 900]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC, load_nist_compositions  # noqa: E402
import generate_synthetic as G  # noqa: E402

SEED = 20260730
OUT = ROOT / "data" / "synthetic"

# widen the state envelope used by the generator
G.T_MIN_ABS, G.T_MAX = 195.0, 540.0
G.P_MAX = 215.0
G.PPR_MAX = 45.0
G.TPR_MARGIN = 1.02

BINARY_PARTNERS = ["nitrogen", "co2", "ethane", "hydrogen", "helium",
                   "propane", "n_butane", "argon", "co", "h2s", "oxygen"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-comps", type=int, default=12000)
    ap.add_argument("--pts", type=int, default=900)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    idx = {c: i for i, c in enumerate(COMPONENTS)}
    nc = len(COMPONENTS)
    rows, fams = [], []

    def add(x, fam):
        rows.append(G._norm(x))
        fams.append(fam)

    # 1. pure components (all 21)
    for c in COMPONENTS:
        x = np.zeros(nc)
        x[idx[c]] = 1.0
        add(x, "pure")

    # 2. binaries: methane + partner across full ratio, plus partner pairs
    n_bin = int(0.30 * args.n_comps)
    for _ in range(n_bin):
        p = BINARY_PARTNERS[rng.integers(len(BINARY_PARTNERS))]
        f = rng.uniform(0.01, 0.99)
        x = np.zeros(nc)
        x[idx["methane"]] = 1 - f
        x[idx[p]] = f
        add(x, "binary_ch4")
    for _ in range(int(0.06 * args.n_comps)):
        a, b = rng.choice(BINARY_PARTNERS, 2, replace=False)
        f = rng.uniform(0.05, 0.95)
        x = np.zeros(nc)
        x[idx[a]] = f
        x[idx[b]] = 1 - f
        add(x, "binary_other")

    # 3. ternaries (common lab synthetic gases)
    for _ in range(int(0.08 * args.n_comps)):
        parts = list(rng.choice(BINARY_PARTNERS, 2, replace=False))
        w = rng.dirichlet([2.0, 1.0, 1.0])
        x = np.zeros(nc)
        x[idx["methane"]] = w[0]
        x[idx[parts[0]]] = w[1]
        x[idx[parts[1]]] = w[2]
        add(x, "ternary")

    # 4. real NIST natural gases + perturbations
    real = load_nist_compositions().to_numpy(dtype=float)
    for r in real:
        add(r, "nist_real")
    for _ in range(int(0.20 * args.n_comps)):
        base = real[rng.integers(len(real))]
        add(base * rng.lognormal(0.0, 0.35, nc), "nist_perturbed")

    # 5. recipe families (sour / rich / H2-blend / N2-rich)
    def recipe(fam, spec):
        x = np.zeros(nc)
        for comp, (lo, hi) in spec.items():
            x[idx[comp]] = rng.uniform(lo, hi)
        x[idx["methane"]] += max(0.0, 1.0 - x.sum())
        add(x, fam)

    while len(rows) < args.n_comps:
        k = len(rows) % 4
        if k == 0:
            recipe("sour", {"h2s": (0, 0.30), "co2": (0, 0.30),
                            "nitrogen": (0, 0.05), "ethane": (0, 0.08)})
        elif k == 1:
            recipe("rich", {"ethane": (0.03, 0.18), "propane": (0.01, 0.12),
                            "n_butane": (0, 0.05), "isobutane": (0, 0.03),
                            "n_pentane": (0, 0.02), "hexane": (0, 0.01),
                            "co2": (0, 0.05), "nitrogen": (0, 0.05)})
        elif k == 2:
            recipe("h2_blend", {"hydrogen": (0, 0.40), "nitrogen": (0, 0.08),
                                "co2": (0, 0.06), "ethane": (0, 0.10)})
        else:
            recipe("n2_rich", {"nitrogen": (0.05, 0.50), "co2": (0, 0.10),
                               "ethane": (0, 0.05), "helium": (0, 0.02)})

    comps = pd.DataFrame(rows, columns=[f"x_{c}" for c in COMPONENTS])
    comps.insert(0, "comp_id", np.arange(200000, 200000 + len(comps)))
    comps.insert(1, "family", fams)
    xm = comps[[f"x_{c}" for c in COMPONENTS]].to_numpy()
    comps["tpc_kay"] = xm @ TC_VEC
    comps["ppc_kay"] = xm @ PC_VEC
    comps["split"] = np.where(rng.random(len(comps)) < 0.10, "val", "train")
    print(comps.family.value_counts().to_string())

    print(f"\ngenerating up to {len(comps) * args.pts:,} GERG points "
          f"(T {G.T_MIN_ABS}-{G.T_MAX} K, P to {G.P_MAX} MPa)...")
    pts, rej, dt = G.generate_points(comps, args.pts, rng)
    print(f"kept {len(pts):,} in {dt/60:.1f} min; rejected {rej}")
    print(G.detail_crosscheck(comps, pts, rng, n=10000))

    comps.to_parquet(OUT / "compositions_v3.parquet", index=False)
    pts.to_parquet(OUT / "points_v3.parquet", index=False)
    print("wrote compositions_v3.parquet, points_v3.parquet")


if __name__ == "__main__":
    main()
