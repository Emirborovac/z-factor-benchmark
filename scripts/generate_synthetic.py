"""Synthetic training-set generator (composition-domain model, tier: synthetic).

Strategy
--------
1. Composition pool (~10k):
   - 200 real NIST natural-gas compositions (anchor set)
   - lognormal perturbations of random real compositions (realistic spread)
   - recipe families for deliberate coverage: lean_sweet, rich, sour, n2_rich
2. For each composition: sample (T, P) with T/Tpc_kay >= 1.05 (stay clear of
   the two-phase region; Kay's rule pseudo-critical), P log-uniform up to
   min(140 MPa, Ppr_kay ~ 30).
3. Z from GERG-2008 (pyaga8, validated vs NIST to machine precision).
   Reject: non-finite, z outside (0.15, 3.0), or mechanically unstable /
   two-phase-root states (dP/drho <= 0).
4. Normalized output: data/synthetic/compositions.parquet (one row per gas)
   + data/synthetic/points.parquet (comp_id, T_K, P_MPa, z).
   Deterministic: fixed RNG seed; rerun reproduces byte-identical sampling.

Run:  python -X utf8 scripts/generate_synthetic.py [--n-comps 10000] [--pts 1000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyaga8

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC, load_nist_compositions  # noqa: E402

SEED = 20260728
OUT = ROOT / "data" / "synthetic"

# canonical component name -> pyaga8 Composition attribute
PYAGA8_ATTR = {
    "methane": "methane", "nitrogen": "nitrogen", "co2": "carbon_dioxide",
    "ethane": "ethane", "propane": "propane", "isobutane": "isobutane",
    "n_butane": "n_butane", "isopentane": "isopentane", "n_pentane": "n_pentane",
    "hexane": "hexane", "heptane": "heptane", "octane": "octane",
    "nonane": "nonane", "decane": "decane", "hydrogen": "hydrogen",
    "oxygen": "oxygen", "co": "carbon_monoxide", "water": "water",
    "h2s": "hydrogen_sulfide", "helium": "helium", "argon": "argon",
}

T_MIN_ABS, T_MAX = 250.0, 450.0        # K
P_MIN, P_MAX = 0.05, 140.0             # MPa
PPR_MAX = 30.0
Z_LO, Z_HI = 0.15, 3.0
TPR_MARGIN = 1.05                      # T >= margin * Tpc_kay


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return x / x.sum()


def build_composition_pool(rng: np.random.Generator, n_total: int) -> pd.DataFrame:
    nist = load_nist_compositions()          # 200 gases, x_* columns, sum=1
    real = nist.to_numpy(dtype=float)
    ncomp = len(COMPONENTS)
    idx = {c: i for i, c in enumerate(COMPONENTS)}

    rows, families = [], []

    n_real = min(len(real), n_total)
    for r in real[:n_real]:
        rows.append(r)
        families.append("nist_real")

    n_perturb = max(0, min(int(0.45 * n_total), n_total - len(rows)))
    for _ in range(n_perturb):
        base = real[rng.integers(len(real))]
        jitter = rng.lognormal(0.0, 0.35, ncomp)
        rows.append(_norm(base * jitter))
        families.append("nist_perturbed")

    def recipe(family: str, spec: dict) -> None:
        x = np.zeros(ncomp)
        for comp, (lo, hi) in spec.items():
            x[idx[comp]] = rng.uniform(lo, hi)
        x[idx["methane"]] += max(0.0, 1.0 - x.sum())   # methane fills remainder
        rows.append(_norm(x))
        families.append(family)

    n_left = n_total - len(rows)
    n_each = n_left // 4
    for _ in range(n_each):
        recipe("lean_sweet", {"nitrogen": (0.0, 0.05), "co2": (0.0, 0.03),
                              "ethane": (0.0, 0.06), "propane": (0.0, 0.02)})
    for _ in range(n_each):
        recipe("rich", {"ethane": (0.03, 0.15), "propane": (0.01, 0.10),
                        "n_butane": (0.0, 0.05), "isobutane": (0.0, 0.03),
                        "n_pentane": (0.0, 0.02), "hexane": (0.0, 0.01),
                        "co2": (0.0, 0.05), "nitrogen": (0.0, 0.05)})
    for _ in range(n_each):
        recipe("sour", {"h2s": (0.0, 0.25), "co2": (0.0, 0.25),
                        "nitrogen": (0.0, 0.05), "ethane": (0.0, 0.08),
                        "propane": (0.0, 0.03)})
    while len(rows) < n_total:
        recipe("n2_rich", {"nitrogen": (0.05, 0.40), "co2": (0.0, 0.10),
                           "ethane": (0.0, 0.05)})

    comps = pd.DataFrame(rows, columns=[f"x_{c}" for c in COMPONENTS])
    comps.insert(0, "comp_id", np.arange(len(comps)))
    comps.insert(1, "family", families)
    comps["tpc_kay"] = comps[[f"x_{c}" for c in COMPONENTS]].to_numpy() @ TC_VEC
    comps["ppc_kay"] = comps[[f"x_{c}" for c in COMPONENTS]].to_numpy() @ PC_VEC
    # composition-level split: entire gas goes to train or val (10% val)
    comps["split"] = np.where(rng.random(len(comps)) < 0.10, "val", "train")
    return comps


def generate_points(comps: pd.DataFrame, pts_per_comp: int,
                    rng: np.random.Generator):
    gerg = pyaga8.Gerg2008()
    xcols = [f"x_{c}" for c in COMPONENTS]
    attrs = [PYAGA8_ATTR[c] for c in COMPONENTS]

    out_comp, out_T, out_P, out_z = [], [], [], []
    n_reject = {"unstable": 0, "z_bounds": 0, "error": 0}
    t_start = time.perf_counter()

    for row in comps.itertuples(index=False):
        x = np.array([getattr(row, c) for c in xcols])
        c = pyaga8.Composition()
        for attr, val in zip(attrs, x):
            setattr(c, attr, float(val))
        try:
            gerg.set_composition(c)
        except Exception:
            n_reject["error"] += pts_per_comp
            continue

        t_min = max(T_MIN_ABS, TPR_MARGIN * row.tpc_kay)
        if t_min >= T_MAX:
            n_reject["error"] += pts_per_comp
            continue
        T = rng.uniform(t_min, T_MAX, pts_per_comp)
        p_hi = min(P_MAX, PPR_MAX * row.ppc_kay)
        P = np.exp(rng.uniform(np.log(P_MIN), np.log(p_hi), pts_per_comp))

        for Ti, Pi in zip(T, P):
            gerg.temperature = Ti
            gerg.pressure = Pi * 1000.0          # kPa
            try:
                gerg.calc_density(0)
                gerg.calc_properties()
            except Exception:
                n_reject["error"] += 1
                continue
            z = gerg.z
            if not np.isfinite(z) or not (Z_LO < z < Z_HI):
                n_reject["z_bounds"] += 1
                continue
            if gerg.dp_dd <= 0:                  # mechanically unstable root
                n_reject["unstable"] += 1
                continue
            out_comp.append(row.comp_id)
            out_T.append(Ti)
            out_P.append(Pi)
            out_z.append(z)

    dt = time.perf_counter() - t_start
    pts = pd.DataFrame({
        "comp_id": np.array(out_comp, dtype=np.int32),
        "T_K": np.array(out_T, dtype=np.float64),
        "P_MPa": np.array(out_P, dtype=np.float64),
        "z": np.array(out_z, dtype=np.float64),
    })
    return pts, n_reject, dt


def detail_crosscheck(comps: pd.DataFrame, pts: pd.DataFrame,
                      rng: np.random.Generator, n=20000) -> str:
    """Recompute a random sample with AGA8-DETAIL; report agreement."""
    det = pyaga8.Detail()
    attrs = [PYAGA8_ATTR[c] for c in COMPONENTS]
    xcols = [f"x_{c}" for c in COMPONENTS]
    sample = pts.sample(min(n, len(pts)), random_state=int(rng.integers(2**31)))
    cx = comps.set_index("comp_id")
    diffs = []
    for r in sample.itertuples(index=False):
        x = cx.loc[r.comp_id, xcols].to_numpy(dtype=float)
        c = pyaga8.Composition()
        for attr, val in zip(attrs, x):
            setattr(c, attr, float(val))
        try:
            det.set_composition(c)
            det.temperature = r.T_K
            det.pressure = r.P_MPa * 1000.0
            det.calc_density()
            det.calc_properties()
            if np.isfinite(det.z) and det.z > 0:
                diffs.append(abs(det.z - r.z) / r.z)
        except Exception:
            continue
    d = np.array(diffs)
    return (f"DETAIL cross-check on {len(d)} sampled points: "
            f"median |dz|/z = {np.median(d)*100:.4f}%, "
            f"p95 = {np.percentile(d, 95)*100:.4f}%, "
            f"max = {d.max()*100:.3f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-comps", type=int, default=10_000)
    ap.add_argument("--pts", type=int, default=1_000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Building composition pool ({args.n_comps})...")
    comps = build_composition_pool(rng, args.n_comps)
    print(comps.family.value_counts().to_string())

    print(f"\nGenerating up to {args.n_comps * args.pts:,} GERG-2008 points...")
    pts, rej, dt = generate_points(comps, args.pts, rng)
    print(f"kept {len(pts):,} points in {dt/60:.1f} min "
          f"({len(pts)/dt:,.0f} pts/s); rejected: {rej}")

    check = detail_crosscheck(comps, pts, rng)
    print(check)

    comps.to_parquet(OUT / "compositions.parquet", index=False)
    pts.to_parquet(OUT / "points.parquet", index=False)

    manifest = [
        "# Synthetic tier generation manifest",
        f"- seed: {SEED}",
        f"- compositions: {len(comps)} "
        f"({dict(comps.family.value_counts())}); split: {dict(comps.split.value_counts())}",
        f"- T range: [{T_MIN_ABS}, {T_MAX}] K with T >= {TPR_MARGIN} * Tpc_kay",
        f"- P range: log-uniform [{P_MIN}, {P_MAX}] MPa, capped at Ppr_kay <= {PPR_MAX}",
        f"- z bounds: ({Z_LO}, {Z_HI}); stability filter: dP/drho > 0",
        f"- points kept: {len(pts):,}; rejected: {rej}",
        f"- z stats: {pts.z.describe().round(4).to_dict()}",
        f"- {check}",
        "- EOS: GERG-2008 via pyaga8 (validated vs NIST reference to machine precision)",
    ]
    (OUT / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")
    print(f"\nWrote {OUT / 'points.parquet'}, compositions.parquet, MANIFEST.md")


if __name__ == "__main__":
    main()
