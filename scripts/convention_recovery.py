"""Leakage-free recovery of each test mix's pseudo-critical convention, then
GERG-2008 prediction at the reconstructed absolute (T, P).

Identification signal: laboratories measure at round temperatures (5 C or
10 F grids) and round pressures (0.5/1 MPa or 100/250/500 psi grids). The
(Tpc, Ppc) scale that maps a mix's published reduced coordinates onto round
lab values is the convention its source paper used. Measured z is NEVER used
in the recovery -> no leakage; this is input-metadata reconstruction.

Run:  python -X utf8 scripts/convention_recovery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyaga8

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from train_models_v2 import MW_VEC  # noqa: E402
from generate_synthetic import PYAGA8_ATTR  # noqa: E402
from zfactor.correlations import dak, hall_yarborough  # noqa: E402

X_COLS = [f"x_{c}" for c in COMPONENTS]
PSI = 0.00689476  # MPa per psi


def t_round_dist(T_K: np.ndarray) -> float:
    """Mean distance (K) to nearest round lab temperature (5C or 10F grid)."""
    c = T_K - 273.15
    f = c * 1.8 + 32
    d_c = np.abs(c - np.round(c / 5) * 5)
    d_f = np.abs(f - np.round(f / 10) * 10) / 1.8
    return float(np.minimum(d_c, d_f).mean())


def p_round_dist(P_MPa: np.ndarray) -> float:
    """Mean normalized distance to round pressure grids (fraction of spacing)."""
    best = None
    for step in [1.0, 0.5, 100 * PSI, 250 * PSI, 500 * PSI]:
        d = np.abs(P_MPa - np.round(P_MPa / step) * step) / step
        m = float(d.mean())
        best = m if best is None else min(best, m)
    return best


def wichert_aziz(tpc_K, ppc_MPa, x_co2, x_h2s):
    """Wichert-Aziz sour-gas correction (native units degR/psia)."""
    A, B = x_co2 + x_h2s, x_h2s
    tpc_R = tpc_K * 1.8
    ppc_psi = ppc_MPa / PSI
    eps = 120 * (A**0.9 - A**1.6) + 15 * (B**0.5 - B**4)
    tpc_R_c = tpc_R - eps
    ppc_c = ppc_psi * tpc_R_c / (tpc_R + B * (1 - B) * eps)
    return tpc_R_c / 1.8, ppc_c * PSI


def recover_mix(g: pd.DataFrame):
    """Return (Tpc, Ppc, t_score, p_score) recovered from roundness only.

    Candidates: Kay / Sutton / Standing, each with and without Wichert-Aziz
    sour correction; the best discrete candidate is chosen by combined
    roundness, then refined continuously within +-1.5% (narrow window
    prevents locking onto a wrong-but-round grid).
    """
    x = g[X_COLS].fillna(0).iloc[0].to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    x_co2 = x[COMPONENTS.index("co2")]
    x_h2s = x[COMPONENTS.index("h2s")]
    base = {"kay": (x @ TC_VEC, x @ PC_VEC),
            "sutton": ((169.2 + 349.5 * gamma - 74 * gamma**2) / 1.8,
                       (756.8 - 131.0 * gamma - 3.6 * gamma**2) * PSI),
            "standing": ((168 + 325 * gamma - 12.5 * gamma**2) / 1.8,
                         (677 + 15 * gamma - 37.5 * gamma**2) * PSI)}
    cands = dict(base)
    if x_co2 + x_h2s > 0.02:
        for k, (tc, pc) in base.items():
            cands[k + "+WA"] = wichert_aziz(tc, pc, x_co2, x_h2s)

    tprs = np.array(sorted(g.Tpr.unique()))
    pprs = g.Ppr.to_numpy()

    def combined(tc, pc):
        return t_round_dist(tprs * tc) / 0.5 + p_round_dist(pprs * pc) / 0.1

    best = min(cands, key=lambda k: combined(*cands[k]))
    tpc_b, ppc_b = cands[best]

    # narrow continuous refinement around the winning discrete candidate
    scales = np.arange(0.985, 1.0151, 0.0005)
    tpc = tpc_b * scales[int(np.argmin([t_round_dist(tprs * tpc_b * s)
                                        for s in scales]))]
    ppc = ppc_b * scales[int(np.argmin([p_round_dist(pprs * ppc_b * s)
                                        for s in scales]))]
    return tpc, ppc, t_round_dist(tprs * tpc), p_round_dist(pprs * ppc)


def gerg_z(x: np.ndarray, T: np.ndarray, P_MPa: np.ndarray) -> np.ndarray:
    g = pyaga8.Gerg2008()
    c = pyaga8.Composition()
    for comp, attr in PYAGA8_ATTR.items():
        setattr(c, attr, float(x[COMPONENTS.index(comp)]))
    g.set_composition(c)
    out = np.full(len(T), np.nan)
    for k, (Ti, Pi) in enumerate(zip(T, P_MPa)):
        try:
            g.temperature = Ti
            g.pressure = Pi * 1000.0
            g.calc_density(0)
            g.calc_properties()
            if np.isfinite(g.z) and 0.1 < g.z < 3.5:
                out[k] = g.z
        except Exception:
            pass
    return out


def main():
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    t = m[(m.tier == "experimental") & (m.quality_flag == "ok")].copy()

    rows, zg_all = [], np.full(len(t), np.nan)
    t = t.reset_index(drop=True)
    for mix, g in t.groupby("mix_id"):
        tpc, ppc, ts, ps = recover_mix(g)
        x = g[X_COLS].fillna(0).iloc[0].to_numpy()
        T = g.Tpr.to_numpy() * tpc
        P = g.Ppr.to_numpy() * ppc
        zg = gerg_z(x, T, P)
        zg_all[g.index] = zg
        ok = np.isfinite(zg)
        y = g.z.to_numpy()
        mape = np.mean(np.abs(zg[ok] - y[ok]) / y[ok]) * 100 if ok.any() else np.nan
        zdak = dak(g.Ppr.to_numpy(), g.Tpr.to_numpy())
        rows.append({"mix": mix, "n": len(g), "Tpc": round(tpc, 1),
                     "Ppc": round(ppc, 3), "t_round_K": round(ts, 2),
                     "p_round": round(ps, 3),
                     "GERG_MAPE": round(mape, 3),
                     "DAK_MAPE": round(float(np.mean(np.abs(zdak - y) / y) * 100), 3)})

    rep = pd.DataFrame(rows).sort_values("mix", key=lambda s: s.str[3:].astype(int))
    print(rep.to_string(index=False))

    y = t.z.to_numpy()
    ok = np.isfinite(zg_all)
    zdak = dak(t.Ppr.to_numpy(), t.Tpr.to_numpy())
    zhy = hall_yarborough(t.Ppr.to_numpy(), t.Tpr.to_numpy())
    print(f"\nOVERALL (n={int(ok.sum())} of {len(t)}):")
    print(f"  GERG @ recovered (T,P): MAPE {np.mean(np.abs(zg_all[ok]-y[ok])/y[ok])*100:.4f}%")
    print(f"  DAK  (same points):     MAPE {np.mean(np.abs(zdak[ok]-y[ok])/y[ok])*100:.4f}%")
    print(f"  HY   (same points):     MAPE {np.mean(np.abs(zhy[ok]-y[ok])/y[ok])*100:.4f}%")

    out = t[["mix_id", "Ppr", "Tpr", "z"]].copy()
    out["z_gerg_recovered"] = zg_all
    out.to_parquet(ROOT / "reports" / "convention_recovery.parquet")


if __name__ == "__main__":
    main()
