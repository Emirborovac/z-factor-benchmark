"""Regional benchmark.

A) Libya field spot-check: measured z (gas gravity route, chart-track only).
B) Regional composition study: every method vs GERG-2008 on 1,374 real
   regional gases across a realistic operating grid. Measures the error an
   engineer in each region would incur by using a correlation instead of the
   reference EOS.

Run:  python -X utf8 scripts/eval_regional.py
"""
from __future__ import annotations

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
# realistic operating grid: transmission / gathering / reservoir
T_GRID = np.array([283.15, 298.15, 323.15, 353.15, 393.15])          # K
P_GRID = np.array([1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0, 40.0])      # MPa


def load_nn():
    ck = torch.load(ROOT / "models" / "nn" / "eos_nn.pt", weights_only=False)
    net = Net(ck["d_in"], ck["width"], ck["blocks"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def nn_predict(net, ck, T, P, ppr, tpr, x, gamma):
    F = build_features(T, P, ppr, tpr, x, gamma)
    with torch.no_grad():
        r = net(torch.tensor((F - ck["mu"]) / ck["sd"])).squeeze(1).numpy()
    return C.dak(ppr, tpr) + r


def gerg_batch(x_rows, T, P):
    g = pyaga8.Gerg2008()
    out = np.full(len(T), np.nan)
    last = None
    for i in range(len(T)):
        xr = x_rows[i]
        if last is None or not np.array_equal(xr, last):
            c = pyaga8.Composition()
            for comp, attr in PYAGA8_ATTR.items():
                setattr(c, attr, float(xr[COMPONENTS.index(comp)]))
            try:
                g.set_composition(c)
            except Exception:
                last = None
                continue
            last = xr
        try:
            g.temperature = float(T[i])
            g.pressure = float(P[i]) * 1000.0
            g.calc_density(0)
            g.calc_properties()
            if np.isfinite(g.z) and 0.1 < g.z < 3.5:
                out[i] = g.z
        except Exception:
            pass
    return out


def part_a(net, ck):
    a = pd.read_parquet(ROOT / "data" / "processed" / "regional_measured.parquet")
    gg = a.gas_gravity.to_numpy()
    T, P, z = a.T_K.to_numpy(), a.P_MPa.to_numpy(), a.z_measured.to_numpy()
    # Sutton pseudo-criticals from gravity (the only route available: no
    # composition is reported in these papers)
    tpc = (169.2 + 349.5 * gg - 74 * gg ** 2) / 1.8
    ppc = (756.8 - 131.0 * gg - 3.6 * gg ** 2) * PSI
    ppr, tpr = P / ppc, T / tpc
    # NOTE: these papers report gas gravity only - no composition. The
    # composition-based NN is therefore NOT APPLICABLE here (feeding it a
    # placeholder composition would fabricate its inputs). Only the
    # gravity->pseudo-critical->correlation route is defensible on this data.
    preds = {"DAK": C.dak(ppr, tpr), "HY": C.hall_yarborough(ppr, tpr),
             "Kareem": C.kareem(ppr, tpr), "DPR": C.dpr(ppr, tpr)}
    rows = []
    for k, p in preds.items():
        e = np.abs(p - z) / z * 100
        rows.append({"method": k, "MAPE_%": e.mean(), "max_APE_%": e.max()})
    df = pd.DataFrame(rows).set_index("method").sort_values("MAPE_%")
    return a, df


def part_b(net, ck):
    b = pd.read_parquet(ROOT / "data" / "processed" / "regional_compositions.parquet")
    x = b[X_COLS].to_numpy()
    n_gas = len(b)
    TT, PP = np.meshgrid(T_GRID, P_GRID, indexing="ij")
    grid = np.column_stack([TT.ravel(), PP.ravel()])
    n_pt = len(grid)

    X = np.repeat(x, n_pt, axis=0)
    T = np.tile(grid[:, 0], n_gas)
    P = np.tile(grid[:, 1], n_gas)
    region = np.repeat(b.region.to_numpy(), n_pt)

    gamma = (X @ MW_VEC) / 28.9647
    tpc0, ppc0 = X @ TC_VEC, X @ PC_VEC
    tpcw, ppcw = wichert_aziz(tpc0, ppc0, X[:, COMPONENTS.index("co2")],
                              X[:, COMPONENTS.index("h2s")])
    ppr_w, tpr_w = P / ppcw, T / tpcw
    ppr_k, tpr_k = P / ppc0, T / tpc0

    print(f"  evaluating {n_gas} gases x {n_pt} states = {len(T):,} points...")
    ref = gerg_batch(X, T, P)
    preds = {
        "DAK (Kay+WA)": C.dak(ppr_w, tpr_w),
        "HY (Kay+WA)": C.hall_yarborough(ppr_w, tpr_w),
        "Kareem (Kay+WA)": C.kareem(ppr_w, tpr_w),
        "DAK (Sutton+WA)": C.dak(P / ((756.8 - 131.0 * gamma - 3.6 * gamma**2)
                                      * PSI),
                                 T / ((169.2 + 349.5 * gamma - 74 * gamma**2)
                                      / 1.8)),
        "Our NN (raw)": nn_predict(net, ck, T, P, ppr_k, tpr_k, X, gamma),
    }
    from zfactor.predict import predict_z
    z_guard, status = predict_z(T, P, X)
    preds["Our NN (domain-guarded)"] = z_guard
    print(f"  domain guard: {(status=='ok').sum():,}/{len(status):,} in-domain "
          f"({100*(status=='ok').mean():.1f}%), rest fall back to DAK")
    ok = np.isfinite(ref)
    rows = []
    for k, p in preds.items():
        m = ok & np.isfinite(p)
        e = np.abs(p[m] - ref[m]) / ref[m] * 100
        rows.append({"method": k, "n": int(m.sum()), "MAPE_vs_GERG_%": e.mean(),
                     "p95_%": np.percentile(e, 95), "max_%": e.max()})
    overall = pd.DataFrame(rows).set_index("method").sort_values("MAPE_vs_GERG_%")

    per = {}
    for k, p in preds.items():
        vals = {}
        for r in np.unique(region):
            m = ok & np.isfinite(p) & (region == r)
            vals[r] = np.mean(np.abs(p[m] - ref[m]) / ref[m]) * 100 if m.any() else np.nan
        per[k] = vals
    per_df = pd.DataFrame(per)
    per_df["n_gases"] = b.groupby("region").size()
    return overall, per_df


def main():
    net, ck = load_nn()
    a_data, a_tbl = part_a(net, ck)
    print("PART A - Libya field measurements (n=%d, measured z)" % len(a_data))
    print(a_tbl.round(4).to_string())

    print("\nPART B - regional composition benchmark (reference = GERG-2008)")
    overall, per_df = part_b(net, ck)
    print(overall.round(4).to_string())
    print("\nPer-region MAPE vs GERG (%):")
    print(per_df.round(3).to_string())

    out = ["# Regional benchmark\n",
           "## Part A - Libya field data (measured z)\n",
           f"n = {len(a_data)} laboratory points from two Libyan studies "
           "(Ghani oilfield 2024; Libyan Petroleum Institute 2022). "
           "Low-pressure differential-liberation data (0.10-3.10 MPa, "
           "z 0.94-1.00): a regional spot-check, not a discriminating test. "
           "Only gas gravity is reported, so pseudo-criticals use Sutton.\n\n",
           a_tbl.round(4).to_markdown(),
           "\n\n## Part B - regional gas compositions vs GERG-2008\n",
           f"{len(pd.read_parquet(ROOT/'data'/'processed'/'regional_compositions.parquet'))} "
           "real regional gases (Netherlands NLOG national database; GIIGNL LNG "
           "exporters; published sour field compositions) evaluated over "
           f"{len(T_GRID)}x{len(P_GRID)} realistic operating states. No measured "
           "z exists for these gases, so GERG-2008 is the reference: this "
           "quantifies the error an engineer incurs by using a correlation.\n\n",
           overall.round(4).to_markdown(),
           "\n\n### Per-region\n\n", per_df.round(3).to_markdown()]
    (ROOT / "reports" / "regional_benchmark.md").write_text("\n".join(out),
                                                            encoding="utf-8")
    print("\nwrote reports/regional_benchmark.md")


if __name__ == "__main__":
    main()
