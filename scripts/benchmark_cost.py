"""Measure the computational cost the paper's title claims.

The title promises "reference-equation accuracy at correlation cost" and nothing
in the manuscript measured the cost. This does, on the same machine, over the
same states, using the released implementations.

Two things are reported, because one alone would be misleading:

  * Throughput as delivered by the released implementations. This is what a user
    actually experiences, but it is not a clean measure of algorithmic
    complexity: pyaga8 is a Rust library driven from Python one state at a time,
    so its numbers carry per-call interpreter overhead, while the correlations
    and the network are evaluated over whole arrays at once. The reference-EOS
    figures are therefore an upper bound on their intrinsic cost.

  * Iteration counts, which are hardware- and language-independent. A method
    that needs no iteration cannot fail to converge, which is a property rather
    than a speed.

Run:  python -X utf8 scripts/benchmark_cost.py
"""
from __future__ import annotations

import json
import sys
import time
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
from convention_recovery import wichert_aziz  # noqa: E402
from generate_synthetic import PYAGA8_ATTR  # noqa: E402
import zfactor.correlations as C  # noqa: E402

X_COLS = [f"x_{c}" for c in COMPONENTS]
N_BATCH = 20_000          # states per timed run
REPEATS = 5


def timeit(fn, repeats=REPEATS):
    fn()                                     # warm up
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def newton_iters_dak(ppr, tpr, tol=1e-10, itmax=100):
    """Iteration count for the DAK/DPR/HY family, counted the same way the
    solver in zfactor.correlations does it."""
    A = [0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361,
         0.1844, 0.1056, 0.6134, 0.7210]
    y = 0.27 * ppr / tpr                     # initial reduced density
    used = np.zeros(len(ppr), dtype=int)
    live = np.ones(len(ppr), dtype=bool)
    for k in range(itmax):
        if not live.any():
            break
        t = tpr[live]
        r = y[live]
        c1 = A[0] + A[1] / t + A[2] / t**3 + A[3] / t**4 + A[4] / t**5
        c2 = A[5] + A[6] / t + A[7] / t**2
        c3 = A[8] * (A[6] / t + A[7] / t**2)
        c4 = A[9] * (1 + A[10] * r**2) * (r**2 / t**3) * np.exp(-A[10] * r**2)
        z = 1 + c1 * r + c2 * r**2 - c3 * r**5 + c4
        f = z - 0.27 * ppr[live] / (r * t)
        dz = c1 + 2 * c2 * r - 5 * c3 * r**4
        df = dz + 0.27 * ppr[live] / (r**2 * t)
        step = f / np.where(np.abs(df) < 1e-14, 1e-14, df)
        rn = np.clip(r - step, 1e-8, 3.0)
        conv = np.abs(rn - r) < tol
        y[live] = rn
        used[live] += 1
        idx = np.where(live)[0]
        live[idx[conv]] = False
    return used


def main() -> None:
    t = pd.read_parquet(ROOT / "data" / "processed" / "test_b.parquet")
    x = t[X_COLS].to_numpy()
    P, T = t.P_MPa.to_numpy(), t.T_K.to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    tpc, ppc = x @ TC_VEC, x @ PC_VEC
    tpcw, ppcw = wichert_aziz(tpc, ppc, t.x_co2.to_numpy(), t.x_h2s.to_numpy())
    ppr, tpr = P / ppc, T / tpc

    # tile the benchmark up to N_BATCH states, all inside the NG domain
    ng = ((t.x_methane >= 0.5).to_numpy() & (tpr <= 3.0) & (T <= 500))
    idx = np.where(ng)[0]
    rep = int(np.ceil(N_BATCH / len(idx)))
    sel = np.tile(idx, rep)[:N_BATCH]
    Xb, Pb, Tb = x[sel], P[sel], T[sel]
    pprb, tprb = ppr[sel], tpr[sel]
    pprw, tprw = (P / ppcw)[sel], (T / tpcw)[sel]
    gammab = gamma[sel]

    ck = torch.load(ROOT / "models" / "nn" / "eos_nn_detail.pt",
                    weights_only=False)
    net = Net(ck["d_in"], ck["width"], ck["blocks"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    Fb = build_features(Tb, Pb, pprb, tprb, Xb, gammab)
    Fn = torch.tensor((Fb - ck["mu"]) / ck["sd"])

    def run_nncf():
        with torch.no_grad():
            r = net(Fn).squeeze(1).numpy()
        return C.dak(pprb, tprb) + r

    def eos_runner(cls):
        def go():
            eos = cls()
            out = np.empty(N_BATCH)
            last = None
            for i in range(N_BATCH):
                xr = Xb[i]
                if last is None or not np.array_equal(xr, last):
                    comp = pyaga8.Composition()
                    for c_, a_ in PYAGA8_ATTR.items():
                        setattr(comp, a_, float(xr[COMPONENTS.index(c_)]))
                    eos.set_composition(comp)
                    last = xr
                eos.temperature = float(Tb[i])
                eos.pressure = float(Pb[i]) * 1000.0
                if cls is pyaga8.Gerg2008:
                    eos.calc_density(0)
                else:
                    eos.calc_density()
                eos.calc_properties()
                out[i] = eos.z
            return out
        return go

    cases = [
        ("NNCF (this work)", run_nncf, "1 forward pass"),
        ("DAK", lambda: C.dak(pprw, tprw), "Newton"),
        ("Hall-Yarborough", lambda: C.hall_yarborough(pprw, tprw), "Newton"),
        ("DPR", lambda: C.dpr(pprw, tprw), "Newton"),
        ("AGA8-DETAIL", eos_runner(pyaga8.Detail), "density solve"),
        ("GERG-2008", eos_runner(pyaga8.Gerg2008), "density solve"),
    ]

    print(f"cost benchmark: {N_BATCH:,} states, best of {REPEATS}, "
          f"CPU, {torch.get_num_threads()} torch threads\n")
    rows = []
    for name, fn, kind in cases:
        dt = timeit(fn, repeats=2 if "solve" in kind else REPEATS)
        per = dt / N_BATCH * 1e6
        rows.append((name, kind, per, N_BATCH / dt))
        print(f"  {name:20s} {kind:16s} {per:10.2f} us/state "
              f"{N_BATCH / dt:12,.0f} states/s")

    it = newton_iters_dak(pprw, tprw)
    print(f"\nNewton iterations for the correlation family (tol 1e-10): "
          f"mean {it.mean():.1f}, max {int(it.max())}")
    print("NNCF iterations: 0 (fixed-depth network; convergence failure is "
          "not possible)")

    nncf_us = rows[0][2]
    print("\nrelative to NNCF:")
    for name, kind, per, _ in rows[1:]:
        print(f"  {name:20s} {per / nncf_us:8.1f}x")

    out = {
        "n_states": N_BATCH, "repeats": REPEATS,
        "torch_threads": torch.get_num_threads(),
        "us_per_state": {r[0]: round(r[2], 3) for r in rows},
        "states_per_second": {r[0]: int(r[3]) for r in rows},
        "newton_iters_correlations": {"mean": round(float(it.mean()), 2),
                                      "max": int(it.max())},
        "nncf_iters": 0,
        "caveat": ("pyaga8 is a Rust library driven from Python one state at a "
                   "time, so the reference-EOS timings include per-call "
                   "interpreter overhead and are an upper bound on their "
                   "intrinsic cost. The correlations and the network are "
                   "evaluated over whole arrays."),
    }
    (ROOT / "reports" / "cost_benchmark.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote reports/cost_benchmark.json")


if __name__ == "__main__":
    main()
