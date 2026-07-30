"""Unit tests: Python correlations vs zFactor R package reference grids.

The fixtures in tests/fixtures/*.csv were exported verbatim from the R
package's own test data (16 Tpr isotherms x 7 Ppr points each). Iterative
correlations must match the R implementation almost exactly; small explicit-
formula differences (float handling) get a slightly looser tolerance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zfactor.correlations import (  # noqa: E402
    dak, hall_yarborough, dpr, beggs_brill, shell, papay, kareem,
)

FIXTURES = ROOT / "tests" / "fixtures"


def load_grid(name):
    df = pd.read_csv(FIXTURES / f"{name}.csv", index_col=0)
    tpr = df.index.astype(float).to_numpy()
    ppr = df.columns.astype(float).to_numpy()
    P, T = np.meshgrid(ppr, tpr)
    return P, T, df.to_numpy(dtype=float)


CASES = [
    ("dak_16x7", dak, 5e-4),
    ("hy_16x7", hall_yarborough, 5e-4),
    ("dpr_16x7", dpr, 5e-4),
    ("bb_16x7", beggs_brill, 5e-4),
    ("sh_16x7", shell, 5e-4),
]


@pytest.mark.parametrize("name,func,rtol", CASES, ids=[c[0] for c in CASES])
def test_matches_r_reference(name, func, rtol):
    P, T, expected = load_grid(name)
    got = func(P, T)
    # compare where the R package produced a finite value
    mask = np.isfinite(expected)
    rel = np.abs(got[mask] - expected[mask]) / np.abs(expected[mask])
    worst = rel.max()
    assert worst < rtol, (
        f"{name}: worst relative diff {worst:.2e} at "
        f"Tpr={T[mask][rel.argmax()]}, Ppr={P[mask][rel.argmax()]} "
        f"(got {got[mask][rel.argmax()]:.6f}, ref {expected[mask][rel.argmax()]:.6f})"
    )


def test_papay_known_values():
    # Papay is explicit; verify the formula at hand-computed points
    z = papay(2.0, 1.5)
    expected = 1 - 3.53 * 2.0 / 10 ** (0.9813 * 1.5) + 0.274 * 4.0 / 10 ** (0.8157 * 1.5)
    assert abs(z - expected) < 1e-12


def test_kareem_tracks_hall_yarborough():
    # Kareem 2016 is an explicit approximation of HY; within its validity
    # range the paper reports <1% average deviation from iterative HY.
    tpr = np.linspace(1.15, 3.0, 20)
    ppr = np.linspace(0.2, 15.0, 40)
    P, T = np.meshgrid(ppr, tpr)
    zk = kareem(P, T)
    zh = hall_yarborough(P, T)
    mape = np.mean(np.abs(zk - zh) / zh) * 100
    assert mape < 1.0, f"Kareem vs HY MAPE {mape:.3f}% exceeds 1%"


def test_ideal_gas_limit():
    # all correlations must approach z=1 as Ppr -> 0
    for f in (dak, hall_yarborough, dpr, kareem):
        z = f(1e-6, 1.5)
        assert abs(z - 1.0) < 1e-3, f"{f.__name__} ideal-gas limit broken: {z}"


def test_vectorized_equals_scalar():
    ppr = np.array([0.5, 2.5, 6.5])
    tpr = np.array([1.3, 1.3, 1.3])
    for f in (dak, hall_yarborough, dpr, beggs_brill, shell, papay, kareem):
        vec = f(ppr, tpr)
        for i in range(3):
            assert abs(vec[i] - f(float(ppr[i]), 1.3)) < 1e-12, f.__name__
