"""Classical Z-factor correlations.

All functions are vectorized: ``f(ppr, tpr) -> z`` where inputs are scalars or
numpy arrays of pseudo-reduced pressure/temperature. Values outside a
correlation's published validity range are still computed (callers decide);
use ``VALIDITY`` to mask if needed.

References
----------
- Dranchuk & Abou-Kassem (1975), J. Can. Pet. Technol. 14(3).
- Hall & Yarborough (1973), Oil Gas J. 71(25).
- Dranchuk, Purvis & Robinson (1974), Inst. Pet. Technol. IP-74-008.
- Beggs & Brill (1973), JPT 25(5).
- Shell correlation as given in Kumar (2004) / zFactor R package.
- Papay (1968), OGIL Musz. Tud. Kozl.
- Kareem, Iwalewa & Al-Marhoun (2016), J Petrol Explor Prod Technol 6:481-492,
  Table 3 constants (explicit, linearized HY isotherms).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "dak", "hall_yarborough", "dpr", "beggs_brill", "shell", "papay",
    "kareem", "CORRELATIONS", "VALIDITY",
]


def _as_arrays(ppr, tpr):
    p = np.atleast_1d(np.asarray(ppr, dtype=float))
    t = np.atleast_1d(np.asarray(tpr, dtype=float))
    p, t = np.broadcast_arrays(p, t)
    return p.copy(), t.copy()


def _ret(z, ppr):
    return float(z[0]) if np.isscalar(ppr) or np.ndim(ppr) == 0 else z


# --------------------------------------------------------------------------- #
# Dranchuk & Abou-Kassem 1975 (11-constant Starling-type EOS fit)
# --------------------------------------------------------------------------- #
_DAK_A = (0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475,
          -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)


def _dak_z_of_rho(rho, t):
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11 = _DAK_A
    c1 = A1 + A2 / t + A3 / t**3 + A4 / t**4 + A5 / t**5
    c2 = A6 + A7 / t + A8 / t**2
    c3 = A9 * (A7 / t + A8 / t**2)
    r2 = rho * rho
    return (1.0 + c1 * rho + c2 * r2 - c3 * rho**5
            + A10 * (1.0 + A11 * r2) * (r2 / t**3) * np.exp(-A11 * r2))


def _dak_dz_drho(rho, t):
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11 = _DAK_A
    c1 = A1 + A2 / t + A3 / t**3 + A4 / t**4 + A5 / t**5
    c2 = A6 + A7 / t + A8 / t**2
    c3 = A9 * (A7 / t + A8 / t**2)
    r2 = rho * rho
    e = np.exp(-A11 * r2)
    dexp = (2 * rho / t**3) * e * (A10 * (1 + A11 * r2)) \
        + (r2 / t**3) * (A10 * (2 * A11 * rho)) * e \
        + A10 * (1 + A11 * r2) * (r2 / t**3) * e * (-2 * A11 * rho)
    return c1 + 2 * c2 * rho - 5 * c3 * rho**4 + dexp


def dak(ppr, tpr, tol=1e-10, maxiter=60):
    """Dranchuk-Abou-Kassem (1975). Newton solve on reduced density."""
    p, t = _as_arrays(ppr, tpr)
    k = 0.27 * p / t                      # rho_r * z = k
    rho = np.where(p > 0, k / 1.0, 1e-12)  # start at z = 1
    for _ in range(maxiter):
        zr = _dak_z_of_rho(rho, t)
        f = rho * zr - k                            # rho*z(rho) = k
        df = zr + rho * _dak_dz_drho(rho, t)
        step = f / df
        rho = np.clip(rho - step, 1e-12, None)
        if np.all(np.abs(step) < tol):
            break
    z = np.where(p > 0, k / rho, 1.0)
    return _ret(z, ppr)


# --------------------------------------------------------------------------- #
# Hall & Yarborough 1973 (Carnahan-Starling hard sphere)
# --------------------------------------------------------------------------- #
def hall_yarborough(ppr, tpr, tol=1e-12, maxiter=100):
    p, t_in = _as_arrays(ppr, tpr)
    t = 1.0 / t_in
    A = 0.06125 * t * np.exp(-1.2 * (1 - t) ** 2)
    B = 14.76 * t - 9.76 * t**2 + 4.58 * t**3
    C = 90.7 * t - 242.2 * t**2 + 42.4 * t**3
    D = 2.18 + 2.82 * t
    y = np.clip(A * p, 1e-12, 0.6)  # ideal-gas start
    for _ in range(maxiter):
        f = (-A * p + (y + y**2 + y**3 - y**4) / (1 - y) ** 3
             - B * y**2 + C * y**D)
        df = ((1 + 4 * y + 4 * y**2 - 4 * y**3 + y**4) / (1 - y) ** 4
              - 2 * B * y + C * D * y ** (D - 1))
        step = f / df
        y = np.clip(y - step, 1e-12, 0.99999)
        if np.all(np.abs(step) < tol):
            break
    z = np.where(p > 0, A * p / y, 1.0)
    return _ret(z, ppr)


# --------------------------------------------------------------------------- #
# Dranchuk, Purvis & Robinson 1974 (8-constant BWR fit)
# --------------------------------------------------------------------------- #
_DPR_A = (0.31506237, -1.0467099, -0.57832729, 0.53530771,
          -0.61232032, -0.10488813, 0.68157001, 0.68446549)


def _dpr_z_of_rho(rho, t):
    A1, A2, A3, A4, A5, A6, A7, A8 = _DPR_A
    r2 = rho * rho
    return (1.0 + (A1 + A2 / t + A3 / t**3) * rho + (A4 + A5 / t) * r2
            + A5 * A6 * rho**5 / t
            + A7 * (r2 / t**3) * (1 + A8 * r2) * np.exp(-A8 * r2))


def _dpr_dz_drho(rho, t):
    A1, A2, A3, A4, A5, A6, A7, A8 = _DPR_A
    r2 = rho * rho
    e = np.exp(-A8 * r2)
    dlast = (A7 / t**3) * e * (2 * rho * (1 + A8 * r2)
                               + r2 * (2 * A8 * rho)
                               + (r2 * (1 + A8 * r2)) * (-2 * A8 * rho))
    return ((A1 + A2 / t + A3 / t**3) + 2 * (A4 + A5 / t) * rho
            + 5 * A5 * A6 * rho**4 / t + dlast)


def dpr(ppr, tpr, tol=1e-12, maxiter=100):
    """Dranchuk-Purvis-Robinson (1974). Newton solve on reduced density."""
    p, t = _as_arrays(ppr, tpr)
    k = 0.27 * p / t
    rho = np.where(p > 0, k, 1e-12)
    for _ in range(maxiter):
        f = rho * _dpr_z_of_rho(rho, t) - k
        df = _dpr_z_of_rho(rho, t) + rho * _dpr_dz_drho(rho, t)
        step = f / df
        rho = np.clip(rho - step, 1e-12, None)
        if np.all(np.abs(step) < tol):
            break
    z = np.where(p > 0, k / rho, 1.0)
    return _ret(z, ppr)


# --------------------------------------------------------------------------- #
# Beggs & Brill 1973 (explicit)
# --------------------------------------------------------------------------- #
def beggs_brill(ppr, tpr):
    p, t = _as_arrays(ppr, tpr)
    A = 1.39 * np.sqrt(np.clip(t - 0.92, 0, None)) - 0.36 * t - 0.101
    B = ((0.62 - 0.23 * t) * p
         + (0.066 / (t - 0.86) - 0.037) * p**2
         + 0.32 * p**6 / 10 ** (9 * (t - 1)))
    C = 0.132 - 0.32 * np.log10(t)
    D = 10 ** (0.3106 - 0.49 * t + 0.1824 * t**2)
    # exp(B) overflows harmlessly for large B; suppress the warning
    with np.errstate(over="ignore"):
        z = A + (1 - A) / np.exp(B) + C * p**D
    return _ret(z, ppr)


# --------------------------------------------------------------------------- #
# Shell correlation (Kumar 2004; as in the zFactor R package)
# --------------------------------------------------------------------------- #
def shell(ppr, tpr):
    p, t = _as_arrays(ppr, tpr)
    A = -0.101 - 0.36 * t + 1.3868 * np.sqrt(np.clip(t - 0.919, 0, None))
    B = 0.021 + 0.04275 / (t - 0.65)
    C = 0.6222 - 0.224 * t
    D = 0.0657 / (t - 0.85) - 0.037
    E = 0.32 * np.exp(-19.53 * (t - 1))
    F = 0.122 * np.exp(-11.3 * (t - 1))
    G = p * (C + D * p + E * p**4)
    z = A + B * p + (1 - A) * np.exp(-G) - F * (p / 10) ** 4
    return _ret(z, ppr)


# --------------------------------------------------------------------------- #
# Papay 1968 (explicit, 2-term)
# --------------------------------------------------------------------------- #
def papay(ppr, tpr):
    p, t = _as_arrays(ppr, tpr)
    z = 1 - 3.53 * p / 10 ** (0.9813 * t) + 0.274 * p**2 / 10 ** (0.8157 * t)
    return _ret(z, ppr)


# --------------------------------------------------------------------------- #
# Kareem, Iwalewa & Al-Marhoun 2016 (explicit linearized HY; Table 3 constants)
# --------------------------------------------------------------------------- #
_KAREEM_A = {
    1: 0.317842, 2: 0.382216, 3: -7.768354, 4: 14.290531, 5: 0.000002,
    6: -0.004693, 7: 0.096254, 8: 0.166720, 9: 0.966910, 10: 0.063069,
    11: -1.966847, 12: 21.0581, 13: -27.0246, 14: 16.23, 15: 207.783,
    16: -488.161, 17: 176.29, 18: 1.88453, 19: 3.05921,
}


def kareem(ppr, tpr):
    p, t_in = _as_arrays(ppr, tpr)
    a = _KAREEM_A
    t = 1.0 / t_in
    A = a[1] * t * np.exp(a[2] * (1 - t) ** 2) * p
    B = a[3] * t + a[4] * t**2 + a[5] * t**6 * p**6
    C = a[9] + a[8] * t * p + a[7] * t**2 * p**2 + a[6] * t**3 * p**3
    D = a[10] * t * np.exp(a[11] * (1 - t) ** 2)
    E = a[12] * t + a[13] * t**2 + a[14] * t**3
    F = a[15] * t + a[16] * t**2 + a[17] * t**3
    G = a[18] + a[19] * t
    y = D * p / ((1 + A**2) / C - A**2 * B / C**3)
    z = (D * p * (1 + y + y**2 - y**3)
         / ((D * p + E * y**2 - F * y**G) * (1 - y) ** 3))
    return _ret(z, ppr)


CORRELATIONS = {
    "DAK": dak,
    "HY": hall_yarborough,
    "DPR": dpr,
    "BB": beggs_brill,
    "SH": shell,
    "PP": papay,
    "KAREEM": kareem,
}

# published validity ranges (tpr_min, tpr_max, ppr_min, ppr_max)
VALIDITY = {
    "DAK": (1.0, 3.0, 0.2, 30.0),
    "HY": (1.15, 3.0, 0.2, 20.5),   # HY unreliable below Tpr ~1.1
    "DPR": (1.05, 3.0, 0.2, 30.0),
    "BB": (1.15, 2.4, 0.2, 15.0),
    "SH": (1.05, 3.0, 0.2, 15.0),
    "PP": (1.2, 3.0, 0.2, 15.0),
    "KAREEM": (1.15, 3.0, 0.2, 15.0),
}
