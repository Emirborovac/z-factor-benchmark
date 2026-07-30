"""Deployable predictor with an applicability-domain guard.

The neural surrogate matches GERG-2008 inside its training envelope and
degrades sharply outside it. A silent wrong answer is worse than a refusal,
so `predict_z` returns a status for every point and falls back to DAK when a
point is out of domain.

    z, status = predict_z(T_K, P_MPa, composition)

status: "ok"          - inside the applicability domain, NN used
        "fallback"    - outside domain, DAK (with Wichert-Aziz) used instead
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .correlations import dak

ROOT = Path(__file__).resolve().parents[2]

COMPONENTS = [
    "methane", "nitrogen", "co2", "ethane", "propane", "isobutane", "n_butane",
    "isopentane", "n_pentane", "hexane", "heptane", "octane", "nonane",
    "decane", "hydrogen", "oxygen", "co", "water", "h2s", "helium", "argon",
]
TC = np.array([190.564, 126.192, 304.128, 305.322, 369.825, 407.817, 425.125,
               460.35, 469.70, 507.60, 540.13, 568.74, 594.55, 617.70, 33.145,
               154.581, 132.86, 647.096, 373.10, 5.1953, 150.687])
PC = np.array([4.5992, 3.3958, 7.3773, 4.8722, 4.2512, 3.6290, 3.7960, 3.3780,
               3.3675, 3.0250, 2.7360, 2.4970, 2.2810, 2.1030, 1.2964, 5.0430,
               3.4940, 22.0640, 9.0000, 0.2276, 4.8630])
MW = np.array([16.043, 28.014, 44.01, 30.07, 44.097, 58.123, 58.123, 72.15,
               72.15, 86.177, 100.204, 114.231, 128.258, 142.285, 2.016,
               31.999, 28.01, 18.015, 34.081, 4.003, 39.948])

# applicability domain = the declared training envelope
DOMAIN = dict(ch4_min=0.30, T=(200.0, 520.0), Tpr=(1.00, 3.05),
              Ppr=(0.02, 16.0))

_MODEL = None


class _Res(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(w, w), torch.nn.SiLU(),
                                     torch.nn.Linear(w, w))
        self.act = torch.nn.SiLU()

    def forward(self, h):
        return self.act(h + self.f(h))


class _Net(torch.nn.Module):
    """Must mirror scripts/train_nn.py:Net so state_dict keys match."""

    def __init__(self, d_in, w, blocks):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(d_in, w), torch.nn.SiLU())
        self.body = torch.nn.Sequential(*[_Res(w) for _ in range(blocks)])
        self.out = torch.nn.Linear(w, 1)

    def forward(self, x):
        return self.out(self.body(self.inp(x)))


def _net(d_in, w, blocks):
    return _Net(d_in, w, blocks)


def _load():
    global _MODEL
    if _MODEL is None:
        ck = torch.load(ROOT / "models" / "nn" / "eos_nn.pt", weights_only=False)
        net = _net(ck["d_in"], ck["width"], ck["blocks"])
        net.load_state_dict(ck["state_dict"])
        net.eval()
        _MODEL = (net, ck)
    return _MODEL


def wichert_aziz(tpc, ppc, x_co2, x_h2s):
    A, B = x_co2 + x_h2s, x_h2s
    tpc_R, ppc_psi = tpc * 1.8, ppc / 0.00689476
    eps = 120 * (A ** 0.9 - A ** 1.6) + 15 * (B ** 0.5 - B ** 4)
    tpc_c = tpc_R - eps
    return tpc_c / 1.8, (ppc_psi * tpc_c / (tpc_R + B * (1 - B) * eps)) * 0.00689476


def predict_z(T_K, P_MPa, composition):
    """composition: array (n, 21) of mole fractions in COMPONENTS order."""
    T = np.atleast_1d(np.asarray(T_K, float))
    P = np.atleast_1d(np.asarray(P_MPa, float))
    x = np.atleast_2d(np.asarray(composition, float))
    x = x / x.sum(1, keepdims=True)

    gamma = (x @ MW) / 28.9647
    tpc, ppc = x @ TC, x @ PC
    ppr, tpr = P / ppc, T / tpc

    in_dom = ((x[:, 0] >= DOMAIN["ch4_min"])
              & (T >= DOMAIN["T"][0]) & (T <= DOMAIN["T"][1])
              & (tpr >= DOMAIN["Tpr"][0]) & (tpr <= DOMAIN["Tpr"][1])
              & (ppr >= DOMAIN["Ppr"][0]) & (ppr <= DOMAIN["Ppr"][1]))

    tpc_w, ppc_w = wichert_aziz(tpc, ppc, x[:, 2], x[:, 18])
    z = dak(P / ppc_w, T / tpc_w)                      # fallback everywhere
    status = np.where(in_dom, "ok", "fallback")

    if in_dom.any():
        net, ck = _load()
        i = in_dom
        F = np.column_stack([
            T[i], P[i], ppr[i], tpr[i], 1.0 / tpr[i],
            np.log(np.clip(ppr[i], 1e-6, None)), ppr[i] / tpr[i],
            ppr[i] ** 2 / tpr[i] ** 3, gamma[i], x[i]]).astype(np.float32)
        with torch.no_grad():
            r = net(torch.tensor((F - ck["mu"]) / ck["sd"])).squeeze(1).numpy()
        z[i] = dak(ppr[i], tpr[i]) + r
    return z, status
