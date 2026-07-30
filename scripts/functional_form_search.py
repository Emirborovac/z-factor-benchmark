"""Systematic functional-form search for a residual correction.

Question: does ANY functional form of the residual (z_lab - z_base)
generalize to an unseen laboratory? Searched over families, not hand-picks:

  1. Polynomial   : total-degree 1..6 in (Ppr, Tpr), ridge-regularized
  2. Gaussian RBF : k in {4,9,16,25} centers on a (Ppr,Tpr) grid, widths scanned
  3. Rational     : Pade-type (linear/linear, quad/linear) via least squares
  4. Spline       : natural cubic B-splines, tensor product, df scanned
  5. Physical     : departure-scaled families, nonlinear fit (scipy)
  6. Symbolic     : genetic programming over {+,-,*,/,sqrt,log,exp} (gplearn)
  7. Fourier      : low-order trigonometric basis in scaled coordinates

Protocol for every candidate: leave-one-source-out. Fit on 9 labs, score on
the 10th. Reported metric = out-of-sample MAPE over all held-out folds, plus
in-sample MAPE to expose overfitting. Applied to each base method as control.

Run:  python -X utf8 scripts/functional_form_search.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyaga8
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from train_models_v2 import COMP_FEATS, MW_VEC, comp_features  # noqa: E402
from convention_recovery import wichert_aziz  # noqa: E402
from eval_test_b import eos_predict  # noqa: E402
import zfactor.correlations as C  # noqa: E402

X_COLS = [f"x_{c}" for c in COMPONENTS]
V3 = ROOT / "models" / "v3"
V2 = ROOT / "models" / "v2"
SEED = 20260730


def mape(y, p):
    return float(np.mean(np.abs(p - y) / np.abs(y)) * 100)


# --------------------------------------------------------------------------- #
# candidate families: each returns fit(Xtr, rtr) -> predict(Xte) for residual r
# --------------------------------------------------------------------------- #
def make_poly(degree, alpha=1e-6):
    def build(Xtr, rtr):
        pf = PolynomialFeatures(degree, include_bias=True)
        A = pf.fit_transform(Xtr)
        mdl = Ridge(alpha=alpha, fit_intercept=False).fit(A, rtr)
        return lambda Xte: mdl.predict(pf.transform(Xte))
    return build


def make_rbf(k, width_scale, alpha=1e-6):
    def build(Xtr, rtr):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Z = (Xtr - mu) / sd
        side = int(round(np.sqrt(k)))
        qs = np.linspace(0.1, 0.9, side)
        centers = np.array([[np.quantile(Z[:, 0], a), np.quantile(Z[:, 1], b)]
                            for a in qs for b in qs])
        h = width_scale * (1.0 / max(side - 1, 1)) * 3.0

        def feat(Zx):
            d2 = ((Zx[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            return np.hstack([np.exp(-d2 / (2 * h * h)), np.ones((len(Zx), 1))])
        mdl = Ridge(alpha=alpha, fit_intercept=False).fit(feat(Z), rtr)
        return lambda Xte: mdl.predict(feat((Xte - mu) / sd))
    return build


def make_spline(df, alpha=1e-6):
    def build(Xtr, rtr):
        st = SplineTransformer(n_knots=df, degree=3, include_bias=True)
        A = st.fit_transform(Xtr)
        mdl = Ridge(alpha=alpha, fit_intercept=False).fit(A, rtr)
        return lambda Xte: mdl.predict(st.transform(Xte))
    return build


def make_rational(num_deg):
    """r = (a0 + a1*p + a2*t [+ a3*p^2]) / (1 + b1*p + b2*t) via curve_fit."""
    def model(X, *th):
        p, t = X
        if num_deg == 1:
            a0, a1, a2, b1, b2 = th
            num = a0 + a1 * p + a2 * t
        else:
            a0, a1, a2, a3, b1, b2 = th
            num = a0 + a1 * p + a2 * t + a3 * p * p
        return num / (1.0 + b1 * p + b2 * t)

    n_par = 5 if num_deg == 1 else 6

    def build(Xtr, rtr):
        p, t = Xtr[:, 0], Xtr[:, 1]
        try:
            th, _ = curve_fit(model, (p, t), rtr, p0=np.zeros(n_par) + 1e-3,
                              maxfev=20000)
        except Exception:
            th = np.zeros(n_par)
        return lambda Xte: model((Xte[:, 0], Xte[:, 1]), *th)
    return build


def make_physical(kind):
    """Departure-scaled nonlinear families fitted by least squares."""
    def f_exp(X, a, b, c):
        p, t = X
        return a * p * np.exp(-b * (t - 1.0)) + c

    def f_powt(X, a, b, c, d):
        p, t = X
        return a * p ** b / t ** c + d

    def f_gauss(X, a, mu_p, s_p, mu_t, s_t):
        p, t = X
        return a * np.exp(-((p - mu_p) ** 2 / (2 * s_p ** 2)
                            + (t - mu_t) ** 2 / (2 * s_t ** 2)))
    fns = {"exp_decay": (f_exp, [1e-3, 1.0, 0.0]),
           "power_law": (f_powt, [1e-3, 1.0, 1.0, 0.0]),
           "gaussian_bump": (f_gauss, [1e-2, 3.0, 2.0, 1.5, 0.5])}
    fn, p0 = fns[kind]

    def build(Xtr, rtr):
        try:
            th, _ = curve_fit(fn, (Xtr[:, 0], Xtr[:, 1]), rtr, p0=p0,
                              maxfev=20000)
        except Exception:
            th = np.array(p0) * 0
        return lambda Xte: fn((Xte[:, 0], Xte[:, 1]), *th)
    return build


def make_fourier(order, alpha=1e-6):
    def build(Xtr, rtr):
        mu, sd = Xtr.min(0), (Xtr.max(0) - Xtr.min(0)) + 1e-9

        def feat(X):
            Z = (X - mu) / sd
            cols = [np.ones(len(Z))]
            for k in range(1, order + 1):
                for j in range(2):
                    cols += [np.sin(np.pi * k * Z[:, j]),
                             np.cos(np.pi * k * Z[:, j])]
            return np.column_stack(cols)
        mdl = Ridge(alpha=alpha, fit_intercept=False).fit(feat(Xtr), rtr)
        return lambda Xte: mdl.predict(feat(Xte))
    return build


def make_symbolic(gens=12, pop=800):
    def build(Xtr, rtr):
        from gplearn.genetic import SymbolicRegressor
        n = min(len(Xtr), 4000)
        idx = np.random.default_rng(SEED).choice(len(Xtr), n, replace=False)
        sr = SymbolicRegressor(
            population_size=pop, generations=gens, tournament_size=20,
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "inv"),
            metric="mean absolute error", parsimony_coefficient=0.01,
            random_state=SEED, n_jobs=1, verbose=0)
        sr.fit(Xtr[idx], rtr[idx])
        return lambda Xte: sr.predict(Xte), sr
    def wrapped(Xtr, rtr):
        f, sr = build(Xtr, rtr)
        wrapped.last_expr = str(sr._program)
        return f
    wrapped.last_expr = ""
    return wrapped


def candidates(quick: bool):
    cands = {}
    for d in ([1, 2, 3] if quick else [1, 2, 3, 4, 5, 6]):
        cands[f"poly_deg{d}"] = make_poly(d)
    for k in ([9] if quick else [4, 9, 16, 25]):
        for w in ([1.0] if quick else [0.7, 1.0, 1.5]):
            cands[f"rbf_k{k}_w{w}"] = make_rbf(k, w)
    for df in ([5] if quick else [4, 5, 7, 9]):
        cands[f"spline_df{df}"] = make_spline(df)
    cands["rational_1_1"] = make_rational(1)
    cands["rational_2_1"] = make_rational(2)
    for kind in ["exp_decay", "power_law", "gaussian_bump"]:
        cands[f"phys_{kind}"] = make_physical(kind)
    for o in ([2] if quick else [1, 2, 3]):
        cands[f"fourier_o{o}"] = make_fourier(o)
    return cands


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-symbolic", action="store_true")
    args = ap.parse_args()

    t = pd.read_parquet(ROOT / "data" / "processed" / "test_b.parquet")
    y = t.z.to_numpy()
    x = t[X_COLS].to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    tpc_kay, ppc_kay = x @ TC_VEC, x @ PC_VEC
    tpc_kw, ppc_kw = wichert_aziz(tpc_kay, ppc_kay, t.x_co2.to_numpy(),
                                  t.x_h2s.to_numpy())
    P, T = t.P_MPa.to_numpy(), t.T_K.to_numpy()
    ppr, tpr = P / ppc_kay, T / tpc_kay

    base = {"DAK": C.dak(P / ppc_kw, T / tpc_kw),
            "HY": C.hall_yarborough(P / ppc_kw, T / tpc_kw),
            "GERG-2008": eos_predict(pyaga8.Gerg2008, t),
            "AGA8-DETAIL": eos_predict(pyaga8.Detail, t)}

    # NNCF itself must be a base method here. An earlier version of this search
    # only had the gradient-boosted residual model ("ML-hybrid(v2)") standing in
    # for "the surrogate", which is a different model and cannot support a claim
    # about NNCF.
    ck_f = ROOT / "models" / "nn" / "eos_nn_detail.pt"
    if ck_f.exists():
        import torch
        from train_nn import Net, build_features
        ck = torch.load(ck_f, weights_only=False)
        net = Net(ck["d_in"], ck["width"], ck["blocks"])
        net.load_state_dict(ck["state_dict"])
        net.eval()
        Fnn = build_features(T, P, ppr, tpr, x, gamma)
        with torch.no_grad():
            rnn = net(torch.tensor((Fnn - ck["mu"]) / ck["sd"])).squeeze(1).numpy()
        base["NNCF"] = C.dak(ppr, tpr) + rnn
    for key, label in [("native_direct", "ML-native"),
                       ("hybrid_reduced", "ML-hybrid")]:
        if (V3 / f"{key}_meta.json").exists():
            meta = json.loads((V3 / f"{key}_meta.json").read_text())
            fn = lgb.Booster(model_file=str(V3 / f"{key}_lgbm.txt")).predict
            cols = {"Ppr": ppr, "Tpr": tpr, "T_K": T, "P_MPa": P}
            cf = comp_features(x, gamma)
            for i, n in enumerate(COMP_FEATS):
                cols[n] = cf[:, i]
            for i, c in enumerate(X_COLS):
                cols[c] = x[:, i]
            X = np.column_stack([cols[f] for f in meta["features"]]).astype(np.float32)
            p = fn(X)
            base[label] = C.dak(ppr, tpr) + p if meta["residual"] else p
    if not any(k.startswith("ML") for k in base):
        b = lgb.Booster(model_file=str(V2 / "composition_lgbm.txt"))
        base["ML-hybrid(v2)"] = C.dak(ppr, tpr) + b.predict(
            np.column_stack([ppr, tpr, comp_features(x, gamma)]))

    gerg = base["GERG-2008"]
    suspect = [doi for doi, g in t.groupby("doi")
               if np.nanmedian(np.abs(gerg[g.index.to_numpy()] - y[g.index.to_numpy()])
                               / y[g.index.to_numpy()]) * 100 > 10]
    ng = ((~t.doi.isin(suspect)).to_numpy() & (t.x_methane >= 0.5).to_numpy()
          & (tpr <= 3.0) & (T <= 500))
    tn = t[ng].reset_index(drop=True)
    yn = y[ng]
    Xf = np.column_stack([ppr[ng], tpr[ng]])
    src = tn.doi.to_numpy()
    uniq = np.unique(src)
    print(f"natural-gas domain: n={len(tn)}, {len(uniq)} sources")

    cands = candidates(args.quick)
    if not args.no_symbolic:
        try:
            import gplearn  # noqa: F401
            cands["symbolic_gp"] = make_symbolic()
        except ImportError:
            print("(gplearn unavailable - symbolic regression skipped)")
    print(f"searching {len(cands)} functional forms x {len(base)} base methods "
          f"x {len(uniq)} folds\n")

    rows = []
    for bname, pall in base.items():
        z0 = pall[ng]
        r_true = yn - z0
        base_mape = mape(yn, z0)
        for cname, build in cands.items():
            oof = np.empty_like(z0)
            ins = []
            try:
                for s in uniq:
                    te = src == s
                    tr = ~te
                    f = build(Xf[tr], r_true[tr])
                    oof[te] = z0[te] + f(Xf[te])
                    ins.append(mape(yn[tr], z0[tr] + f(Xf[tr])))
                rows.append({"base": bname, "form": cname,
                             "base_MAPE": round(base_mape, 4),
                             "in_sample_MAPE": round(float(np.mean(ins)), 4),
                             "LOSO_MAPE": round(mape(yn, oof), 4),
                             "gain_pp": round(base_mape - mape(yn, oof), 4)})
            except Exception as e:
                rows.append({"base": bname, "form": cname,
                             "base_MAPE": round(base_mape, 4),
                             "in_sample_MAPE": np.nan, "LOSO_MAPE": np.nan,
                             "gain_pp": np.nan, "error": type(e).__name__})
        print(f"  {bname}: done")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "functional_form_search.csv", index=False)

    out = ["# Systematic functional-form search for a residual correction\n",
           "Every candidate fitted on 9 laboratories, scored on the 10th "
           "(leave-one-source-out). `in_sample_MAPE` exposes overfitting: "
           "forms that fit well in-sample but lose out-of-sample are learning "
           "lab-specific noise, not physics.\n",
           f"\nDomain: natural gas, n={len(tn)}, {len(uniq)} sources; "
           f"{len(cands)} functional forms tested.\n"]

    for bname in base:
        sub = df[df.base == bname].dropna(subset=["LOSO_MAPE"]) \
            .sort_values("LOSO_MAPE")
        best = sub.iloc[0]
        out.append(f"\n## Base: {bname} (uncorrected {best.base_MAPE}%)\n")
        out.append(sub.head(8).drop(columns="base").to_markdown(index=False))
        verdict = ("IMPROVES" if best.LOSO_MAPE < best.base_MAPE
                   else "NO FORM IMPROVES")
        out.append(f"\n**{verdict}** - best out-of-sample form: `{best.form}` "
                   f"at {best.LOSO_MAPE}% ({best.gain_pp:+.4f} pp)\n")

    if "symbolic_gp" in cands and hasattr(cands["symbolic_gp"], "last_expr"):
        out.append(f"\nLast symbolic expression found: "
                   f"`{cands['symbolic_gp'].last_expr}`\n")

    txt = "\n".join(out)
    (ROOT / "reports" / "functional_form_search.md").write_text(txt, encoding="utf-8")
    print("\n" + txt[-2500:])


if __name__ == "__main__":
    main()
