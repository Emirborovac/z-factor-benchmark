"""Publication figures - NNCF - Plotly/kaleido engine.

Why Plotly: automatic margin computation and outside-legend layout eliminate
the clipped-label failure class entirely; kaleido exports true vector PDF and
>=500 dpi PNG at exact millimetre sizes (Elsevier artwork spec).

Content rules carried over from the adversarial review:
  * AGA8-DETAIL appears in every comparison (it beats NNCF; hiding it is
    cherry-picking). Forest compares vs Hall-Yarborough (best correlation).
  * Regional chart reports agreement with the reference EOS, never "better".
  * AAD and tolerance shares only - no undefined "accuracy" metric.
  * One colour per method across all figures; NNCF is the hero.

Run:  python -X utf8 scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyaga8
import torch
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import COMPONENTS, TC_VEC, PC_VEC  # noqa: E402
from train_models_v2 import MW_VEC  # noqa: E402
from train_nn import Net, build_features  # noqa: E402
from convention_recovery import wichert_aziz  # noqa: E402
from eval_test_b import eos_predict  # noqa: E402
import zfactor.correlations as C  # noqa: E402

FIG = ROOT / "paper" / "figures"
X_COLS = [f"x_{c}" for c in COMPONENTS]
PX_PER_MM = 96 / 25.4          # kaleido logical pixels per millimetre

MODEL = "NNCF"
INK, SUBC, FAINT = "#1A1A1A", "#5A6169", "#A8ADB4"
GRIDC = "#E8EAEE"

COLOR = {MODEL: "#0F62FE", "GERG-2008": "#6929C4", "AGA8-DETAIL": "#00877E",
         "DAK": "#F1740A", "Hall–Yarborough": "#E8388A", "DPR": "#9A7B00"}

# bar palette: saturated blue reserved for NNCF; reference EOS muted;
# classical correlations in grays (hero-focus rule from design review)
BAR = {MODEL: "#0F62FE", "GERG-2008": "#8D6FC0", "AGA8-DETAIL": "#57A99F",
       "DPR": "#C7CCD1", "DAK": "#A9AFB6", "Hall–Yarborough": "#8C939B"}

LAB = {"10.1016/j.jct.2005.10.004": "Chamorro 2006",
       "10.1016/j.fluid.2016.08.002": "González-Pérez 2016",
       "10.1021/acs.jced.6b00137": "Tibaduiza 2016",
       "10.1021/acs.jced.7b01125": "Hernández-Gómez 2018",
       "10.1016/j.jct.2015.12.006": "Hernández-Gómez 2016a",
       "10.1016/j.jct.2016.05.024": "Hernández-Gómez 2016b",
       "10.1021/acs.jced.8b00433": "Liu 2018",
       "10.1016/j.jct.2016.03.035": "McLinden 2016",
       "10.1021/je500792x": "Richter 2014",
       "10.1016/j.jct.2012.12.018": "Li 2013"}

# ---- base template ----------------------------------------------------------
pio.templates["paper"] = go.layout.Template(layout=dict(
    font=dict(family="Arial", size=15, color=INK),
    paper_bgcolor="white", plot_bgcolor="white",
    title=dict(x=0.02, xanchor="left", yref="container", y=0.955,
               yanchor="top",
               font=dict(size=19, family="Arial", color=INK)),
    xaxis=dict(showline=True, linecolor="#C3C8CF", linewidth=1.2, ticks="",
               gridcolor=GRIDC, gridwidth=1, zeroline=False,
               title_font=dict(size=15.5), tickfont=dict(size=14)),
    yaxis=dict(showline=False, ticks="", gridcolor=GRIDC, gridwidth=1,
               zeroline=False, title_font=dict(size=15.5),
               tickfont=dict(size=14)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13.5)),
    margin=dict(l=70, r=45, t=28, b=66),
))
pio.templates.default = "paper"


def title(main, sub=None):
    """Elsevier artwork must not contain titles or subtitles - all explanation
    belongs in the figure caption. Retained as a no-op so each figure still
    documents, in code, what its caption must say."""
    return dict(text="")


def save(fig, name, w_mm=185, h_mm=110):
    FIG.mkdir(parents=True, exist_ok=True)
    w, h = int(w_mm * PX_PER_MM), int(h_mm * PX_PER_MM)
    fig.write_image(str(FIG / f"{name}.pdf"), width=w, height=h)
    fig.write_image(str(FIG / f"{name}.png"), width=w, height=h, scale=6)
    print(f"  {name}")


def aad(y, p):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean(np.abs(p[m] - y[m]) / np.abs(y[m])) * 100)


def within(y, p, tol):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean(np.abs(p[m] - y[m]) / np.abs(y[m]) * 100 <= tol) * 100)


def bin_median(x, v, edges, min_n=15):
    c, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b) & np.isfinite(v)
        if m.sum() < min_n:
            continue
        c.append(np.sqrt(a * b) if a > 0 else 0.5 * b)
        med.append(np.median(v[m]))
    return np.asarray(c), np.asarray(med)


def bin_band(x, v, edges, q=(10, 50, 90), min_n=15):
    """Per-bin percentile band. A translucent point cloud is invisible once
    exported at figure scale, so the spread is drawn as an explicit ribbon."""
    c, out = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b) & np.isfinite(v)
        if m.sum() < min_n:
            continue
        c.append(0.5 * (a + b))
        out.append(np.percentile(v[m], q))
    return np.asarray(c), np.asarray(out)


# --------------------------------------------------------------------------- #
def load_everything():
    t = pd.read_parquet(ROOT / "data" / "processed" / "test_b.parquet")
    y = t.z.to_numpy()
    x = t[X_COLS].to_numpy()
    gamma = (x @ MW_VEC) / 28.9647
    tpc, ppc = x @ TC_VEC, x @ PC_VEC
    tpcw, ppcw = wichert_aziz(tpc, ppc, t.x_co2.to_numpy(), t.x_h2s.to_numpy())
    P, T = t.P_MPa.to_numpy(), t.T_K.to_numpy()
    ppr, tpr = P / ppc, T / tpc

    ck = torch.load(ROOT / "models" / "nn" / "eos_nn_detail.pt",
                    weights_only=False)
    net = Net(ck["d_in"], ck["width"], ck["blocks"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    F = build_features(T, P, ppr, tpr, x, gamma)
    with torch.no_grad():
        r = net(torch.tensor((F - ck["mu"]) / ck["sd"])).squeeze(1).numpy()

    gerg = eos_predict(pyaga8.Gerg2008, t)
    det = eos_predict(pyaga8.Detail, t)
    susp = [d for d, g in t.groupby("doi")
            if np.nanmedian(np.abs(gerg[g.index.to_numpy()] - y[g.index.to_numpy()])
                            / y[g.index.to_numpy()]) * 100 > 10]
    keep = (~t.doi.isin(susp)).to_numpy()
    ng = keep & (t.x_methane >= 0.5).to_numpy() & (tpr <= 3.0) & (T <= 500)

    preds = {MODEL: C.dak(ppr, tpr) + r, "GERG-2008": gerg, "AGA8-DETAIL": det,
             "DAK": C.dak(P / ppcw, T / tpcw),
             "Hall–Yarborough": C.hall_yarborough(P / ppcw, T / tpcw),
             "DPR": C.dpr(P / ppcw, T / tpcw)}
    return dict(t=t, y=y, P=P, T=T, ppr=ppr, tpr=tpr, ng=ng, preds=preds,
                net=net, ck=ck)


ALL6 = ["AGA8-DETAIL", "GERG-2008", MODEL, "DPR", "DAK", "Hall–Yarborough"]


# ------------------------------------------------------------------ FIG 1 ---
def fig1_leaderboard(D):
    y, ng, preds = D["y"], D["ng"], D["preds"]
    yn = y[ng]
    vals = sorted([(k, aad(yn, preds[k][ng])) for k in ALL6],
                  key=lambda kv: kv[1], reverse=True)
    fig = go.Figure()
    for k, v in vals:
        hero = k == MODEL
        fig.add_trace(go.Bar(
            y=[k], x=[v], orientation="h",
            marker=dict(color=BAR[k]),
            width=0.62,
            text=[f"<b>{v:.3f} %</b>" if hero else f"{v:.3f} %"],
            textposition="outside", textfont=dict(size=14.5, color=INK),
            cliponaxis=False, showlegend=False))
    fig.update_layout(
        title=title("Average absolute deviation on laboratory data",
                    "2,426 measurements · 10 independent laboratories · natural-gas "
                    "domain of Test Set B<br>lower is better · NNCF and "
                    "AGA8-DETAIL are statistically indistinguishable"),
        xaxis=dict(title="average absolute deviation (%)", range=[0, 2.35],
                   showgrid=True),
        yaxis=dict(showgrid=False,
                   tickfont=dict(size=14.5),
                   ticklabelstandoff=12),
        bargap=0.32, margin=dict(l=150, r=80, t=26, b=66))
    fig.update_yaxes(tickmode="array", tickvals=[k for k, _ in vals],
                     ticktext=[f"<b>{k}</b>" if k == MODEL else k
                               for k, _ in vals])
    save(fig, "fig1_leaderboard", 185, 105)


# ------------------------------------------------------------------ FIG 2 ---
def fig2_tolerance(D):
    y, ng, preds = D["y"], D["ng"], D["preds"]
    yn = y[ng]
    tols = [0.5, 1.0, 2.0]
    fig = go.Figure()
    for k in ALL6:
        sh = [within(yn, preds[k][ng], tl) for tl in tols]
        fig.add_trace(go.Bar(
            name=k, x=[f"within ± {tl:g} %" for tl in tols], y=sh,
            marker=dict(color=BAR[k]),
            text=[f"{s:.0f}" for s in sh], textposition="outside",
            textfont=dict(size=12, color=SUBC), cliponaxis=False))
    fig.update_layout(
        title=title("Share of predictions within tolerance",
                    "how often each method lands within ± 0.5 / 1 / 2 % of the "
                    "measured z · n = 2,426 measurements"),
        # value labels are as wide as the bars, so the within-group gap has to
        # be large enough to keep neighbouring labels apart where three
        # methods share a value (the 71/71/71 and 40/40/40 triplets)
        barmode="group", bargroupgap=0.20, bargap=0.24,
        yaxis=dict(title="share of predictions (%)", range=[0, 108],
                   tickvals=[0, 25, 50, 75, 100], showgrid=True),
        xaxis=dict(showgrid=False, tickfont=dict(size=15)),
        # legend placed vertically at the right so its top-to-bottom order is
        # the same as the bars' left-to-right order. A wrapped horizontal
        # legend fills column-major and silently disagrees with the bars.
        legend=dict(orientation="v", x=1.015, xanchor="left", y=0.98,
                    yanchor="top", traceorder="normal"),
        margin=dict(l=70, r=178, t=26, b=66))
    save(fig, "fig2_tolerance", 185, 100)


# ------------------------------------------------------------------ FIG 3 ---
def fig3_forest(D):
    t, y, ng, preds = D["t"], D["y"], D["ng"], D["preds"]
    yn = y[ng]
    src = t.doi.to_numpy()[ng]
    uniq = np.unique(src)
    a_nn = np.abs(preds[MODEL][ng] - yn) / yn * 100
    a_bc = np.abs(preds["DPR"][ng] - yn) / yn * 100
    rng = np.random.default_rng(3)
    rows = []
    for s in uniq:
        m = src == s
        d = a_bc[m] - a_nn[m]
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                       for _ in range(3000)])
        rows.append((f"{LAB.get(s, s)}  ({int(m.sum())})", d.mean(),
                     *np.percentile(bs, [2.5, 97.5])))
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    mus = [r[1] for r in rows]
    los = [r[1] - r[2] for r in rows]
    his = [r[3] - r[1] for r in rows]
    fig = go.Figure(go.Scatter(
        x=mus, y=labels, mode="markers",
        marker=dict(color=COLOR[MODEL], size=11),
        error_x=dict(type="data", symmetric=False, array=his, arrayminus=los,
                     color="#9DBEFF", thickness=5, width=0),
        showlegend=False))
    fig.add_vline(x=0, line=dict(color=INK, width=1.4))
    fig.update_layout(
        title=title(f"{MODEL} error reduction per laboratory",
                    "versus DPR, the strongest classical correlation on this benchmark"
                    "<br>dots: mean · bars: 95 % bootstrap CI · "
                    "every laboratory improves"),
        xaxis=dict(title="percentage points of error removed",
                   range=[-0.15, 2.05], showgrid=True),
        yaxis=dict(showgrid=False, tickfont=dict(size=13.5),
                   ticklabelstandoff=10),
        margin=dict(l=245, r=45, t=26, b=66))
    save(fig, "fig3_forest", 190, 112)


# ------------------------------------------------------------------ FIG 4 ---
def fig4_pressure_profile(D):
    y, ng, preds, ppr = D["y"], D["ng"], D["preds"], D["ppr"]
    yn = y[ng]
    edges = np.geomspace(0.08, 6.0, 10)
    fig = go.Figure()
    for k in ALL6:
        ape = np.abs(preds[k][ng] - yn) / yn * 100
        cx, med = bin_median(ppr[ng], ape, edges)
        # every series solid: dashing carried no information and read as noise
        fig.add_trace(go.Scatter(
            x=cx, y=med, mode="lines", name=k,
            line=dict(color=COLOR[k], width=4.5 if k == MODEL else 2.6)))
    fig.update_layout(
        title=title("Median error across the pressure range",
                    "log–log · natural-gas domain of Test Set B"),
        xaxis=dict(title="reduced pressure  <i>P</i><sub>pr</sub>",
                   type="log", tickvals=[0.1, 0.3, 1, 3, 6], showgrid=True,
                   range=[np.log10(0.085), np.log10(7.4)]),
        yaxis=dict(title="median absolute error (%)", type="log",
                   # a tick at 2 keeps the upper quarter of the panel, where
                   # the correlations peak, from being unreferenced
                   tickvals=[0.03, 0.1, 0.3, 1, 2]),
        legend=dict(x=1.02, xanchor="left", y=0.5, yanchor="middle"),
        margin=dict(l=70, r=185, t=26, b=66))
    save(fig, "fig4_pressure_profile", 190, 115)


# ------------------------------------------------------------------ FIG 5 ---
def fig5_regional():
    # agreement with the TEACHING equation (AGA8-DETAIL), recomputed by
    # scripts/eval_regional_detail.py after the teacher swap
    vals = {"Trinidad": (0.009, 0.774, 1), "Nigeria": (0.013, 0.628, 1),
            "Qatar": (0.017, 0.644, 1), "Malaysia": (0.018, 0.587, 1),
            "Australia": (0.024, 0.528, 1), "Netherlands": (0.029, 0.758, 1366),
            "Kazakhstan": (0.041, 1.139, 1), "UAE": (0.103, 0.921, 2)}
    order = sorted(vals, key=lambda r: vals[r][0], reverse=True)
    labels = [f"{r}  ({vals[r][2]})" for r in order]
    zn = [vals[r][0] for r in order]
    dk = [vals[r][1] for r in order]
    fig = go.Figure()
    for lab_, a, b in zip(labels, zn, dk):
        fig.add_trace(go.Scatter(x=[a, b], y=[lab_, lab_], mode="lines",
                                 line=dict(color="#DFE3E8", width=3),
                                 showlegend=False, hoverinfo="skip"))
    tr_dak = (go.Scatter(x=dk, y=labels, mode="markers+text", name="DAK",
                             marker=dict(color=COLOR["DAK"], size=12),
                             text=[f"{v:.2f} %" for v in dk],
                             textposition="middle right",
                             textfont=dict(size=13, color=SUBC),
                             cliponaxis=False))
    tr_n = (go.Scatter(x=zn, y=labels, mode="markers+text", name=MODEL,
                             marker=dict(color=COLOR[MODEL], size=13),
                             text=[f"<b>{v:.3f} %</b>" for v in zn],
                             textposition="middle left",
                             textfont=dict(size=13, color=INK),
                             cliponaxis=False))
    fig.add_trace(tr_n[0] if isinstance(tr_n, tuple) else tr_n)
    fig.add_trace(tr_dak[0] if isinstance(tr_dak, tuple) else tr_dak)
    fig.update_layout(
        title=title("Deviation from GERG-2008 across producing regions",
                    "(n) = gas compositions per region · NNCF is trained toward "
                    "GERG-2008<br>this measures surrogate fidelity, not "
                    "accuracy vs experiment (see Fig. 1)"),
        xaxis=dict(title="mean deviation from AGA8-DETAIL (%)",
                   type="log", tickvals=[0.01, 0.03, 0.1, 0.3, 1],
                   range=[np.log10(0.0025), np.log10(2.6)], showgrid=True),
        yaxis=dict(showgrid=False, tickfont=dict(size=14),
                   ticklabelstandoff=10),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.04,
                    yanchor="bottom"),
        margin=dict(l=150, r=60, t=46, b=66))
    save(fig, "fig5_regional", 185, 108)


# ------------------------------------------------------------------ FIG 6 ---
def fig6_teacher_fidelity(D):
    """Fidelity to the TEACHING equation (AGA8-DETAIL), which is what a
    distillation claim must be made against. Named for the teacher, not for
    GERG: the file name is part of the released artefact."""
    ng, preds, ppr = D["ng"], D["preds"], D["ppr"]
    m = ng & (ppr <= 6.5)
    ref = preds["AGA8-DETAIL"]
    d = (preds[MODEL][m] - ref[m]) / ref[m] * 100
    CS = [[0.0, "rgba(255,255,255,0)"], [1e-9, "#EAF1FA"],
          [0.3, "#5C97CE"], [1.0, "#08306B"]]
    out_share = float(np.mean(np.abs(d) > 0.5) * 100)
    fig = go.Figure(go.Histogram2d(
        x=ppr[m], y=d, colorscale=CS,
        xbins=dict(start=0, end=6.5, size=0.07),
        ybins=dict(start=-0.5, end=0.5, size=0.015),
        showscale=True,
        colorbar=dict(title=dict(text="count", font=dict(size=13)),
                      thickness=13, len=0.7)))
    fig.add_hline(y=0, line=dict(color="#9AA0A8", width=1.2))
    for g in (0.25, -0.25):
        fig.add_hline(y=g, line=dict(color="#C2C8CF", width=1.0,
                                     dash="dash"))
    fig.update_layout(
        title=title(f"{MODEL} against its teaching equation",
                    f"AGA8-DETAIL · median |Δ| = "
                    f"{np.median(np.abs(d)):.3f} % · 95 % of points within "
                    f"± {np.percentile(np.abs(d), 95):.2f} %<br>dashed guides: "
                    f"± 0.25 % · {out_share:.1f} % of points beyond the "
                    f"± 0.5 % axis"),
        xaxis=dict(title="reduced pressure  <i>P</i><sub>pr</sub>",
                   showgrid=False),
        yaxis=dict(title="relative difference in <i>z</i> (%)", showgrid=False,
                   range=[-0.5, 0.5]),
        margin=dict(l=76, r=45, t=26, b=66))
    save(fig, "fig6_teacher_fidelity", 185, 100)


# ------------------------------------------------------------------ FIG 7 ---
def fig7_deviation(D):
    y, ng, preds, ppr = D["y"], D["ng"], D["preds"], D["ppr"]
    yn = y[ng]
    panels = [MODEL, "GERG-2008", "AGA8-DETAIL", "DAK",
              "Hall–Yarborough", "DPR"]
    edges = np.array([0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6])
    fig = make_subplots(
        rows=2, cols=3, shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.045, vertical_spacing=0.085,
        subplot_titles=[f"<b>{k}</b> — AAD {aad(yn, preds[k][ng]):.2f} %"
                        for k in panels])
    for i, k in enumerate(panels):
        r, c = i // 3 + 1, i % 3 + 1
        d = (preds[k][ng] - yn) / yn * 100
        cx, q = bin_band(ppr[ng], d, edges + 1e-9)
        rgb = tuple(int(COLOR[k][j:j + 2], 16) for j in (1, 3, 5))
        fig.add_trace(go.Scatter(          # P10-P90 spread, drawn as a ribbon
            x=np.concatenate([cx, cx[::-1]]),
            y=np.concatenate([q[:, 2], q[::-1, 0]]),
            fill="toself", mode="lines", line=dict(width=0),
            fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.20)",
            hoverinfo="skip", showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(          # median
            x=cx, y=q[:, 1], mode="lines",
            line=dict(color=COLOR[k], width=3.4),
            showlegend=False), row=r, col=c)
        fig.add_hline(y=0, line=dict(color="#9AA0A8", width=1.1),
                      row=r, col=c)
        for g in (-2, 2):
            fig.add_hline(y=g, line=dict(color="#D3D8DE", width=1,
                                         dash="dot"), row=r, col=c)
    fig.update_xaxes(range=[0, 6.2], showgrid=False, showline=False,
                     tickvals=[0, 2, 4, 6])
    fig.update_yaxes(range=[-4.4, 4.4], tickvals=[-4, -2, 0, 2, 4],
                     showgrid=True)
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="<i>P</i><sub>pr</sub>", row=2, col=c)
    for r in (1, 2):
        fig.update_yaxes(title_text="deviation (%)", row=r, col=1)
    fig.update_layout(
        title=title("Deviation from experiment across Test Set B",
                    "line: running median · band: P10–P90 · dotted: ± 2 % · "
                    "n = 2,426 · natural-gas domain"),
        margin=dict(l=70, r=45, t=30, b=66))
    fig.update_annotations(font=dict(size=14.5))
    save(fig, "fig7_deviation", 190, 122)


# ------------------------------------------------------------------ FIG 8 ---
def fig8_isotherms(D):
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    sk = m[m.tier == "chart_digitized"]
    net, ck = D["net"], D["ck"]
    xg = np.zeros(len(COMPONENTS))
    for c_, v_ in [("methane", .93), ("ethane", .04), ("propane", .01),
                   ("nitrogen", .01), ("co2", .01)]:
        xg[COMPONENTS.index(c_)] = v_
    tpc, ppc = xg @ TC_VEC, xg @ PC_VEC
    gam = (xg @ MW_VEC) / 28.9647
    fig = go.Figure()
    shown = False
    LAD = [1.05, 1.2, 1.4, 1.7, 2.4]
    for tpr_v in LAD:
        d = sk[np.isclose(sk.Tpr, tpr_v)].sort_values("Ppr")
        fig.add_trace(go.Scattergl(
            x=d.Ppr, y=d.z, mode="markers", name="digitised chart",
            marker=dict(color="#C8CDD4", size=2.6), showlegend=False))
        g = np.linspace(0.2, 15, 260)
        X = np.tile(xg, (len(g), 1))
        F = build_features(np.full_like(g, tpr_v * tpc), g * ppc, g,
                           np.full_like(g, tpr_v), X, np.full_like(g, gam))
        with torch.no_grad():
            r = net(torch.tensor((F - ck["mu"]) / ck["sd"])).squeeze(1).numpy()
        fig.add_trace(go.Scatter(
            x=g, y=C.dak(g, np.full_like(g, tpr_v)) + r, mode="lines",
            name=MODEL, line=dict(color=COLOR[MODEL], width=3),
            showlegend=False))
        # The curve ends are separated by >= 0.08 in z, which is legible at
        # 185 mm, so each isotherm is labelled directly at its own end. An
        # earlier ladder-plus-leader version was wrong: with showarrow=True
        # Plotly treats (x, y) as the arrow HEAD and (ax, ay) as the text, so
        # the leaders pointed away from the curves they named.
        y_end = float(C.dak(15.0, tpr_v))
        fig.add_annotation(x=15.35, y=y_end, xref="x", yref="y",
                           text=f"<i>T</i><sub>pr</sub> = {tpr_v:g}",
                           showarrow=False, xanchor="left",
                           font=dict(size=13, color=SUBC))
        shown = True
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             name="digitised chart",
                             marker=dict(color="#B7BDC5", size=9)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=MODEL,
                             line=dict(color=COLOR[MODEL], width=4)))
    fig.update_layout(
        title=title(f"{MODEL} against the digitised Standing–Katz chart",
                    "57,060 chart points carry 1–3 % inherent digitisation "
                    "uncertainty<br>NNCF drawn for a lean-gas composition"),
        xaxis=dict(title="reduced pressure  <i>P</i><sub>pr</sub>",
                   range=[0, 18.6], showgrid=True,
                   tickvals=[0, 5, 10, 15]),
        yaxis=dict(title="compressibility factor  <i>z</i>",
                   range=[0.2, 1.85]),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        margin=dict(l=70, r=60, t=26, b=66))
    save(fig, "fig8_isotherms", 185, 118)


# ------------------------------------------------------------------ FIG 9 ---
def fig9_model_families():
    items = [("Gradient boosting · raw (T, P, x)", 2.451),
             ("Gradient boosting · reduced coords", 2.231),
             ("Gradient boosting · broad pool", 2.143),
             ("Gradient boosting · focused pool", 1.853),
             (f"{MODEL} · taught by GERG-2008", 0.923),
             (f"{MODEL} · taught by AGA8-DETAIL", 0.789)]
    fig = go.Figure()
    for k, v in items:
        hero = k.endswith("AGA8-DETAIL")
        fig.add_trace(go.Bar(
            y=[k], x=[v], orientation="h",
            marker=dict(color=COLOR[MODEL] if hero else FAINT,
                        opacity=1 if hero else 0.65),
            width=0.6, showlegend=False))
        # value labels live in a clean column at the right, clear of the
        # dashed reference lines that would otherwise cross the text
        fig.add_annotation(x=1.005, xref="paper", y=k, xanchor="left",
                           text=f"<b>{v:.2f} %</b>" if hero else f"{v:.2f} %",
                           showarrow=False,
                           font=dict(size=14, color=INK if hero else SUBC))
    # Reference-line labels live ABOVE the plot area (never on the bars).
    # AGA8-DETAIL (0.769) and GERG-2008 (0.917) are close: anchor their
    # labels on opposite sides of their lines so they cannot collide.
    for xv, lab_, anch in [(1.919, "DAK", "left"),
                           (0.917, "GERG-2008", "left"),
                           (0.769, "AGA8-DETAIL", "right")]:
        fig.add_vline(x=xv, line=dict(color=COLOR.get(lab_, SUBC), width=1.6,
                                      dash="dash"))
        fig.add_annotation(x=xv, yref="paper", y=1.02, yanchor="bottom",
                           xanchor=anch, text=lab_, showarrow=False,
                           font=dict(size=13, color=COLOR.get(lab_, SUBC)))
    fig.update_layout(
        title=title("Model families on the laboratory benchmark",
                    "identical states, architecture and seed · only the model family "
                    "or the teaching reference equation changes<br>dashed: "
                    "reference-equation and correlation accuracy levels"),
        xaxis=dict(title="average absolute deviation (%)", range=[0, 2.85],
                   showgrid=True),
        yaxis=dict(showgrid=False, tickfont=dict(size=14),
                   ticklabelstandoff=10, autorange="reversed"),
        margin=dict(l=250, r=110, t=40, b=66))
    save(fig, "fig9_model_families", 190, 112)


# ----------------------------------------------------------------- FIG 10 ---
def fig10_correction():
    f = ROOT / "reports" / "functional_form_search.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).dropna(subset=["LOSO_MAPE"])
    YLO, YHI = -1.15, 0.5
    fig = go.Figure()
    best, off_total = None, 0
    # Group medians, not means: a handful of functional forms diverge on the
    # held-out laboratory by four orders of magnitude, which makes the group
    # mean (~-1400 pp) meaningless and put it off the axis entirely.
    for bases, col, lab_, sym in [
            (["DAK", "HY"], COLOR["DAK"], "empirical correlations",
             "triangle-up"),
            (["AGA8-DETAIL", "GERG-2008", MODEL], "#5A7184",
             f"reference equations and {MODEL}", "circle")]:
        s = d[d.base.isin(bases)]
        gx = (s.base_MAPE - s.in_sample_MAPE).to_numpy()
        gy = (s.base_MAPE - s.LOSO_MAPE).to_numpy()
        n_up = int((gy > 0).sum())
        off_total += int(((gy < YLO) | (gy > YHI)).sum())
        md = float(np.median(gy))
        # The group statistics live in the LEGEND, not in floating annotations.
        # At these y-values every horizontal position is occupied by markers,
        # so an in-panel label necessarily lands on data or on a tick label.
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="markers",
            name=f"{lab_}<br>{n_up} of {len(s)} transfer · median "
                 + f"{md:+.2f}".replace("-", "−") + " pp",
            marker=dict(color=col, size=9, symbol=sym, opacity=0.8)))
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y",
                      y0=md, y1=md,
                      line=dict(color=col, width=1.8, dash="dot"))
        if bases[0] == "DAK":                     # the family that transfers
            j = int(np.argmax(gy))
            best = (float(gx[j]), float(gy[j]))
    fig.add_hline(y=0, line=dict(color=INK, width=1.4))
    if best:
        fig.add_annotation(x=best[0], y=best[1], ax=-52, ay=-30,
                           text="best transferring family", showarrow=True,
                           arrowhead=0, arrowwidth=1.1, arrowcolor=SUBC,
                           xanchor="right", font=dict(size=12.5, color=INK))
    # never truncate silently: say how many fits diverged off the axis
    fig.add_annotation(x=0.994, xref="paper", y=YLO, yref="y", yshift=6,
                       xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"{off_total} of {len(d)} fits diverge below this axis",
                       font=dict(size=11.5, color=FAINT))
    fig.update_layout(
        title=title("Fitted residual corrections: apparent vs realised gain",
                    "31 functional families x 6 base methods, "
                    "leave-one-laboratory-out"),
        xaxis=dict(title="apparent gain while fitting (percentage points)",
                   showgrid=True, range=[-0.19, 0.87]),
        yaxis=dict(title="realised gain on an unseen laboratory (pp)",
                   range=[YLO, YHI]),
        # legend inside the empty upper-left quadrant; two lines per entry
        legend=dict(orientation="v", x=0.012, xanchor="left", y=0.99,
                    yanchor="top", font=dict(size=12.5), tracegroupgap=4),
        margin=dict(l=86, r=45, t=26, b=66))
    save(fig, "fig10_correction", 185, 106)


# ----------------------------------------------------------------- FIG 11 ---
def fig11_domain(D):
    """Two panels, because two different sections need a coverage map and they
    are not the same map: (a) where the *measurements* are, which is what the
    benchmark section must show, and (b) where real gases put the deployed
    model relative to its declared window."""
    from zfactor.predict import predict_z
    b = pd.read_parquet(ROOT / "data" / "processed" / "regional_compositions.parquet")
    x = b[X_COLS].to_numpy()
    Tg = np.array([283.15, 298.15, 323.15, 353.15, 393.15])
    Pg = np.array([1., 3., 5., 7.5, 10., 15., 25., 40.])
    TT, PP = np.meshgrid(Tg, Pg, indexing="ij")
    grid = np.column_stack([TT.ravel(), PP.ravel()])
    X = np.repeat(x, len(grid), axis=0)
    T = np.tile(grid[:, 0], len(b))
    P = np.tile(grid[:, 1], len(b))
    _, status = predict_z(T, P, X)
    tpc, ppc = X @ TC_VEC, X @ PC_VEC
    ppr, tpr = P / ppc, T / tpc
    ok = status == "ok"
    CS = [[0.0, "rgba(255,255,255,0)"], [1e-9, "#EAF1FA"],
          [0.3, "#5C97CE"], [1.0, "#08306B"]]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.105,
                        subplot_titles=["<b>(a)</b>  benchmark measurements",
                                        "<b>(b)</b>  deployment on real gases"])

    # ---- (a) where the laboratory measurements are -------------------------
    t, ng = D["t"], D["ng"]
    bp, bt = D["ppr"][ng], D["tpr"][ng]
    sour = ((t.x_co2.to_numpy() + t.x_h2s.to_numpy()) > 0.02)[ng]
    for m, col, nm in [(~sour, "#9AA6B2", "sweet"), (sour, "#F1740A", "sour")]:
        fig.add_trace(go.Scatter(
            x=bp[m], y=bt[m], mode="markers", name=f"{nm} ({int(m.sum())})",
            marker=dict(color=col, size=4, opacity=0.55,
                        line=dict(width=0))), row=1, col=1)

    # ---- (b) where real gases put the deployed model -----------------------
    fig.add_trace(go.Histogram2d(
        x=ppr[ok], y=tpr[ok], colorscale=CS,
        xbins=dict(start=0, end=16.05, size=0.12),
        ybins=dict(start=0.98, end=3.08, size=0.025),
        zmax=30, showscale=True,
        colorbar=dict(title=dict(text="count", font=dict(size=12.5)),
                      thickness=11, len=0.62, x=1.005, xanchor="left",
                      y=0.46, yanchor="middle")), row=1, col=2)
    fig.add_shape(type="rect", x0=0.02, y0=1.0, x1=16, y1=3.05,
                  line=dict(color=COLOR[MODEL], width=2.2, dash="dash"),
                  row=1, col=2)

    fig.update_xaxes(title_text="reduced pressure  <i>P</i><sub>pr</sub>",
                     showgrid=True, type="log",
                     tickvals=[0.03, 0.1, 0.3, 1, 3, 10, 40],
                     range=[np.log10(0.022), np.log10(60)], row=1, col=1)
    fig.update_yaxes(title_text="reduced temperature  <i>T</i><sub>pr</sub>",
                     showgrid=True, range=[0.95, 3.15], row=1, col=1)
    fig.update_xaxes(title_text="reduced pressure  <i>P</i><sub>pr</sub>",
                     showgrid=False, range=[0, 17.6], row=1, col=2)
    fig.update_yaxes(showgrid=False, range=[0.95, 3.15], row=1, col=2)
    fig.update_layout(
        title=title("Coverage of the benchmark and of the deployed model"),
        legend=dict(x=0.005, xanchor="left", y=1.0, yanchor="top",
                    font=dict(size=12.5)),
        margin=dict(l=76, r=88, t=34, b=66))
    fig.update_annotations(font=dict(size=14))
    save(fig, "fig11_domain", 190, 92)


def fig0_graphical_abstract(D):
    """Graphical abstract. Elsevier: min 531 x 1328 px (h x w), must remain
    readable at 5 x 13 cm, so type is large and the content is one message."""
    y, ng, preds = D["y"], D["ng"], D["preds"]
    yn = y[ng]
    rows = [("AGA8-DETAIL", "reference equation"),
            (MODEL, "this work"),
            ("GERG-2008", "reference equation"),
            ("DAK", "classical correlation"),
            ("Hall–Yarborough", "classical correlation")]
    vals = [(k, sub, aad(yn, preds[k][ng])) for k, sub in rows]

    fig = go.Figure()
    for k, sub, v in vals:
        hero = k == MODEL
        fig.add_trace(go.Bar(
            y=[k], x=[v], orientation="h", width=0.55,
            marker=dict(color=COLOR[MODEL] if hero else
                        ("#9FA6AD" if "correlation" in sub else "#BFD6F5")),
            showlegend=False))
        fig.add_annotation(x=v, y=k, xanchor="left", xshift=10,
                           text=f"<b>{v:.2f} %</b>" if hero else f"{v:.2f} %",
                           showarrow=False,
                           font=dict(size=27, color=INK if hero else SUBC))
    # No title and no subtitle, same rule as every other figure. A graphical
    # abstract is displayed without a caption, so the context it genuinely
    # needs (what is measured, on how many points, which direction is better)
    # is carried by the axis title instead of a headline block.
    fig.update_layout(
        title=title(""),
        xaxis=dict(title=dict(
            text="average absolute deviation from experiment (%)",
            font=dict(size=23)),
            range=[0, 2.45], tickfont=dict(size=23), showgrid=True),
        yaxis=dict(showgrid=False, tickfont=dict(size=26),
                   ticklabelstandoff=14),
        bargap=0.20,
        margin=dict(l=330, r=150, t=34, b=96))
    save(fig, "fig0_graphical_abstract", 340, 108)


def main():
    print("loading ...")
    D = load_everything()
    print("figures:")
    fig0_graphical_abstract(D)
    fig1_leaderboard(D)
    fig2_tolerance(D)
    fig3_forest(D)
    fig4_pressure_profile(D)
    fig5_regional()
    fig6_teacher_fidelity(D)
    fig7_deviation(D)
    fig8_isotherms(D)
    fig9_model_families()
    fig10_correction()
    fig11_domain(D)
    print(f"\n-> {FIG}")


if __name__ == "__main__":
    main()
