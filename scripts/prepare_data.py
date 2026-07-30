"""Data preparation pipeline for the Z-factor benchmark.

Reads immutable sources from data/raw/, cleans and labels each source into
data/interim/, then unifies everything into data/processed/master.parquet (+csv).
Writes a QC report to reports/qc_report.md.

Run:  python -X utf8 scripts/prepare_data.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

R_GAS = 8.314462618  # J/(mol K)

# canonical component order (matches AGA8 21-component slate)
COMPONENTS = [
    "methane", "nitrogen", "co2", "ethane", "propane", "isobutane", "n_butane",
    "isopentane", "n_pentane", "hexane", "heptane", "octane", "nonane", "decane",
    "hydrogen", "oxygen", "co", "water", "h2s", "helium", "argon",
]
X_COLS = [f"x_{c}" for c in COMPONENTS]

# header names used in the NIST composition sheets -> canonical
NIST_COMP_MAP = {
    "methane": "methane", "nitrogen": "nitrogen", "co2": "co2", "ethane": "ethane",
    "propane": "propane", "ibutane": "isobutane", "isobutan": "isobutane",
    "butane": "n_butane", "ipentane": "isopentane", "pentane": "n_pentane",
    "hexane": "hexane", "heptane": "heptane", "octane": "octane", "nonane": "nonane",
    "decane": "decane", "hydrogen": "hydrogen", "oxygen": "oxygen", "co": "co",
    "water": "water", "h2s": "h2s", "helium": "helium", "argon": "argon",
}

MASTER_COLS = (
    ["record_id", "source_id", "tier", "role", "gas_class", "region",
     "T_K", "P_MPa", "rho_mol_l", "Ppr", "Tpr", "ppr_tpr_origin", "z", "z_method",
     "sour", "mix_id", "reference", "quality_flag",
     "track_chart", "track_composition", "split"]
    + X_COLS
)

# critical temperature (K) and pressure (MPa) per component, for Kay's-rule
# pseudo-criticals on composition-domain records
CRIT_PROPS = {
    "methane": (190.564, 4.5992), "nitrogen": (126.192, 3.3958),
    "co2": (304.128, 7.3773), "ethane": (305.322, 4.8722),
    "propane": (369.825, 4.2512), "isobutane": (407.817, 3.6290),
    "n_butane": (425.125, 3.7960), "isopentane": (460.35, 3.3780),
    "n_pentane": (469.70, 3.3675), "hexane": (507.60, 3.0250),
    "heptane": (540.13, 2.7360), "octane": (568.74, 2.4970),
    "nonane": (594.55, 2.2810), "decane": (617.70, 2.1030),
    "hydrogen": (33.145, 1.2964), "oxygen": (154.581, 5.0430),
    "co": (132.86, 3.4940), "water": (647.096, 22.0640),
    "h2s": (373.10, 9.0000), "helium": (5.1953, 0.2276),
    "argon": (150.687, 4.8630),
}
TC_VEC = np.array([CRIT_PROPS[c][0] for c in COMPONENTS])
PC_VEC = np.array([CRIT_PROPS[c][1] for c in COMPONENTS])

qc_lines: list[str] = []


def qc(msg: str) -> None:
    print(msg)
    qc_lines.append(msg)


def record_ids(df: pd.DataFrame, source: str) -> pd.Series:
    """Deterministic id from source + row content (stable across reruns)."""
    key = df.astype(str).agg("|".join, axis=1)
    return key.map(lambda s: source + "_" + hashlib.sha1(s.encode()).hexdigest()[:12])


def classify_region(ppr: pd.Series, tpr: pd.Series) -> pd.Series:
    out = pd.Series("unclassified", index=ppr.index, dtype=object)
    m = ppr.notna() & tpr.notna()
    out[m & (ppr <= 0.2)] = "near_ideal"
    out[m & (ppr > 0.2) & (tpr < 1.2) & (ppr <= 6)] = "near_critical_trough"
    out[m & (ppr > 0.2) & (tpr >= 1.2) & (ppr <= 8)] = "moderate"
    out[m & (ppr > 8) & (ppr <= 15)] = "high_pressure"
    out[m & (ppr > 15)] = "ultra_high_pressure"
    # correction: trough label only applies below Ppr 6; Tpr<1.2 above 6 is high pressure
    out[m & (tpr < 1.2) & (ppr > 6) & (ppr <= 15)] = "high_pressure"
    return out


def classify_gas(xrow: np.ndarray) -> str:
    n = int((xrow > 1e-6).sum())
    if n <= 1:
        return "pure"
    if n == 2:
        return "binary"
    if n <= 4:
        return "multicomponent"
    return "natural_gas"


# --------------------------------------------------------------------------- #
# 1. Standing-Katz digitized chart
# --------------------------------------------------------------------------- #
def prepare_standing_katz() -> pd.DataFrame:
    df = pd.read_csv(RAW / "zFactor.DL" / "extras" / "tidy_SK.csv")
    n0 = len(df)
    df = df.rename(columns={"z": "z"})
    df = df.drop_duplicates(subset=["Tpr", "Ppr", "z"])
    dupes = n0 - len(df)

    bad = df[(df["z"] <= 0) | (df["z"] > 2.5) | (df["Ppr"] < 0) | (df["Ppr"] > 15.2)
             | (df["Tpr"] < 1.0) | (df["Tpr"] > 3.0)]
    df = df.drop(bad.index)

    out = pd.DataFrame(index=df.index)
    out["Ppr"] = df["Ppr"]
    out["Tpr"] = df["Tpr"]
    out["z"] = df["z"]
    out["source_id"] = "sk_chart"
    out["tier"] = "chart_digitized"
    out["role"] = "train"
    out["gas_class"] = "chart_generic"
    out["z_method"] = "digitized_chart"
    out["mix_id"] = "SK_Tpr_" + df["Tpr"].astype(str)
    out["reference"] = "Standing & Katz 1942 chart, digitized by Reyes (zFactor.DL)"
    out["quality_flag"] = "ok"
    qc(f"- Standing-Katz: {n0} raw -> {len(out)} clean "
       f"({dupes} exact duplicates, {len(bad)} out-of-bounds removed)")
    return out


# --------------------------------------------------------------------------- #
# 2. NIST AGA8 test data (EOS-calculated, code-validation only)
# --------------------------------------------------------------------------- #
def load_nist_compositions() -> pd.DataFrame:
    raw = pd.read_excel(RAW / "AGA8" / "TESTDATA" / "NG Compositions.XLS",
                        sheet_name=0, header=None)
    headers = [NIST_COMP_MAP[str(h).strip().lower()] for h in raw.iloc[0]]
    comp = raw.iloc[1:].copy()
    comp.columns = headers
    comp = comp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # per the sheet footer: "Gases are identified by the row number, and start
    # at 2 and end at 201. Row #1 is the table header." -> Gas # == Excel row
    comp.index = range(2, len(comp) + 2)
    comp = comp[comp.sum(axis=1) > 0]  # drop empty trailing rows
    frac = comp / 100.0
    sums = frac.sum(axis=1)
    bad = sums[(sums < 0.98) | (sums > 1.02)]
    if len(bad):
        qc(f"  WARNING: {len(bad)} NIST compositions do not sum to ~1 "
           f"(renormalized): {dict(bad.round(3))}")
    frac = frac.div(sums, axis=0)  # renormalize
    qc(f"  NIST compositions loaded: {len(frac)} gases (#"
       f"{frac.index.min()}-#{frac.index.max()})")
    frac.columns = [f"x_{c}" for c in frac.columns]
    return frac


def prepare_nist() -> pd.DataFrame:
    xls = pd.ExcelFile(RAW / "AGA8" / "TESTDATA" / "Test Data.xls")
    comps = load_nist_compositions()

    frames = []

    # --- Natural Gas Calculations: Gas# | T | d | GERG P... | DETAIL P...
    ng = xls.parse("Natural Gas Calculations", header=None)
    ng = ng.iloc[11:, [0, 1, 2, 4, 9]]
    ng.columns = ["gas_no", "T_K", "rho_mol_l", "P_gerg_MPa", "P_detail_MPa"]
    ng = ng.apply(pd.to_numeric, errors="coerce").dropna(subset=["T_K", "rho_mol_l"])
    ng["gas_no"] = ng["gas_no"].ffill().astype(int)
    ng["mix_id"] = "NIST_NG_" + ng["gas_no"].astype(str)
    ng = ng.join(comps, on="gas_no")
    unmatched = ng[ng[comps.columns[0]].isna()]["gas_no"].unique()
    if len(unmatched):
        qc(f"  NOTE: {int((ng[comps.columns[0]].isna()).sum())} NG rows reference "
           f"gas numbers with no composition ({list(unmatched)}); kept, flagged no_composition")
    frames.append(("natural_gas_sheet", ng))

    # --- Binary Mixture Calculations: comp1 | comp2 | T | d | GERG P | DETAIL P
    bi = xls.parse("Binary Mixture Calculations", header=None)
    bi = bi.iloc[11:, [0, 1, 2, 3, 5, 10]]
    bi.columns = ["c1", "c2", "T_K", "rho_mol_l", "P_gerg_MPa", "P_detail_MPa"]
    bi[["c1", "c2"]] = bi[["c1", "c2"]].ffill()
    num = bi[["T_K", "rho_mol_l", "P_gerg_MPa", "P_detail_MPa"]].apply(
        pd.to_numeric, errors="coerce")
    bi = pd.concat([bi[["c1", "c2"]], num], axis=1).dropna(subset=["T_K", "rho_mol_l"])
    bi["mix_id"] = ("NIST_BIN_" + bi["c1"].astype(str).str.strip() + "+"
                    + bi["c2"].astype(str).str.strip())
    for xc in X_COLS:
        bi[xc] = 0.0
    for cname, col in (("c1", None), ("c2", None)):
        canon = bi[cname].astype(str).str.strip().str.lower().map(NIST_COMP_MAP)
        for canon_name in canon.dropna().unique():
            bi.loc[canon == canon_name, f"x_{canon_name}"] += 0.5
    frames.append(("binary_sheet", bi))

    outs = []
    for label, df in frames:
        n0 = len(df)
        # long format: one row per (point, method)
        for method, pcol in [("gerg2008", "P_gerg_MPa"), ("aga8_detail", "P_detail_MPa")]:
            d = df.dropna(subset=[pcol]).copy()
            # Z = P / (rho R T);  P MPa -> Pa (1e6), rho mol/l -> mol/m3 (1e3)
            d["z"] = d[pcol] * 1e6 / (d["rho_mol_l"] * 1e3 * R_GAS * d["T_K"])
            d["P_MPa"] = d[pcol]
            d["z_method"] = method
            outs.append(d)
        qc(f"- NIST {label}: {n0} (T,rho) states -> {2 * n0} (point,method) rows")

    nist = pd.concat(outs, ignore_index=True)
    nist["source_id"] = "nist_aga8"
    nist["tier"] = "eos_calculated"
    # saturation-boundary states incl. liquid phase -> never train on these
    nist["role"] = "code_validation"
    nist["reference"] = "NIST usnistgov/AGA8 TESTDATA (REFPROP 9.1-based saturation states)"
    xmat = nist[X_COLS].fillna(0.0).to_numpy()
    nist["gas_class"] = [classify_gas(r) for r in xmat]
    # NIST's own note: DETAIL is invalid in the liquid/critical region; some
    # states also return P=0 (out of range). Flag rather than delete.
    nist["quality_flag"] = "ok"
    nist.loc[nist[X_COLS].fillna(0.0).sum(axis=1) < 0.99, "quality_flag"] = "no_composition"
    nist.loc[(nist["P_MPa"] <= 0) | (nist["z"] <= 0), "quality_flag"] = "eos_failed"
    nist.loc[(nist["quality_flag"] == "ok") & (nist["z"] > 3), "quality_flag"] = "eos_out_of_range"
    nbad = dict(nist["quality_flag"].value_counts())
    qc(f"- NIST combined: {len(nist)} rows; flags: {nbad} "
       f"(liquid/critical-region states kept, role=code_validation)")
    return nist


# --------------------------------------------------------------------------- #
# 3. Zenodo experimental databank (Bougha et al. 2026, CC-BY-4.0)
# --------------------------------------------------------------------------- #
def load_zenodo_compositions() -> pd.DataFrame:
    s2 = pd.read_excel(RAW / "zenodo_bougha2026" / "Z_factor_data_Natural_Gas.xlsx",
                       sheet_name="Table S2", header=None)
    # component names in column 1, formulas col 2, one 3-wide block per mix
    name_col = s2.iloc[3:, 1].astype(str).str.strip().str.lower()
    zen_map = {
        "hydrogen": "hydrogen", "hydrogen sulphide": "h2s", "carbon dioxide": "co2",
        "nitrogen": "nitrogen", "methane": "methane", "ethane": "ethane",
        "propane": "propane", "isobutane": "isobutane", "n-butane": "n_butane",
        "isopentane": "isopentane", "n-pentane": "n_pentane", "hexane": "hexane",
        "heptanes plus": "heptane", "heptane": "heptane", "helium": "helium",
        "water": "water", "oxygen": "oxygen", "carbon monoxide": "co",
        "octane": "octane", "nonane": "nonane", "decane": "decane", "argon": "argon",
    }
    rows = {}
    for c in range(s2.shape[1]):
        v = s2.iat[1, c]
        if isinstance(v, str) and v.strip().replace(" ", "").startswith("Mix"):
            mix = v.strip().replace(" ", "")
            vals = pd.to_numeric(s2.iloc[3:, c], errors="coerce")
            comp = {}
            for name, val in zip(name_col, vals):
                canon = zen_map.get(name)
                if canon and pd.notna(val):
                    comp[f"x_{canon}"] = comp.get(f"x_{canon}", 0.0) + float(val)
            rows[mix] = comp
    comp_df = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0) / 100.0
    sums = comp_df.sum(axis=1)
    ok = (sums > 0.98) & (sums < 1.02)
    if (~ok).any():
        qc(f"  WARNING: Zenodo mixes with composition sum far from 1: "
           f"{dict(sums[~ok].round(3))} (kept, renormalized)")
    comp_df = comp_df.div(sums, axis=0)
    return comp_df


def prepare_experimental() -> pd.DataFrame:
    s1 = pd.read_excel(RAW / "zenodo_bougha2026" / "Z_factor_data_Natural_Gas.xlsx",
                       sheet_name="Table S1", header=1)
    s1.columns = ["reference", "mix_id", "Ppr", "Tpr", "z"]
    n0 = len(s1)
    s1[["reference", "mix_id"]] = s1[["reference", "mix_id"]].ffill()
    s1["mix_id"] = s1["mix_id"].astype(str).str.replace(" ", "", regex=False)
    s1 = s1.dropna(subset=["Ppr", "Tpr", "z"])

    # physical-bounds cleaning
    bad = s1[(s1["Tpr"] < 0.9) | (s1["Tpr"] > 3.5) | (s1["Ppr"] <= 0)
             | (s1["Ppr"] > 31) | (s1["z"] <= 0.2) | (s1["z"] > 2.5)]
    if len(bad):
        qc(f"- Zenodo experimental: dropping {len(bad)} physically implausible rows: "
           f"{bad[['Ppr', 'Tpr', 'z']].round(3).to_dict('records')}")
    s1 = s1.drop(bad.index)
    ndup = int(s1.duplicated(subset=["mix_id", "Ppr", "Tpr", "z"]).sum())
    s1 = s1.drop_duplicates(subset=["mix_id", "Ppr", "Tpr", "z"])

    comps = load_zenodo_compositions()
    s1 = s1.join(comps, on="mix_id")
    matched = int(s1[X_COLS[0]].notna().sum()) if X_COLS[0] in s1 else 0

    s1["source_id"] = "zenodo_bougha2026"
    s1["tier"] = "experimental"
    s1["role"] = "test"
    s1["z_method"] = "measured"
    have_x = s1[[c for c in X_COLS if c in s1.columns]].notna().any(axis=1)
    xmat = s1.reindex(columns=X_COLS).fillna(0.0).to_numpy()
    s1["gas_class"] = [classify_gas(r) if h else "unknown"
                       for r, h in zip(xmat, have_x)]
    s1["quality_flag"] = "ok"
    qc(f"- Zenodo experimental: {n0} raw -> {len(s1)} clean "
       f"({ndup} duplicates removed); {matched} rows joined to a composition")
    return s1


# --------------------------------------------------------------------------- #
def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    qc("# Z-factor benchmark - data preparation QC report\n")
    qc("## Per-source cleaning\n")
    sk = prepare_standing_katz()
    nist = prepare_nist()
    exp = prepare_experimental()

    parts = []
    for name, df in [("standing_katz", sk), ("nist_aga8", nist), ("experimental", exp)]:
        df = df.reindex(columns=[c for c in MASTER_COLS if c != "record_id"])
        df.insert(0, "record_id", record_ids(df, name))
        df[X_COLS] = df[X_COLS].astype(float).fillna(np.nan)
        df.to_parquet(INTERIM / f"{name}.parquet", index=False)
        parts.append(df)

    master = pd.concat(parts, ignore_index=True)
    master["sour"] = (master[["x_h2s", "x_co2"]].fillna(0.0).sum(axis=1) > 0.02)

    # fill missing Ppr/Tpr from composition via Kay's rule (composition-domain
    # records); records that shipped with reduced coordinates keep them
    master["ppr_tpr_origin"] = np.where(master["Ppr"].notna(), "as_published", "none")
    xmat = master[X_COLS].fillna(0.0).to_numpy()
    have_x = xmat.sum(axis=1) > 0.99
    tpc = xmat @ TC_VEC
    ppc = xmat @ PC_VEC
    fill = master["Ppr"].isna() & have_x & master["P_MPa"].notna()
    master.loc[fill, "Ppr"] = master.loc[fill, "P_MPa"] / ppc[fill]
    master.loc[fill, "Tpr"] = master.loc[fill, "T_K"] / tpc[fill]
    master.loc[fill, "ppr_tpr_origin"] = "kay_rule"
    qc(f"\n- Ppr/Tpr filled via Kay's rule for {int(fill.sum())} composition-domain records")

    master["region"] = classify_region(master["Ppr"], master["Tpr"])

    # ---- model-track eligibility -------------------------------------------
    # chart model: inputs (Ppr, Tpr) -> needs published reduced coordinates
    #   (kay_rule coords are derived labels, not measured inputs)
    # composition model: inputs (P, T, x_i) -> needs absolute state + composition
    has_comp = master[X_COLS].fillna(0.0).sum(axis=1) > 0.99
    master["track_chart"] = master["ppr_tpr_origin"].eq("as_published") & master["Ppr"].notna()
    master["track_composition"] = has_comp & master["P_MPa"].notna() & master["T_K"].notna()
    # experimental rows have composition but only reduced coordinates published;
    # they can serve the composition track via reconstructed P,T (flagged deriv.)
    exp_recon = (master["tier"] == "experimental") & has_comp
    master["track_composition"] = master["track_composition"] | exp_recon

    # ---- train/val/test split ----------------------------------------------
    # deterministic hash split so reruns never reshuffle:
    #   - experimental tier is always test (untouchable)
    #   - NIST tier is code_validation (never a model split)
    #   - SK chart: 90/10 train/val by record_id hash
    def bucket(rid: str) -> str:
        return "val" if int(hashlib.sha1(rid.encode()).hexdigest(), 16) % 10 == 0 else "train"

    master["split"] = "none"
    sk_mask = master["tier"] == "chart_digitized"
    master.loc[sk_mask, "split"] = master.loc[sk_mask, "record_id"].map(bucket)
    master.loc[master["tier"] == "experimental", "split"] = "test"

    master = master[MASTER_COLS]

    assert master["record_id"].is_unique, "record_id collision"
    master.to_parquet(PROCESSED / "master.parquet", index=False)
    master.to_csv(PROCESSED / "master.csv", index=False)

    qc("\n## Unified master table\n")
    qc(f"- Total records: **{len(master)}**")
    qc(f"- Columns: {len(master.columns)} "
       f"(ids/labels + T,P,rho,Ppr,Tpr,z + {len(X_COLS)} mole fractions)")
    qc("\n### Records by tier / role\n")
    qc(master.groupby(["tier", "role"]).size().to_frame("n").to_markdown())
    qc("\n### Records by source and z_method\n")
    qc(master.groupby(["source_id", "z_method"]).size().to_frame("n").to_markdown())
    qc("\n### Gas class distribution\n")
    qc(master["gas_class"].value_counts().to_frame("n").to_markdown())
    qc("\n### Region distribution (where Ppr/Tpr known)\n")
    qc(master["region"].value_counts().to_frame("n").to_markdown())
    qc("\n### z statistics by tier (quality_flag == ok only)\n")
    ok = master[master["quality_flag"] == "ok"]
    qc(ok.groupby("tier")["z"].describe().round(4).to_markdown())
    qc("\n### Split x model-track matrix (quality ok only)\n")
    okm = master[master["quality_flag"] == "ok"]
    tab = okm.groupby(["split", "tier"]).agg(
        n=("z", "size"),
        chart_track=("track_chart", "sum"),
        composition_track=("track_composition", "sum")).reset_index()
    qc(tab.to_markdown(index=False))

    qc(f"\n- Sour-gas records (H2S+CO2 > 2 mol%): {int(master['sour'].sum())}")
    qc(f"- Quality flags: {dict(master['quality_flag'].value_counts())}")

    (REPORTS / "qc_report.md").write_text("\n".join(qc_lines), encoding="utf-8")
    print(f"\nWrote {PROCESSED / 'master.parquet'} and reports/qc_report.md")


if __name__ == "__main__":
    main()
