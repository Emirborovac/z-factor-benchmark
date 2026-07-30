"""Regional dataset assembly + regional benchmark.

Two honest parts:

  PART A - REGIONAL FIELD VALIDATION (measured z, small but real)
      Libya / Ghani oilfield (Surman J. Sci. Tech. 6(2) 2024, Table 3):
          differential vaporization at 130 F, 5 z points, incremental gas
          gravity reported.
      Libya / Libyan Petroleum Institute (Lamoj 2022, Zenodo 7321892,
          CC-BY-4.0, Table 2): differential liberation at 204 F, 5 z points
          with gas gravity.
      NOTE: both are low-pressure (<= ~1.7 MPa) liberation data with z in
      0.94-1.00, i.e. near-ideal. Reported as a regional spot-check, NOT as a
      discriminating benchmark.

  PART B - REGIONAL COMPOSITION BENCHMARK (no measured z; GERG as reference)
      Netherlands: NLOG national database, 1366 analyses / 563 boreholes.
      LNG exporters: GIIGNL 2018 (Nigeria, Qatar, Trinidad, Australia NWS,
          Malaysia Bintulu).
      Sour fields: Abu Dhabi Shah, Kazakhstan Kashagan (published values).
      Every method is run at realistic operating conditions and compared to
      GERG-2008; this measures method DISAGREEMENT on real regional gases,
      which is what an engineer in that region would actually experience.

Run:  python -X utf8 scripts/build_regional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from prepare_data import COMPONENTS  # noqa: E402

RAW = ROOT / "data" / "raw" / "regional"
OUT = ROOT / "data" / "processed"
X_COLS = [f"x_{c}" for c in COMPONENTS]
PSI_MPA = 0.00689476

# --------------------------------------------------------------------------- #
# PART A - measured z, Libya
# --------------------------------------------------------------------------- #
# Ghani oilfield, Table 3 (pressure psig -> psia by +14.7), T = 130 F
GHANI = [  # p_psig, z, incremental gas gravity
    (250, 0.942, 1.001),
    (200, 0.958, 0.941),
    (150, 0.968, 0.997),
    (100, 0.975, 1.131),
    (70, 0.980, 1.257),
]
# Lamoj 2022, Table 2 (p psia), T = 204 F
LAMOJ = [  # p_psia, z, gas gravity
    (450, 0.9651, 0.7693),
    (300, 0.9734, 0.8051),
    (200, 0.9802, 0.8371),
    (100, 0.9888, 0.9118),
    (15, 0.9981, 1.0250),
]


def part_a() -> pd.DataFrame:
    rows = []
    for p_psig, z, gg in GHANI:
        rows.append({"region": "Libya", "field": "Ghani (Fasha/Gir)",
                     "source": "Surman J Sci Tech 6(2) 2024, Table 3",
                     "T_K": (130 - 32) / 1.8 + 273.15,
                     "P_MPa": (p_psig + 14.7) * PSI_MPA,
                     "gas_gravity": gg, "z_measured": z})
    for p_psia, z, gg in LAMOJ:
        rows.append({"region": "Libya", "field": "LPI sample (Lamoj 2022)",
                     "source": "Zenodo 7321892 (CC-BY-4.0), Table 2",
                     "T_K": (204 - 32) / 1.8 + 273.15,
                     "P_MPa": p_psia * PSI_MPA,
                     "gas_gravity": gg, "z_measured": z})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# PART B - regional compositions
# --------------------------------------------------------------------------- #
NLOG_MAP = {  # NLOG column -> canonical
    "METHANE": "methane", "ETHANE": "ethane", "PROPANE": "propane",
    "N2": "nitrogen", "CO2": "co2", "H2S": "h2s", "HE": "helium",
    "ISO_BUTANE": "isobutane", "NORMAL_BUTANE": "n_butane",
    "ISO_PENTANE": "isopentane", "NORMAL_PENTANE": "n_pentane",
    "HEXANE": "hexane", "H2": "hydrogen", "AR": "argon",
}

GIIGNL = {  # GIIGNL 2018 Annual Report, via GIIGNL Information Paper No.1
    "Nigeria": dict(nitrogen=0.03, methane=91.70, ethane=5.52, propane=2.17,
                    n_butane=0.58),
    "Qatar": dict(nitrogen=0.27, methane=90.91, ethane=6.43, propane=1.66,
                  n_butane=0.74),
    "Trinidad": dict(nitrogen=0.01, methane=96.78, ethane=2.78, propane=0.37,
                     n_butane=0.06),
    "Australia (NWS)": dict(nitrogen=0.04, methane=87.33, ethane=8.33,
                            propane=3.33, n_butane=0.97),
    "Malaysia (Bintulu)": dict(nitrogen=0.14, methane=91.69, ethane=4.64,
                               propane=2.60, n_butane=0.93),
}

SOUR_FIELDS = {  # published approximate compositions
    "UAE (Abu Dhabi, sour stream)": dict(
        methane=65.0, ethane=6.5, propane=3.0, n_butane=0.46, isobutane=0.54,
        n_pentane=0.4, hexane=0.1, h2s=16.0, co2=8.0),
    "UAE (Shah field)": dict(methane=62.0, ethane=4.0, propane=1.0,
                             h2s=23.0, co2=10.0),
    "Kazakhstan (Kashagan-type)": dict(methane=72.0, ethane=6.0, propane=2.5,
                                       n_butane=1.0, h2s=17.0, co2=1.5),
}


def _norm_row(d: dict) -> dict:
    tot = sum(d.values())
    return {f"x_{k}": v / tot for k, v in d.items()}


def part_b() -> pd.DataFrame:
    rows = []

    # --- Netherlands (NLOG)
    f = RAW / "nlog_gascompos.xlsx"
    if f.exists():
        df = pd.read_excel(f)
        cols = {k: v for k, v in NLOG_MAP.items() if k in df.columns}
        sub = df[list(cols)].apply(pd.to_numeric, errors="coerce")
        sub.columns = [f"x_{v}" for v in cols.values()]
        s = sub.sum(axis=1)
        ok = (s > 95) & (s < 105) & sub["x_methane"].notna()
        sub = sub[ok].div(s[ok], axis=0).fillna(0.0)
        names = df.loc[ok, "SHORT_NM"].astype(str).to_numpy()
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append({"region": "Netherlands", "field": names[i],
                         "source": "NLOG national gas database", **r.to_dict()})
        print(f"  Netherlands: {len(sub)} analyses")

    for name, comp in GIIGNL.items():
        rows.append({"region": name, "field": "LNG export stream",
                     "source": "GIIGNL 2018 Annual Report", **_norm_row(comp)})
    for name, comp in SOUR_FIELDS.items():
        rows.append({"region": name.split(" (")[0], "field": name,
                     "source": "published field composition", **_norm_row(comp)})

    out = pd.DataFrame(rows)
    for c in X_COLS:
        if c not in out:
            out[c] = 0.0
    out[X_COLS] = out[X_COLS].fillna(0.0)
    return out[["region", "field", "source"] + X_COLS]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = part_a()
    a.to_parquet(OUT / "regional_measured.parquet", index=False)
    print(f"PART A - measured z: {len(a)} points, "
          f"{a.region.nunique()} region(s), P {a.P_MPa.min():.3f}-{a.P_MPa.max():.3f} MPa")

    b = part_b()
    b.to_parquet(OUT / "regional_compositions.parquet", index=False)
    print(f"\nPART B - compositions: {len(b)} gases")
    print(b.groupby("region").size().to_frame("n").to_markdown())
    sour = b[b.x_h2s > 0.01]
    print(f"\nsour gases (H2S>1%): {len(sour)}; "
          f"max H2S {b.x_h2s.max()*100:.1f}%, max CO2 {b.x_co2.max()*100:.1f}%, "
          f"max N2 {b.x_nitrogen.max()*100:.1f}%")
    print(f"\nwrote {OUT/'regional_measured.parquet'}, {OUT/'regional_compositions.parquet'}")


if __name__ == "__main__":
    main()
