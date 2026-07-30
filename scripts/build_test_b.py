"""Build Test Set B from NIST ThermoML archives: real experimental gas-phase
(p, rho, T, x) measurements -> exact z = p*MW/(rho*R*T), with absolute inputs.

Scans ThermoML XML files for mixture/pure blocks with:
  property = Mass density (kg/m3), phase Gas (or Supercritical fluid),
  variables Temperature (K) + Pressure (kPa), compounds within the AGA8 slate.

Run:  python -X utf8 scripts/build_test_b.py [--dir data/raw/thermoml/archive]
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import COMPONENTS  # noqa: E402

NS = {"t": "http://www.iupac.org/namespaces/ThermoML"}
R_GAS = 8.314462618

MW = dict(zip(COMPONENTS, [16.043, 28.014, 44.01, 30.07, 44.097, 58.123,
                           58.123, 72.15, 72.15, 86.177, 100.204, 114.231,
                           128.258, 142.285, 2.016, 31.999, 28.01, 18.015,
                           34.081, 4.003, 39.948]))

NAME_MAP = {
    "methane": "methane", "nitrogen": "nitrogen", "carbon dioxide": "co2",
    "ethane": "ethane", "propane": "propane",
    "2-methylpropane": "isobutane", "isobutane": "isobutane",
    "butane": "n_butane", "n-butane": "n_butane",
    "2-methylbutane": "isopentane", "isopentane": "isopentane",
    "pentane": "n_pentane", "n-pentane": "n_pentane",
    "hexane": "hexane", "n-hexane": "hexane",
    "heptane": "heptane", "n-heptane": "heptane",
    "octane": "octane", "n-octane": "octane",
    "nonane": "nonane", "n-nonane": "nonane",
    "decane": "decane", "n-decane": "decane",
    "hydrogen": "hydrogen", "oxygen": "oxygen",
    "carbon monoxide": "co", "water": "water",
    "hydrogen sulfide": "h2s", "helium": "helium", "argon": "argon",
}

GAS_PHASES = {"Gas", "Supercritical fluid", "Fluid (supercritical or subcritical)"}


def parse_file(path: Path) -> list[dict]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    doi_el = root.find(".//t:Citation/t:sDOI", NS)
    doi = doi_el.text if doi_el is not None else path.stem
    year_el = root.find(".//t:Citation//t:yrYrPub", NS)

    compounds = {}
    for c in root.findall("t:Compound", NS):
        org = c.find(".//t:nOrgNum", NS)
        name = c.find(".//t:sCommonName", NS)
        if org is not None and name is not None:
            compounds[org.text] = name.text.strip().lower()

    rows = []
    for b in root.findall("t:PureOrMixtureData", NS):
        prop = b.find(".//t:Property//t:ePropName", NS)
        phase = b.find(".//t:Property//t:ePropPhase", NS)
        if prop is None or "Mass density" not in (prop.text or ""):
            continue
        if phase is None or phase.text not in GAS_PHASES:
            continue

        # block components must all be mappable
        orgs = [e.text for e in b.findall(
            ".//t:Component//t:nOrgNum", NS)]
        names = [compounds.get(o, "?") for o in orgs]
        if not names or any(n not in NAME_MAP for n in names):
            continue

        # variables: number -> (kind, orgnum)
        variables = {}
        for v in b.findall(".//t:Variable", NS):
            num = v.find("t:nVarNumber", NS)
            vtype = v.find(".//t:VariableType/*", NS)
            vorg = v.find(".//t:VariableID//t:nOrgNum", NS)
            if num is not None and vtype is not None:
                variables[num.text] = (vtype.text,
                                       vorg.text if vorg is not None else None)

        # constraints: fixed composition entries
        const_x = {}
        okblock = True
        for c in b.findall(".//t:Constraint", NS):
            ctype = c.find(".//t:ConstraintType/*", NS)
            cval = c.find("t:nConstraintValue", NS)
            corg = c.find(".//t:ConstraintID//t:nOrgNum", NS)
            if ctype is None or cval is None:
                continue
            if "Mole fraction" in ctype.text and corg is not None:
                const_x[corg.text] = float(cval.text)
            elif ctype.text.startswith(("Pressure", "Temperature")):
                pass  # fixed T or p handled below only if also a variable
        for nv in b.findall(".//t:NumValues", NS):
            T = P = None
            x = dict(const_x)
            for vv in nv.findall("t:VariableValue", NS):
                num = vv.find("t:nVarNumber", NS).text
                val = float(vv.find("t:nVarValue", NS).text)
                kind, org = variables.get(num, ("?", None))
                if kind.startswith("Temperature"):
                    T = val
                elif kind.startswith("Pressure"):
                    P = val / 1000.0  # kPa -> MPa
                elif "Mole fraction" in kind and org is not None:
                    x[org] = val
            pv = nv.find(".//t:PropertyValue/t:nPropValue", NS)
            if T is None or P is None or pv is None:
                continue
            rho = float(pv.text)
            # complete composition: infer remaining single component
            known = sum(x.values())
            missing = [o for o in orgs if o not in x]
            if len(missing) == 1 and 0 <= 1 - known <= 1:
                x[missing[0]] = 1 - known
            elif missing:
                okblock = False
                continue
            comp_named = {NAME_MAP[compounds[o]]: v for o, v in x.items()}
            mw = sum(MW[k] * v for k, v in comp_named.items())
            z = P * 1e6 * (mw / 1000.0) / (rho * R_GAS * T)
            rows.append({"doi": doi, "T_K": T, "P_MPa": P,
                         "rho_kg_m3": rho, "MW": mw, "z": z,
                         **{f"x_{k}": v for k, v in comp_named.items()}})
        if not okblock:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "raw" / "thermoml"))
    args = ap.parse_args()
    files = sorted(Path(args.dir).rglob("*.xml"))
    print(f"scanning {len(files)} xml files...")
    all_rows = []
    for f in files:
        rows = parse_file(f)
        if rows:
            print(f"  {f.name}: {len(rows)} gas-density points")
            all_rows.extend(rows)
    if not all_rows:
        print("no data found")
        return
    df = pd.DataFrame(all_rows)
    for c in [f"x_{k}" for k in COMPONENTS]:
        if c not in df:
            df[c] = 0.0
    df[[f"x_{k}" for k in COMPONENTS]] = \
        df[[f"x_{k}" for k in COMPONENTS]].fillna(0.0)
    # physical gas-phase screening
    n0 = len(df)
    df = df[(df.z > 0.15) & (df.z < 3.2) & (df.T_K > 200) & (df.P_MPa > 0.01)]
    print(f"\nTOTAL: {len(df)} points ({n0 - len(df)} screened out) "
          f"from {df.doi.nunique()} sources")
    print(df.groupby("doi").agg(n=("z", "size"), Tmin=("T_K", "min"),
                                Tmax=("T_K", "max"), Pmax=("P_MPa", "max"),
                                zmin=("z", "min"), zmax=("z", "max")).round(2)
          .to_string())
    out = ROOT / "data" / "processed" / "test_b.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
