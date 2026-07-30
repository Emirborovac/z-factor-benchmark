# Z-Factor Benchmark — Master Dataset Datasheet

**File:** `master.parquet` / `master.csv` · **Records:** 73,157 · **Built by:** `scripts/prepare_data.py` (deterministic; rerun to regenerate from `data/raw/`)

## Provenance (three tiers, never mixed in training)

| tier | source_id | n | role | origin |
|---|---|---|---|---|
| `chart_digitized` | `sk_chart` | 57,060 | train | Standing & Katz 1942 chart, digitized by A. Reyes (zFactor.DL, `extras/tidy_SK.csv`) |
| `eos_calculated` | `nist_aga8` | 15,018 | code_validation | NIST usnistgov/AGA8 TESTDATA; REFPROP-9.1 saturation-boundary states; Z computed as P/(ρRT), one row per (state, EOS method) |
| `experimental` | `zenodo_bougha2026` | 1,079 | test | Bougha et al. 2026, *Sustainability* 18(4):1742, Zenodo 10.5281/zenodo.18225906, CC-BY-4.0; lab-measured Z |

## Schema

| column | meaning |
|---|---|
| `record_id` | stable content hash (source-prefixed) |
| `source_id` / `tier` / `role` | provenance and intended use (`train` / `code_validation` / `test`) |
| `gas_class` | `pure` / `binary` / `multicomponent` / `natural_gas` / `chart_generic` (SK chart has no composition) |
| `region` | `near_ideal` (Ppr≤0.2) / `near_critical_trough` (Tpr<1.2, Ppr≤6) / `moderate` (Ppr≤8) / `high_pressure` (8<Ppr≤15) / `ultra_high_pressure` (Ppr>15) |
| `T_K`, `P_MPa`, `rho_mol_l` | absolute state (composition-domain records only) |
| `Ppr`, `Tpr` | pseudo-reduced coordinates |
| `ppr_tpr_origin` | `as_published` (shipped with source) / `kay_rule` (computed here from composition via Kay mixing rule) / `none` |
| `z` | compressibility factor (the target) |
| `z_method` | `measured` / `digitized_chart` / `gerg2008` / `aga8_detail` |
| `sour` | H₂S + CO₂ > 2 mol% |
| `mix_id`, `reference` | mixture identity and literature citation |
| `quality_flag` | `ok` / `eos_failed` (P≤0 out-of-range state) / `eos_out_of_range` (DETAIL in liquid/critical region, per NIST's own warning) / `no_composition` (gas #202, undefined in NIST file) |
| `track_chart` | usable by the chart-domain model (has *published* Ppr/Tpr; Kay-rule coords excluded) |
| `track_composition` | usable by the composition-domain model (full composition + absolute state, or experimental with reconstructable P,T) |
| `split` | `train` / `val` (90/10 deterministic record-id hash on SK chart) / `test` (all experimental) / `none` (NIST code-validation) |
| `x_methane` … `x_argon` | 21 mole fractions (AGA8 slate), renormalized to sum 1; NaN where unknown |

## Cleaning applied

- Zenodo: dropped 2 physically implausible rows (incl. Tpr≈0.34 typo); mole-percent compositions renormalized (Mix11 summed 97.5%).
- NIST: Gas # join follows the file's own convention (**gas # = Excel row, 2–201**); empty trailing composition rows dropped; invalid liquid/critical DETAIL states kept but flagged — filter `quality_flag == "ok"` before any analysis.
- SK chart: no duplicates or out-of-bounds points found (bounds: 1.0≤Tpr≤3.0, 0≤Ppr≤15.2, 0<z≤2.5).

## Rules of use

1. **Never train on `role != "train"` records.** The experimental tier is the untouchable test set; the NIST tier validates EOS code, not models (saturation-boundary states, includes liquid phase).
2. Filter `quality_flag == "ok"` unless specifically studying EOS failure modes.
3. The synthetic training tier (GERG-2008 via pyaga8) is generated separately and is NOT in this file by design — regenerate at any density needed.
4. Cite original sources when publishing: Standing & Katz (1942) + Reyes' digitization; NIST AGA8; Bougha et al. (2026) CC-BY-4.0.
