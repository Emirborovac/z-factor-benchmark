# Z-factor benchmark - data preparation QC report

## Per-source cleaning

- Standing-Katz: 57060 raw -> 57060 clean (0 exact duplicates, 0 out-of-bounds removed)
  NIST compositions loaded: 200 gases (#2-#201)
  NOTE: 20 NG rows reference gas numbers with no composition ([np.int64(202)]); kept, flagged no_composition
- NIST natural_gas_sheet: 3864 (T,rho) states -> 7728 (point,method) rows
- NIST binary_sheet: 3645 (T,rho) states -> 7290 (point,method) rows
- NIST combined: 15018 rows; flags: {'ok': np.int64(14533), 'eos_failed': np.int64(304), 'eos_out_of_range': np.int64(141), 'no_composition': np.int64(40)} (liquid/critical-region states kept, role=code_validation)
  WARNING: Zenodo mixes with composition sum far from 1: {'Mix11': np.float64(0.975)} (kept, renormalized)
- Zenodo experimental: 1081 raw -> 1079 clean (0 duplicates removed); 1079 rows joined to a composition

- Ppr/Tpr filled via Kay's rule for 14978 composition-domain records

## Unified master table

- Total records: **73157**
- Columns: 42 (ids/labels + T,P,rho,Ppr,Tpr,z + 21 mole fractions)

### Records by tier / role

|                                       |     n |
|:--------------------------------------|------:|
| ('chart_digitized', 'train')          | 57060 |
| ('eos_calculated', 'code_validation') | 15018 |
| ('experimental', 'test')              |  1079 |

### Records by source and z_method

|                                   |     n |
|:----------------------------------|------:|
| ('nist_aga8', 'aga8_detail')      |  7509 |
| ('nist_aga8', 'gerg2008')         |  7509 |
| ('sk_chart', 'digitized_chart')   | 57060 |
| ('zenodo_bougha2026', 'measured') |  1079 |

### Gas class distribution

| gas_class      |     n |
|:---------------|------:|
| chart_generic  | 57060 |
| natural_gas    |  8237 |
| binary         |  7298 |
| multicomponent |   482 |
| pure           |    80 |

### Region distribution (where Ppr/Tpr known)

| region               |     n |
|:---------------------|------:|
| moderate             | 36526 |
| high_pressure        | 22206 |
| near_critical_trough | 12876 |
| ultra_high_pressure  |   921 |
| near_ideal           |   588 |
| unclassified         |    40 |

### z statistics by tier (quality_flag == ok only)

| tier            |   count |   mean |    std |    min |    25% |    50% |    75% |    max |
|:----------------|--------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| chart_digitized |   57060 | 1.0214 | 0.2767 | 0.2518 | 0.8439 | 0.9882 | 1.2509 | 1.7536 |
| eos_calculated  |   14533 | 0.6042 | 0.3308 | 0      | 0.3462 | 0.5982 | 0.8206 | 2.9787 |
| experimental    |    1079 | 1.2852 | 0.3511 | 0.445  | 0.9792 | 1.2304 | 1.5758 | 2.1927 |

### Split x model-track matrix (quality ok only)

| split   | tier            |     n |   chart_track |   composition_track |
|:--------|:----------------|------:|--------------:|--------------------:|
| none    | eos_calculated  | 14533 |             0 |               14533 |
| test    | experimental    |  1079 |          1079 |                1079 |
| train   | chart_digitized | 51522 |         51522 |                   0 |
| val     | chart_digitized |  5538 |          5538 |                   0 |

- Sour-gas records (H2S+CO2 > 2 mol%): 3863
- Quality flags: {'ok': np.int64(72672), 'eos_failed': np.int64(304), 'eos_out_of_range': np.int64(141), 'no_composition': np.int64(40)}