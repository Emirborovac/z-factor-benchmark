# Results — authoritative numbers for the manuscript

Regenerate with `python -X utf8 scripts/collect_numbers.py`. Every value in the paper must come from this file.

## 1 Benchmark composition

- Test Set B, total extracted: **6,726** gas-phase points from **40** published studies
- Excluded by the pre-declared screen (both reference EOS exceed 10 % median deviation): 1 source, 30 points (['10.1016/j.fluid.2012.05.005'])
- Natural-gas domain (x_CH4 >= 0.5, Tpr <= 3.0, T <= 500 K): **2,426** points from **10** laboratories
- State range of the NG domain: T 205.0–500.0 K, P 0.120–200.5 MPa, Ppr 0.027–43.5, Tpr 0.890–2.54
- Full Test Set B state range: T 205–927 K, P 0.014–200.5 MPa
- Sour points (H2S + CO2 > 2 mol%) in the NG domain: 610

### Synthetic training pool (counted from the parquet files)

| tier | compositions | DETAIL-labelled states |
|---|---:|---:|
| base | 10,000 | 9,999,484 |
| broadened | 12,000 | 10,683,016 |
| hydrogen-blend | 2,500 | 1,000,000 |
| **total** | **24,500** | **21,682,500** (2.168e7) |

Sampling rules (base tier, from MANIFEST.md; the broadened and hydrogen tiers follow the same rules over wider envelopes):

- seed: 20260728
- compositions: 10000 ({'nist_perturbed': np.int64(4500), 'lean_sweet': np.int64(1325), 'sour': np.int64(1325), 'rich': np.int64(1325), 'n2_rich': np.int64(1325), 'nist_real': np.int64(200)}); split: {'train': np.int64(9026), 'val': np.int64(974)}
- T range: [250.0, 450.0] K with T >= 1.05 * Tpc_kay
- P range: log-uniform [0.05, 140.0] MPa, capped at Ppr_kay <= 30.0
- z bounds: (0.15, 3.0); stability filter: dP/drho > 0
- points kept: 9,999,878; rejected: {'unstable': 0, 'z_bounds': 122, 'error': 0}
- z stats: {'count': 9999878.0, 'mean': 1.0485, 'std': 0.2774, 'min': 0.2581, '25%': 0.9597, '50%': 0.9934, '75%': 0.9992, 'max': 3.0}
- DETAIL cross-check on 20000 sampled points: median |dz|/z = 0.0148%, p95 = 0.4190%, max = 33.330%
- EOS: GERG-2008 via pyaga8 (validated vs NIST reference to machine precision)

## 2 Model

- Architecture: 3 residual blocks x 384 units, SiLU; **899,329** trainable parameters
- Inputs (30): T,P,Ppr,Tpr,1/Tpr,logPpr,Ppr/Tpr,Ppr2/Tpr3,gamma,x1..x21
- Target: z - z_DAK (residual); objective: mean relative error
- Teacher: **AGA8-DETAIL** (labels regenerated on identical states)
- Validation AAD against its teacher: **0.0376 %**

## 3 Headline benchmark — natural-gas domain

| method | AAD (%) | RMSE | bias (%) | within ±0.5 % | ±1 % | ±2 % |
|---|---:|---:|---:|---:|---:|---:|
| AGA8-DETAIL | 0.769 | 0.01287 | +0.477 | 78.5 | 81.1 | 85.2 |
| NNCF | 0.789 | 0.01351 | +0.481 | 76.7 | 80.8 | 85.0 |
| GERG-2008 | 0.917 | 0.01333 | +0.382 | 67.7 | 75.7 | 83.9 |
| DPR | 1.890 | 0.02263 | +0.443 | 20.6 | 40.0 | 71.0 |
| DAK | 1.919 | 0.02370 | +0.490 | 22.1 | 40.3 | 70.9 |
| Hall-Yarborough | 1.928 | 0.02460 | +0.467 | 18.0 | 39.9 | 70.5 |
| Kareem | 12.639 | 7.54320 | -0.035 | 20.8 | 35.1 | 62.1 |

Kareem is evaluated outside its stated validity range for part of the domain; see the validity note below.

- Kareem within its stated validity (1.15 <= Tpr <= 3, 0.2 <= Ppr <= 15): AAD **2.064 %** on 1,890 points
- Fraction of the NG domain outside Kareem validity: **22.1 %**

## 4 Statistical comparison (clustered by laboratory)

| comparison | mean advantage (pp) | 95 % CI | P(NNCF better) | labs won | Wilcoxon p |
|---|---:|---|---:|---:|---:|
| NNCF vs DPR | +1.1012 | [+0.5726, +1.2899] | 1.000 | 10/10 | 0.00e+00 |
| NNCF vs DAK | +1.1296 | [+0.5660, +1.3070] | 1.000 | 10/10 | 0.00e+00 |
| NNCF vs Hall-Yarborough | +1.1392 | [+0.6433, +1.3274] | 1.000 | 10/10 | 0.00e+00 |
| NNCF vs GERG-2008 | +0.1281 | [-0.0181, +0.4922] | 0.946 | 6/10 | 4.24e-45 |
| NNCF vs AGA8-DETAIL | -0.0204 | [-0.0601, +0.0021] | 0.039 | 4/10 | 1.26e-10 |

## 5 Surrogate fidelity to GERG-2008 (NG domain)

- median |Δz|/z: **0.0950 %**
- 95th percentile |Δz|/z: **1.4754 %**
- maximum |Δz|/z: 3.631 %

## 6 Per-laboratory results

| laboratory (DOI) | n | NNCF AAD | DPR AAD | GERG AAD |
|---|---:|---:|---:|---:|
| 10.1016/j.fluid.2016.08.002 | 526 | 3.195 | 4.736 | 3.190 |
| 10.1016/j.jct.2005.10.004 | 242 | 0.038 | 1.834 | 0.010 |
| 10.1016/j.jct.2012.12.018 | 19 | 0.059 | 0.200 | 0.051 |
| 10.1016/j.jct.2015.12.006 | 279 | 0.131 | 1.339 | 0.788 |
| 10.1016/j.jct.2016.03.035 | 178 | 0.112 | 0.421 | 0.229 |
| 10.1016/j.jct.2016.05.024 | 119 | 0.093 | 1.672 | 1.356 |
| 10.1021/acs.jced.6b00137 | 283 | 0.326 | 1.526 | 0.156 |
| 10.1021/acs.jced.7b01125 | 548 | 0.039 | 0.817 | 0.045 |
| 10.1021/acs.jced.8b00433 | 84 | 0.356 | 0.822 | 0.364 |
| 10.1021/je500792x | 148 | 0.087 | 0.346 | 0.153 |

## 7 Regional study

Source: `reports/regional_benchmark.md` (unchanged, reproduced for convenience)

| method   |   MAPE_% |   max_APE_% |
|:---------|---------:|------------:|
| Kareem   |   0.3066 |      0.7404 |
| DAK      |   0.5892 |      1.0282 |
| DPR      |   0.6177 |      1.0761 |
| HY       |   0.6768 |      1.0377 |
| method                  |     n |   MAPE_vs_GERG_% |   p95_% |    max_% |
|:------------------------|------:|-----------------:|--------:|---------:|
| Our NN (domain-guarded) | 54960 |           0.0186 |  0.0255 |   9.9261 |
| Our NN (raw)            | 54960 |           0.0279 |  0.0252 | 261.941  |
| DAK (Kay+WA)            | 54960 |           0.7639 |  1.6321 |   9.9261 |
| HY (Kay+WA)             | 54960 |           0.7902 |  1.6219 |  17.3195 |
| Kareem (Kay+WA)         | 54960 |           0.9962 |  2.3891 |  40.6698 |
| DAK (Sutton+WA)         | 54960 |           2.2333 |  7.2191 | 176.691  |
|                    |   DAK (Kay+WA) |   HY (Kay+WA) |   Kareem (Kay+WA) |   DAK (Sutton+WA) |   Our NN (raw) |   Our NN (domain-guarded) |   n_gases |
|:-------------------|---------------:|--------------:|------------------:|------------------:|---------------:|--------------------------:|----------:|
| Australia (NWS)    |          0.521 |         0.601 |             0.94  |             1.163 |          0.013 |                     0.013 |         1 |
| Kazakhstan         |          0.941 |         0.887 |             0.985 |             3.453 |          0.051 |                     0.051 |         1 |
| Malaysia (Bintulu) |          0.585 |         0.659 |             0.934 |             0.612 |          0.012 |                     0.012 |         1 |
| Netherlands        |          0.764 |         0.79  |             0.996 |             2.236 |          0.028 |                     0.019 |      1366 |
| Nigeria            |          0.625 |         0.698 |             0.958 |             0.609 |          0.011 |                     0.011 |         1 |
| Qatar              |          0.631 |         0.705 |             0.964 |             0.6   |          0.011 |                     0.011 |         1 |
| Trinidad           |          0.779 |         0.818 |             0.998 |             0.415 |          0.007 |                     0.007 |         1 |
| UAE                |          0.887 |         0.901 |             1.044 |             3.947 |          0.096 |                     0.096 |         2 |

## 8 Residual-correction study

- functional families tested: **31**, base predictors: 6, total leave-one-laboratory-out fits: **186**

| base predictor | uncorrected AAD | best out-of-sample form | best AAD | gain (pp) |
|---|---:|---|---:|---:|
| AGA8-DETAIL | 0.7688 | phys_gaussian_bump | 0.7756 | -0.0068 |
| DAK | 1.9188 | rbf_k4_w1.0 | 1.7493 | +0.1695 |
| GERG-2008 | 0.9173 | symbolic_gp | 0.9630 | -0.0457 |
| HY | 1.9284 | rbf_k4_w1.0 | 1.6399 | +0.2885 |
| ML-hybrid | 2.1427 | symbolic_gp | 2.1136 | +0.0291 |
| NNCF | 0.7892 | symbolic_gp | 0.8221 | -0.0329 |

## 9 Model families (same data, same target)

| family | AAD (%) |
|---|---:|
| Gradient boosting, raw (T,P,x) | 2.451 |
| Gradient boosting, reduced coords | 2.231 |
| Gradient boosting, broad pool | 2.143 |
| Gradient boosting, focused pool | 1.853 |
| NNCF, residual neural EOS | 0.789 |

(tree values from reports/ during model selection; NNCF recomputed here)

## 8 Computational cost

Batch of 20,000 states, best of 5, 6 CPU threads. Regenerate with `python -X utf8 scripts/benchmark_cost.py`.

| method | CPU us/state | states/s |
|---|---:|---:|
| Hall-Yarborough | 1.97 | 508,464 |
| DPR | 2.50 | 400,095 |
| DAK | 3.27 | 305,443 |
| GERG-2008 | 4.80 | 208,456 |
| AGA8-DETAIL | 5.49 | 182,299 |
| NNCF (this work) | 14.71 | 67,957 |

- Newton iterations, correlation family: mean 15.42, max 100 (the max is the iteration cap, i.e. non-convergence)
- NNCF iterations: 0 (fixed depth; convergence failure impossible)
- GPU (RTX 3050 Laptop), batched: **0.62 us/state** (~1.6e6 states/s) - measured separately, see the manuscript

> pyaga8 is a Rust library driven from Python one state at a time, so the reference-EOS timings include per-call interpreter overhead and are an upper bound on their intrinsic cost. The correlations and the network are evaluated over whole arrays.
