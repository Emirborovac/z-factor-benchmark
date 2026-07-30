# Systematic functional-form search for a residual correction

Every candidate fitted on 9 laboratories, scored on the 10th (leave-one-source-out). `in_sample_MAPE` exposes overfitting: forms that fit well in-sample but lose out-of-sample are learning lab-specific noise, not physics.


Domain: natural gas, n=2426, 10 sources; 31 functional forms tested.


## Base: DAK (uncorrected 1.9188%)

| form               |   base_MAPE |   in_sample_MAPE |   LOSO_MAPE |   gain_pp |
|:-------------------|------------:|-----------------:|------------:|----------:|
| rbf_k4_w1.0        |      1.9188 |           1.3381 |      1.7493 |    0.1695 |
| rbf_k4_w0.7        |      1.9188 |           1.3487 |      1.774  |    0.1448 |
| rbf_k4_w1.5        |      1.9188 |           1.3513 |      1.8273 |    0.0915 |
| phys_gaussian_bump |      1.9188 |           1.6075 |      1.8397 |    0.0791 |
| rbf_k16_w1.0       |      1.9188 |           1.2671 |      1.8405 |    0.0784 |
| poly_deg1          |      1.9188 |           1.5023 |      1.8464 |    0.0724 |
| rbf_k16_w0.7       |      1.9188 |           1.2899 |      1.8729 |    0.0459 |
| symbolic_gp        |      1.9188 |           1.9332 |      1.94   |   -0.0212 |

**IMPROVES** - best out-of-sample form: `rbf_k4_w1.0` at 1.7493% (+0.1695 pp)


## Base: HY (uncorrected 1.9284%)

| form               |   base_MAPE |   in_sample_MAPE |   LOSO_MAPE |   gain_pp |
|:-------------------|------------:|-----------------:|------------:|----------:|
| rbf_k4_w1.0        |      1.9284 |           1.3523 |      1.6399 |    0.2885 |
| rbf_k4_w0.7        |      1.9284 |           1.3675 |      1.6861 |    0.2423 |
| rbf_k4_w1.5        |      1.9284 |           1.3527 |      1.736  |    0.1924 |
| rbf_k16_w1.0       |      1.9284 |           1.2588 |      1.7598 |    0.1686 |
| rbf_k16_w0.7       |      1.9284 |           1.2738 |      1.7754 |    0.153  |
| rbf_k25_w1.5       |      1.9284 |           1.1269 |      1.8097 |    0.1186 |
| phys_gaussian_bump |      1.9284 |           1.622  |      1.8648 |    0.0636 |
| poly_deg1          |      1.9284 |           1.5843 |      1.8959 |    0.0325 |

**IMPROVES** - best out-of-sample form: `rbf_k4_w1.0` at 1.6399% (+0.2885 pp)


## Base: GERG-2008 (uncorrected 0.9173%)

| form               |   base_MAPE |   in_sample_MAPE |   LOSO_MAPE |   gain_pp |
|:-------------------|------------:|-----------------:|------------:|----------:|
| symbolic_gp        |      0.9173 |           0.9585 |      0.963  |   -0.0457 |
| phys_gaussian_bump |      0.9173 |           0.921  |      1.1132 |   -0.1959 |
| phys_exp_decay     |      0.9173 |           0.9452 |      1.1217 |   -0.2044 |
| rbf_k25_w0.7       |      0.9173 |           0.8677 |      1.1482 |   -0.2309 |
| rbf_k16_w0.7       |      0.9173 |           0.8872 |      1.1777 |   -0.2604 |
| rbf_k25_w1.0       |      0.9173 |           0.876  |      1.1853 |   -0.268  |
| phys_power_law     |      0.9173 |           0.9339 |      1.188  |   -0.2706 |
| rbf_k4_w1.0        |      0.9173 |           0.9865 |      1.2153 |   -0.2979 |

**NO FORM IMPROVES** - best out-of-sample form: `symbolic_gp` at 0.963% (-0.0457 pp)


## Base: AGA8-DETAIL (uncorrected 0.7688%)

| form               |   base_MAPE |   in_sample_MAPE |   LOSO_MAPE |   gain_pp |
|:-------------------|------------:|-----------------:|------------:|----------:|
| phys_gaussian_bump |      0.7688 |           0.7619 |      0.7756 |   -0.0068 |
| symbolic_gp        |      0.7688 |           0.7998 |      0.8058 |   -0.037  |
| phys_exp_decay     |      0.7688 |           0.864  |      0.9826 |   -0.2138 |
| phys_power_law     |      0.7688 |           0.8234 |      0.9898 |   -0.2211 |
| rbf_k16_w0.7       |      0.7688 |           0.779  |      1.0012 |   -0.2325 |
| spline_df7         |      0.7688 |           0.8144 |      1.0091 |   -0.2404 |
| rbf_k16_w1.0       |      0.7688 |           0.7854 |      1.0096 |   -0.2408 |
| rbf_k25_w1.5       |      0.7688 |           0.7803 |      1.0149 |   -0.2461 |

**NO FORM IMPROVES** - best out-of-sample form: `phys_gaussian_bump` at 0.7756% (-0.0068 pp)


## Base: NNCF (uncorrected 0.7892%)

| form               |   base_MAPE |   in_sample_MAPE |   LOSO_MAPE |   gain_pp |
|:-------------------|------------:|-----------------:|------------:|----------:|
| symbolic_gp        |      0.7892 |           0.8163 |      0.8221 |   -0.0329 |
| phys_exp_decay     |      0.7892 |           0.8518 |      0.9458 |   -0.1566 |
| phys_power_law     |      0.7892 |           0.8199 |      0.9899 |   -0.2007 |
| phys_gaussian_bump |      0.7892 |           0.7822 |      0.991  |   -0.2018 |
| rbf_k16_w0.7       |      0.7892 |           0.7916 |      1.0181 |   -0.2289 |
| rbf_k4_w1.5        |      0.7892 |           0.8905 |      1.0272 |   -0.238  |
| rbf_k4_w1.0        |      0.7892 |           0.896  |      1.0274 |   -0.2382 |
| rbf_k16_w1.0       |      0.7892 |           0.796  |      1.0295 |   -0.2403 |

**NO FORM IMPROVES** - best out-of-sample form: `symbolic_gp` at 0.8221% (-0.0329 pp)


## Base: ML-hybrid (uncorrected 2.1427%)

| form           |   base_MAPE |   in_sample_MAPE |   LOSO_MAPE |   gain_pp |
|:---------------|------------:|-----------------:|------------:|----------:|
| symbolic_gp    |      2.1427 |           2.1049 |      2.1136 |    0.0291 |
| rbf_k25_w0.7   |      2.1427 |           1.3676 |      2.3634 |   -0.2207 |
| rbf_k25_w1.0   |      2.1427 |           1.4214 |      2.4023 |   -0.2596 |
| rbf_k16_w1.0   |      2.1427 |           1.5718 |      2.4294 |   -0.2867 |
| rbf_k16_w0.7   |      2.1427 |           1.5191 |      2.4597 |   -0.317  |
| phys_exp_decay |      2.1427 |           2.0195 |      2.4659 |   -0.3232 |
| fourier_o1     |      2.1427 |           2.0259 |      2.5228 |   -0.3801 |
| poly_deg1      |      2.1427 |           2.0658 |      2.5478 |   -0.4051 |

**IMPROVES** - best out-of-sample form: `symbolic_gp` at 2.1136% (+0.0291 pp)


Last symbolic expression found: `-0.001`
