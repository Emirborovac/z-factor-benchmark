# Low-order correction factors (leave-one-source-out)

Coefficients fitted by least squares on held-in sources only; every score is out-of-sample for a new lab.


Natural-gas domain: n=2426, 10 sources


| method        |   uncorrected |   F0_scale |   F1_affine |   F2_departure |   F3_dep_linear |   F4_dep_temp | best_form   |   best_MAPE |   gain_pp |
|:--------------|--------------:|-----------:|------------:|---------------:|----------------:|--------------:|:------------|------------:|----------:|
| DAK           |        1.9188 |     2.362  |      2.8861 |         2.3607 |          2.9382 |        2.5882 | uncorrected |      1.9188 |    0      |
| HY            |        1.9284 |     2.3576 |      2.8403 |         2.3302 |          2.7252 |        2.8234 | uncorrected |      1.9284 |    0      |
| GERG-2008     |        0.9173 |     1.0877 |      1.3114 |         1.0656 |          1.5484 |        1.6017 | uncorrected |      0.9173 |    0      |
| ML-hybrid(v2) |        1.853  |     1.7933 |      1.9487 |         1.9625 |          3.9122 |        3.6815 | F0_scale    |      1.7933 |    0.0597 |