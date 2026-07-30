# Lab-calibrated correction layer (leave-one-source-out)

Every number is out-of-sample: the correction never saw the source it is scored on. The same correction is applied to every base method as a control.


Natural-gas domain: n=2426, 10 independent sources


## Base vs calibrated

| method        |   base_MAPE_% |   calibrated_MAPE_% |   gain_pp |   base_acc_% |   cal_acc_% |
|:--------------|--------------:|--------------------:|----------:|-------------:|------------:|
| GERG-2008     |        0.9173 |              1.1588 |   -0.2415 |       99.083 |      98.841 |
| HY            |        1.9284 |              1.6876 |    0.2408 |       98.072 |      98.312 |
| DAK           |        1.9188 |              1.7829 |    0.136  |       98.081 |      98.217 |
| ML-hybrid(v2) |        1.853  |              2.4579 |   -0.6048 |       98.147 |      97.542 |


## Calibrated MAPE_% per held-out source

|                             |   DAK |    HY |   GERG-2008 |   ML-hybrid(v2) |   n |
|:----------------------------|------:|------:|------------:|----------------:|----:|
| 10.1016/j.fluid.2016.08.002 | 5.474 | 5.112 |       3.33  |           7.864 | 526 |
| 10.1016/j.jct.2005.10.004   | 1.612 | 1.393 |       0.405 |           3.074 | 242 |
| 10.1016/j.jct.2012.12.018   | 0.473 | 0.353 |       0.333 |           0.867 |  19 |
| 10.1016/j.jct.2015.12.006   | 0.141 | 0.17  |       0.302 |           0.15  | 279 |
| 10.1016/j.jct.2016.03.035   | 0.223 | 0.264 |       0.179 |           0.214 | 178 |
| 10.1016/j.jct.2016.05.024   | 0.287 | 0.295 |       0.37  |           0.357 | 119 |
| 10.1021/acs.jced.6b00137    | 1.432 | 1.444 |       0.507 |           0.966 | 283 |
| 10.1021/acs.jced.7b01125    | 0.757 | 0.759 |       0.947 |           0.891 | 548 |
| 10.1021/acs.jced.8b00433    | 0.747 | 0.604 |       0.779 |           1.074 |  84 |
| 10.1021/je500792x           | 0.339 | 0.38  |       0.455 |           0.62  | 148 |