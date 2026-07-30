# Regional benchmark

## Part A - Libya field data (measured z)

n = 10 laboratory points from two Libyan studies (Ghani oilfield 2024; Libyan Petroleum Institute 2022). Low-pressure differential-liberation data (0.10-3.10 MPa, z 0.94-1.00): a regional spot-check, not a discriminating test. Only gas gravity is reported, so pseudo-criticals use Sutton.


| method   |   MAPE_% |   max_APE_% |
|:---------|---------:|------------:|
| Kareem   |   0.3066 |      0.7404 |
| DAK      |   0.5892 |      1.0282 |
| DPR      |   0.6177 |      1.0761 |
| HY       |   0.6768 |      1.0377 |


## Part B - regional gas compositions vs GERG-2008

1374 real regional gases (Netherlands NLOG national database; GIIGNL LNG exporters; published sour field compositions) evaluated over 5x8 realistic operating states. No measured z exists for these gases, so GERG-2008 is the reference: this quantifies the error an engineer incurs by using a correlation.


| method                  |     n |   MAPE_vs_GERG_% |   p95_% |    max_% |
|:------------------------|------:|-----------------:|--------:|---------:|
| Our NN (domain-guarded) | 54960 |           0.0186 |  0.0255 |   9.9261 |
| Our NN (raw)            | 54960 |           0.0279 |  0.0252 | 261.941  |
| DAK (Kay+WA)            | 54960 |           0.7639 |  1.6321 |   9.9261 |
| HY (Kay+WA)             | 54960 |           0.7902 |  1.6219 |  17.3195 |
| Kareem (Kay+WA)         | 54960 |           0.9962 |  2.3891 |  40.6698 |
| DAK (Sutton+WA)         | 54960 |           2.2333 |  7.2191 | 176.691  |


### Per-region


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