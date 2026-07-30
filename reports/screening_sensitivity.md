# Screening sensitivity

One source was excluded by the pre-declared screen: a source is dropped if **both** reference equations deviate from it by more than 10 % in median. This recomputes the headline comparison with that source retained.

Excluded source: `10.1016/j.fluid.2012.05.005`

- As reported (screened): **2,426** points, 10 laboratories
- Screen removed: **2,426** points, 10 laboratories (+0 points from the excluded source)

## Average absolute deviation (%)

| method | screened (reported) | screen removed | change |
|---|---:|---:|---:|
| AGA8-DETAIL | 0.769 | 0.769 | +0.000 |
| NNCF | 0.789 | 0.789 | +0.000 |
| GERG-2008 | 0.917 | 0.917 | +0.000 |
| DPR | 1.890 | 1.890 | +0.000 |
| DAK | 1.919 | 1.919 | +0.000 |
| Hall-Yarborough | 1.928 | 1.928 | +0.000 |

## Ranking

- screened (reported): AGA8-DETAIL < NNCF < GERG-2008 < DPR < DAK < Hall-Yarborough
- screen removed: AGA8-DETAIL < NNCF < GERG-2008 < DPR < DAK < Hall-Yarborough

## Clustered comparison against NNCF, screen removed

> Estimator note: this bootstrap averages the per-laboratory mean advantage without weighting by laboratory size, so the point estimates are NOT directly comparable with Table 2 of the manuscript, which weights by size. Both columns here use the same estimator, which is what makes the screened/unscreened comparison valid.

| comparison | advantage (pp) | 95 % CI | labs won |
|---|---:|---|---:|
| NNCF vs AGA8-DETAIL | -0.026 | [-0.060, +0.002] | 4/10 |
| NNCF vs GERG-2008 | +0.191 | [-0.019, +0.476] | 6/10 |
| NNCF vs DPR | +0.928 | [+0.572, +1.283] | 10/10 |
| NNCF vs DAK | +0.938 | [+0.568, +1.305] | 10/10 |
| NNCF vs Hall-Yarborough | +0.984 | [+0.633, +1.337] | 10/10 |

## Verdict

- Ranking unchanged: **True**
- NNCF still beats all three classical correlations with the whole confidence interval above zero: **True**
- NNCF vs AGA8-DETAIL confidence interval still spans zero (statistical tie): **True** ([-0.060, +0.002])
