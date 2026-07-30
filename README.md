# NNCF — a neural surrogate for the natural-gas compressibility factor

**Reference-equation accuracy at correlation cost, measured on 2426 independent
laboratory measurements.**

| Method | AAD vs experiment | within ±0.5 % | Cost |
|---|---|---|---|
| AGA8-DETAIL | **0.769 %** | 78.5 % | iterative density solve |
| **NNCF (this work)** | **0.789 %** | 76.7 % | one feed-forward pass, ~14 µs |
| GERG-2008 | 0.917 % | 67.7 % | iterative density solve |
| DPR | 1.890 % | 20.6 % | Newton iteration |
| DAK | 1.919 % | 22.1 % | Newton iteration |
| Hall–Yarborough | 1.928 % | 18.0 % | Newton iteration |

*Natural-gas domain of Test Set B: 2426 measurements from 10 independent
laboratories. Lower is better.*

NNCF is statistically indistinguishable from AGA8-DETAIL (−0.020 pp,
95 % CI [−0.060, +0.002], clustered by laboratory) and reduces the error of the
three standard industry correlations by 1.10–1.14 percentage points **at every
one of the ten laboratories**. It does not beat AGA8-DETAIL, and this repository
does not claim that it does.

No laboratory measurement was used in training. The model is distilled from
AGA8-DETAIL on 21.7 million synthetic states, so the benchmark is a genuine
held-out test.

---

## Why this repository exists

Machine learning has been proposed for *z* prediction for fifteen years, and the
reported gains are large. Almost all of them are measured on values digitised
from the 1942 Standing–Katz chart — the same chart the classical correlations
were fitted to. A model trained and tested on chart-derived data is being
compared against correlations that already encode that chart, using inputs that
carry its imprint. Those gains have never had to survive contact with
independent laboratory measurements, because no common test set of measured *z*
with absolute inputs existed.

So the first contribution here is the test set, not the model.

**Test Set B** — 6726 gas-phase measurements from 40 published studies in the
NIST ThermoML archive, each with absolute pressure, temperature and full molar
composition, from which

```
z = pM / (ρRT)
```

follows exactly. No chart, correlation or equation of state enters that
identity, so the values are measurements rather than predictions.

---

## Using the model

```python
import numpy as np
from zfactor.predict import predict_z, COMPONENTS

# mole fractions in COMPONENTS order (21 components, AGA8 slate)
x = np.zeros((1, 21))
x[0, COMPONENTS.index("methane")] = 0.93
x[0, COMPONENTS.index("ethane")]  = 0.04
x[0, COMPONENTS.index("nitrogen")] = 0.02
x[0, COMPONENTS.index("co2")]      = 0.01

z, status = predict_z(T_K=np.array([310.0]),
                      P_MPa=np.array([8.0]),
                      composition=x)
# z -> array([0.8964]),  status -> array(['ok'])
```

`status` is `"ok"` or `"fallback"`. The model is accurate inside the region it
was trained on and degrades outside it, so it ships with a guard rather than a
confident extrapolation: any input outside the declared applicability domain

```
x_CH4 >= 0.30     200 <= T <= 520 K     1.00 <= Tpr <= 3.05     0.02 <= Ppr <= 16
```

is routed to DAK and flagged `"fallback"`. On 1374 real produced-gas
compositions over 40 operating states, 99.1 % of evaluations fall inside the
domain.

The model is 899,329 parameters, 3.6 MB, and runs at ~14 µs per prediction on
CPU in batch. It needs no iterative solve and no convergence tolerance.

---

## What is in here

```
src/zfactor/
  correlations.py     7 classical correlations, vectorised, Newton-solved
  predict.py          the deployable model + applicability-domain guard
scripts/
  prepare_data.py     build the unified master table
  build_test_b.py     extract Test Set B from ThermoML
  generate_synthetic.py / relabel_detail.py
                      21.7M synthetic training states; relabel with a
                      different teaching equation on identical states
  train_nn.py         train NNCF
  eval_test_b.py      benchmark every method on Test Set B
  significance_test.py        laboratory-clustered bootstrap
  functional_form_search.py   31 correction families, leave-one-lab-out
  make_figures.py     all 12 publication figures
  check_figures.py    automated edge-ink QA on every figure
  check_manuscript.py Elsevier/Fuel compliance gate on the manuscript
data/processed/
  test_b.parquet      the benchmark (6726 points, absolute P/T/x)
  master.parquet      unified table, all tiers
  DATASHEET.md        provenance, screening, known limitations
models/nn/            trained checkpoints (GERG- and DETAIL-taught)
paper/                manuscript, figures, bibliography
```

### Reproducing

```bash
pip install -r requirements.txt
python -X utf8 scripts/prepare_data.py        # unified table
python -X utf8 scripts/build_test_b.py        # the benchmark
python -X utf8 scripts/eval_test_b.py         # the leaderboard
python -X utf8 scripts/make_figures.py        # all 12 figures
python -m pytest tests -q                     # correlation verification
```

Training from scratch needs the synthetic pool
(`generate_synthetic.py`, then `train_nn.py --teacher detail`) and a CUDA GPU;
the released checkpoints let you skip it.

Every correlation implementation is verified against the reference grids of an
independent published implementation ([zFactor](https://github.com/f0nzie/zFactor));
largest relative deviation over all verification points is 5×10⁻⁴.

---

## Three findings worth more than the model

**1. The teaching equation sets the ceiling.** Holding sampled states,
architecture, training recipe and random seed fixed, and changing *only* the
reference equation used to label the 21.7 M training states, moved test accuracy
from 0.923 % to 0.789 %. Both variants reproduce their own teacher equally well
(0.0375 % and 0.0376 % on validation), so the difference on measured data is
attributable to the teacher alone. Changing the model family was worth ~0.9
percentage points; changing the teacher ~0.13; every hyperparameter we tried was
worth less than either.

If you publish a machine-learned property model, state your label source. It
bounds what the model can achieve.

**2. Fitted corrections do not save an already-good model.** We searched 31
functional families — polynomials to 6th order, Gaussian RBFs, tensor-product
splines, rational forms, physical nonlinear forms, Fourier bases, symbolic
regression — over 6 base methods, 186 leave-one-laboratory-out fits. For DAK and
Hall–Yarborough, 16 of 62 fits transfer to an unseen laboratory (best: +0.288 pp).
For AGA8-DETAIL, GERG-2008 and NNCF, **0 of 93 fits transfer**. The correlations'
residual is partly systematic and therefore learnable; the reference equations'
is measurement scatter, and cannot be.

**3. The pseudo-critical rule matters more than the correlation.** Using
gravity-based (Sutton) instead of composition-based (Kay) pseudo-criticals
degrades DAK's agreement with the reference equation from 0.76 % to 2.23 %, and
to 3.5–3.9 % on sour gases — a larger effect than the choice among DAK, HY and
DPR. This is rarely reported alongside correlation comparisons, so we treat the
reduction rule as part of the method under test.

---

## Limitations

- **Domain-bounded by construction.** Outside the declared domain the error
  grows sharply. The guard exists for that reason.
- **It inherits its teacher's blind spots.** The largest residual errors occur on
  a sour near-critical mixture and a methane–helium binary — regions where
  AGA8-DETAIL and GERG-2008 are themselves weak. Distillation cannot fix that.
- **Difficult chemistries appear at one laboratory each.** A correction for them
  could be fitted but never validated, so none is offered.
- **Benchmark coverage reflects what is public.** Ten laboratories, mostly
  European, dominate the natural-gas domain; sour and near-critical states rest
  on few sources. Replicated measurements of difficult chemistries are the most
  useful thing further experimental work could add.

See `data/processed/DATASHEET.md` for per-source provenance and the one
pre-declared exclusion.

---

## Citing

If you use the benchmark or the model, please cite the paper (see
`CITATION.cff`) and the 40 constituent experimental studies, which are listed
with DOIs in `paper/refs_sources.bib`. The measurements are other people's work.

Primary data: NIST ThermoML archive, [doi:10.18434/mds2-2422](https://doi.org/10.18434/mds2-2422).

## Licence

Code and trained model: MIT (`LICENSE`).
Derived data tables: CC BY 4.0, with the underlying measurements remaining the
property of their original publishers and cited individually.

## Author

Emir Borovac — independent researcher.
