# Cross-Outcome Falsification Tests with Few Clusters: Size Distortion and Randomization Calibrations

Replication material for the paper of the same name.

Ethan Xie, Charter School of Wilmington. Contact: ethanxie@udel.edu

## What the paper shows

Applied researchers test identifying assumptions by asking whether an estimated
treatment effect varies across outcome categories that differ in exposure to a
suspected violation. The cluster-robust Wald statistic for the joint null of no
cross-category gradient is exactly a scaled Hotelling `T^2`, so its chi-square
calibration errs by an amount set by the ratio of restrictions `m` to clusters
`G`. At `G = 51` and `m = 13` a nominal 5% test has size 0.389.

The paper characterizes the size of three alternatives: a label-permutation
test, exact in finite samples for the joint null under category
exchangeability; a cluster sign-change test, exact under sign symmetry of the
cluster contributions and asymptotically level alpha otherwise; and a
restricted wild cluster bootstrap, for which only simulated size is reported.
It also gives a design rule for how many restrictions a given cluster count
supports, and the optimal directed statistic when the response shape is known
up to scale.

## Layout

```
paper/main.tex                     source of the paper
paper/main.pdf                     compiled paper
src/did_measurement/               library
    inference.py                   the inference procedures of Sections 5-7
    estimators.py                  event-study estimators and cluster-robust OLS
    dgp.py                         the two-layer simulation design of Section 8
    mtest.py, bounds.py, bridge.py, diagnostics.py, ...
scripts/theory_inference.py        reproduces Tables 1 and 2 and Figure 1
scripts/theory_inference_guard.py  the loading-guard and eigen-direction checks
scripts/fix_app_floors.py          realized resolution floors for the applications
tests/test_core.py                 test suite (69 checks)
results/theory/inference/          simulation output files the paper cites
results/fixes/empirics/            application output files the paper cites
results/fixes/referee/             the heteroskedastic-permutation size check
```

## Reproducing the tables

Requires Python 3.11 or later.

```
pip install -r requirements.txt
PYTHONPATH=src python scripts/theory_inference.py
PYTHONPATH=src python scripts/theory_inference_guard.py
```

Output is written to `results/theory/inference/`. The note to each table and
figure in the paper names the output file its entries are read from: Table 1
comes from `size_table.csv` and `anatomy_table.csv`, Table 2 from
`power_curves.csv` and `size_adjustment_audit.csv`, Figure 1 from
`anatomy_expansion.csv`, and Table 3 from
`results/fixes/empirics/inference_gradient.csv` and
`results/fixes/empirics/app_floors_injection.csv`.

Size cells use 2,000 replications and power cells 600, at nominal level 0.05.
The label-permutation test enumerates all `C!` assignments at `C = 3` and
`C = 5` and uses 998 sampled assignments plus the identity at `C = 8`; the wild
cluster bootstrap uses 499 draws and the cluster sign-change test 999, each
including the identity draw. Pseudo-random numbers come from the NumPy PCG64
generator, seeded once per replication by the CRC-32 checksum of a string
naming the design cell and the replication index, so any cell can be
regenerated on its own and the tables reproduce exactly. A full run takes about
61 minutes; the script checkpoints per configuration and skips completed cells,
so it can be interrupted and resumed.

## Tests

```
PYTHONPATH=src python tests/test_core.py
```

The suite checks the library against closed-form results, including the
identity `W = omega_2 * T^2` of Proposition 1, the resolution floor of
Proposition 3, the equivariance of the triad normalization, the injection
reference set used when the loading guard fires, and the design rule of
Corollary 1. It ends with `ALL CHECKS PASSED`.

## Note on scope

`src/did_measurement/` is shared with a companion project on
difference-in-differences under treatment-induced recording change, so it
contains modules (`bounds.py`, `bridge.py`, `diagnostics.py`) beyond what this
paper uses. The paper's procedures are in `inference.py`, with the pooled
gradient regression and cluster-robust covariance in `estimators.py`.

The two constructed panels behind Section 9 belong to that companion project
and are not redistributed here; the output files this repository carries are
the ones the paper's numbers are read from.
