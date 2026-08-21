# Exact Randomization Inference for Cross-Outcome Falsification Tests with Few Clusters

Replication material for the paper of the same name.

Ethan Xie, Charter School of Wilmington. Contact: ethanxie@udel.edu

## What the paper shows

Applied researchers test identifying assumptions by asking whether an estimated
treatment effect varies across outcome categories that differ in exposure to a
suspected violation. The cluster-robust Wald statistic for the joint null of no
cross-category gradient is exactly a scalar multiple of a Hotelling `T^2`, so
its chi-square calibration is wrong by an amount governed by the ratio of
restrictions `m` to clusters `G`. At `G = 51` and `m = 13` the nominal 5% test
rejects a true null 38.9% of the time.

The paper gives three correctly sized alternatives (a label-permutation test
that is exact in finite samples under category exchangeability, a cluster
sign-change test, and a restricted wild cluster bootstrap), a design rule
stating how many restrictions a given cluster count supports, and the optimal
directed statistic.

## Layout

```
paper/main.tex                     source of the paper
paper/main.pdf                     compiled paper
src/did_measurement/               library
    inference.py                   the inference procedures of Sections 5-7
    estimators.py                  event-study estimators and cluster-robust OLS
    dgp.py                         the two-layer simulation design of Section 8
    mtest.py, bounds.py, bridge.py, diagnostics.py, ...
scripts/theory_inference.py        reproduces every table in the paper
tests/test_core.py                 test suite (64 checks)
results/theory/inference/          output files the paper cites by name
```

## Reproducing the tables

Requires Python 3.11 or later.

```
pip install -r requirements.txt
PYTHONPATH=src python scripts/theory_inference.py
```

Output is written to `results/theory/inference/`. Every number reported in the
paper names the file it comes from; the two main tables come from
`size_table.csv` (Table 1) and `power_curves.csv` (Table 2).

Size cells use 2,000 replications and power cells use 600. Seeds are fixed, so
the tables reproduce exactly. A full run takes on the order of an hour on a
laptop; the script checkpoints per configuration and skips completed cells, so
it can be interrupted and resumed.

## Tests

```
PYTHONPATH=src python tests/test_core.py
```

The suite checks the library against closed-form results, including the
identity `W = omega_2 * T^2` of Proposition 1, the resolution floor of
Proposition 2, and the design rule of Corollary 1. It ends with
`ALL CHECKS PASSED`.

## Note on scope

`src/did_measurement/` is shared with a companion project on
difference-in-differences under treatment-induced recording change, so it
contains modules (`bounds.py`, `bridge.py`, `diagnostics.py`) beyond what this
paper uses. The paper's procedures are in `inference.py`, with the pooled
gradient regression and cluster-robust covariance in `estimators.py`.
