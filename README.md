# What Pre-Trends Cannot See: Testing for Treatment-Induced Recording Change with Few Clusters

Replication material for the paper of the same name.

Ethan Xie, Charter School of Wilmington. Contact: ethanxie@udel.edu

## What the paper shows

When a policy changes how an outcome is recorded rather than the outcome
itself, the recorded effect confounds the two. Campbell and Stanley (1963)
named this the instrumentation threat. For a single-series design they marked
it as the one threat a pre-period cannot dispel; for the multiple time-series
design, the ancestor of difference-in-differences, they cleared it on reasoning
that covers only instrument differences arising from how groups were selected.
A recording change caused by the treatment is not such a difference, and no
control series absorbs it, because control units never receive the treatment
that induces it.

Section 3 states the consequence. The parallel-trends deviation that reproduces
a recording response is exactly zero before treatment, so any diagnostic
computed from the observed panel and the cohort labels rejects with probability
equal to its size. Sensitivity analysis inherits the blindness: the
relative-magnitudes class of Rambachan and Roth scales a pre-period quantity
that is identically zero here, so no choice of the sensitivity parameter admits
any deviation and no breakdown value exists. Computed with a reference
implementation, coverage of the latent effect falls from 0.955 at N = 400 to
0.000 by N = 25,600, while a bound motivated by the recording institution holds
coverage at 1.000 throughout.

Information has to come from somewhere other than the pre-period. The remainder
of the paper takes it from the cross-section of outcome categories, which
differ in how much discretion the recording system has over them, and addresses
the inference problem that arises when clusters are few. The cluster-robust Wald
statistic for the joint null of no cross-category gradient is exactly a scaled
Hotelling `T^2` under equal cluster design weights, so its chi-square
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

## What the paper does not claim

That a policy can change the record rather than the outcome is Campbell's
observation, not this paper's. That a smoothness constant cannot be calibrated
from data while retaining honest coverage is Low (1997) and Cai and Low (2004),
stated for econometric practice by Armstrong and Kolesar. Arora and Chand
(2026) note the relative-magnitudes mechanism in prose. What is claimed is the
mechanism that generates an exactly-zero pre-period deviation, the resulting
statement about rejection probabilities, and the calibration of a test that
works where the pre-period does not.

Proposition 2 requires equal cluster design weights, which no design here
satisfies; it enters as a diagnostic and as the source of a design rule, not as
a calibration the paper recommends. The corollary concerns diagnostics computed
from the outcome panel, and says nothing against identifying a recording
response from data on the recording process itself.

## Layout

```
paper/main.tex                     the manuscript, with author details
paper/main_anon.tex                the anonymous manuscript for review
paper/supplement.tex               the online Appendix
paper/supplement_anon.tex          the anonymous online Appendix
paper/build_supplement.py          splits the proofs out of main.tex
src/did_measurement/               estimators, inference, diagnostics
src/did_measurement/api.py         screen_design and falsification_test
scripts/                           analysis and verification scripts
scripts/coverage_zero.py           coverage under a hand-built interval
scripts/coverage_zero_ref.py       coverage under a reference implementation
scripts/oa_studentized.py          exchangeability failure, priced two ways
scripts/oa_onefactor.py            is the one-factor premise of (6.1) credible
scripts/oa_onefactor_calib.py      calibration check for the above
scripts/oa_staggered.py            does the blindness result survive staggering
scripts/crime.py                   builds the crime panel
scripts/medicaid.py                builds the two health panels
data/processed/crime_panel.csv     the constructed crime panel
data/processed/medicaid_panel.csv  the constructed health panel
data/data_manifest.json            every series, its source and its vintage
results/inference/                 simulated size and power
results/coverage/                  coverage under a recording response
results/empirics/                  application p-values and floors
results/inference/design_power.csv  the 48 screened specifications per panel,
                                   with minimum detectable responses and
                                   adequacy ratios
theory/institutional_wedge.md      how kappa_inst is derived, and why the
                                   number this replaces was wrong
tests/test_core.py                 69 checks
```

## Using the method on your own panel

Install the package and call it once. The entry point answers the two questions
in the order the paper says to ask them: whether the design can detect the
response the institution implies, and only then whether the test rejects.

```
pip install -e .
```

```python
from did_measurement import screen_design, falsification_test

# kappa_inst is the recording wedge the institution implies, per unit of
# discretion, on the scale of the outcome.
verdict = screen_design(panel, kappa_inst=0.0044)
print(verdict)          # minimum detectable response, G_eff, chi square size

if verdict.adequate:
    result = falsification_test(panel, kappa_inst=0.0044)
    print(result.pvalue, result.calibration)
```

The reported p-value always comes from a randomisation calibration. The chi
square p-value is carried in `result.all_pvalues` for comparison and is never
the one reported, because its null rejection rate at these designs is the
distortion the paper is about. When the number of categories puts the label
permutation floor above the level, the entry point switches to the bootstrap and
says so. A non-rejection from a design the screen did not certify is flagged as
what it is.

## Reproducing

```
pip install -e ".[replication]"
PYTHONPATH=src python tests/test_core.py
PYTHONPATH=src python scripts/coverage_zero_ref.py
PYTHONPATH=src python scripts/oa_staggered.py
```

The manuscript is set to the ASA style: 12 point, one inch margins, double
spaced throughout. An anonymous twin of each document is kept for journals that
use double-anonymous review, and is not needed where review is single-anonymous.
The four files build independently and in any order. Nothing cross-references
between the paper and the online Appendix at the LaTeX level, because some
journals forbid it, so the Appendix carries the numbers it cites rather than
resolving them; results keep the paper's numbering, while sections, tables and
figures belonging to the Appendix carry an S.

`oa_studentized.py` and `oa_onefactor_calib.py` are self-contained and run from
a clean checkout. `oa_onefactor.py` and `oa_design.py` read the two application
panels from `data/processed/`, which are included here.

## The application panels

`data/processed/crime_panel.csv` and `data/processed/medicaid_panel.csv` are the
two constructed panels, and `scripts/crime.py` and `scripts/medicaid.py` build
them. Every series in both is a public use file, and
`data/data_manifest.json` records each one with its source and vintage: FBI
summary based estimated state totals and NIBRS adoption dates from the Crime
Data Explorer agency API, reporting rates from Bureau of Justice Statistics,
*Criminal Victimization 2019*, NCJ 255113, Table 6, BRFSS crude prevalence from
CDC Socrata dataset `dttw-5yxu`, and Medicaid expansion effective dates from the
Kaiser Family Foundation table of state decisions.

The government series the builders read are cached under `data/raw/`, so both
panels rebuild offline from a clean checkout and reproduce the vintages the
paper reports, which `data_manifest.json` dates to 2026-08-13:

```
PYTHONPATH=src python scripts/crime.py
PYTHONPATH=src python scripts/medicaid.py
```

The Medicaid expansion effective dates are not fetched. They are hardcoded in
`scripts/medicaid_lib.py` from the KFF adoption table, and the third-party pages
used to check them are not redistributed here.

The p-value columns written by `oa_onefactor.py` are not calibrated references
and the paper does not report them. `oa_onefactor_calib.py` generates data from
a true one-factor model and finds those references reject between 0.090 and
0.995 on the eigenvalue ratio and between 0.090 and 0.178 on the triad
dispersion, against a nominal 0.05, because the null is fitted to the data. The
paper reports the descriptive columns only.

The label-permutation test enumerates all `C!` assignments at `C = 3` and
samples otherwise. Every table in the paper can be regenerated on its own and
reproduces exactly. A full run takes about an hour on a laptop; the coverage
scripts at the largest sample sizes take longer.
