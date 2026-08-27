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

Proposition 1 requires equal cluster design weights, which no design here
satisfies; it enters as a diagnostic and as the source of a design rule, not as
a calibration the paper recommends. The corollary concerns diagnostics computed
from the outcome panel, and says nothing against identifying a recording
response from data on the recording process itself.

## Layout

```
paper/                             main.tex and the compiled PDF
src/did_measurement/               estimators, inference, diagnostics
scripts/                           analysis and verification scripts
scripts/coverage_zero.py           coverage under a hand-built interval
scripts/coverage_zero_ref.py       coverage under a reference implementation
results/theory/inference/          simulated size and power
results/fixes/referee/             the heteroskedastic-permutation size check
tests/test_core.py                 69 checks
```

## Reproducing

```
pip install -r requirements.txt
python tests/test_core.py
python scripts/coverage_zero_ref.py
```

The label-permutation test enumerates all `C!` assignments at `C = 3` and
samples otherwise. Every table in the paper can be regenerated on its own and
reproduces exactly. A full run takes about an hour on a laptop; the coverage
scripts at the largest sample sizes take longer.
