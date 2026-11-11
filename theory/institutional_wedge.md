# The institution-implied gradient magnitude

The design screen of Section 10 compares the minimum detectable response with the
magnitude the recording institution implies. This note derives that magnitude for
the crime panel and records why an earlier version of the number was wrong.

## What the right object is

The null under test is `gamma_P = 0` in

    y_itc = a_ic + p_tc + sum_l beta_l X + sum_l gamma_l delta_c X + u_itc ,

so the test is built to detect a response whose cross-category shape is
proportional to the centred exposure index `delta_c = d_c - mean(d)`. If the
recording change produces a log response `m_c` in category `c`, the component of
that response lying in the detectable direction is its projection on the index,

    gamma_inst = sum_c delta_c m_c / sum_c delta_c^2 .

This is the quantity the MDE has to be compared with. Both inputs are external to
the outcome panel: `d_c` comes from the NCVS reporting rates and `m_c` from
published comparisons of the two reporting systems, so the adequacy ratio is
computable before the outcome is examined, which is the point of computing it.

## The crime panel

`d_c` is from `data/data_manifest.json`, field `crime.discretion_d`:

| category | d_c | delta_c | FBI 2014 | m_c | BJS 2000 | m_c |
|---|---|---|---|---|---|---|
| homicide | 0.0000 | -0.5997 | 0.0% | 0.00000 | 0.0% | 0.00000 |
| motor vehicle theft | 0.2898 | -0.3099 | 2.7% | 0.02664 | 4.5% | 0.04402 |
| aggravated assault | 0.6044 | 0.0047 | 0.6% | 0.00598 | 0.5% | 0.00499 |
| robbery | 0.6279 | 0.0282 | 0.6% | 0.00598 | 0.5% | 0.00499 |
| burglary | 0.6992 | 0.0995 | 1.0% | 0.00995 | -0.5% | -0.00501 |
| rape | 0.9765 | 0.3768 | <0.1% | 0.00050 | 0.5% | 0.00499 |
| larceny | 1.0000 | 0.4003 | 2.6% | 0.02567 | 3.4% | 0.03343 |

`sum_c delta_c^2 = 0.7686`, which is the exposure spread the paper reports as
0.769 for the seven-category gradient.

Percent changes are the change in recorded counts when the same incidents are
tabulated under NIBRS rather than SRS. FBI (2014) covers 6,299 NIBRS agencies and
the seven index offences; BJS (2000) covers 1,131 agencies. Both are summarised in
Congressional Research Service report R46668. `m_c = log(1 + pct/100)`.

    gamma_inst  =  0.0044   (FBI 2014)
    gamma_inst  =  0.0017   (BJS 2000)

The headline screen uses 0.0044, the larger of the two and from the larger study,
which is the choice most favourable to declaring a design adequate.

## Why it is so much smaller than the responses themselves

The mean response is about 0.011 and the largest is 0.027, but the projection on
`delta_c` is 0.0044. The reason is that the two are nearly orthogonal:

    corr(delta_c, m_c) = 0.14   (FBI 2014)
    corr(delta_c, m_c) = 0.03   (BJS 2000)

`d_c` grades categories by the share of victimisations *not* reported to police,
so it orders them by victim reporting propensity. The SRS-to-NIBRS change does not
act on victim reporting at all. It acts on police counting rules: the hierarchy
rule is dropped and co-occurring offences become countable. Those two orderings
come apart, and in places reverse. Motor vehicle theft has the second lowest `d_c`
and among the largest responses, because vehicle theft often co-occurs with
another index offence. Rape has nearly the highest `d_c` and essentially no
response, because rape rarely co-occurs with a more serious index offence.

So the crime design is misaligned rather than merely underpowered. The response
the institution implies is real, but almost none of it lies in the direction the
gradient test can see. That is checkable in advance, at the cost of one inner
product, and it is the fifth lever of Section 10.

## The number this replaces

An earlier version of `scripts/oa_design.py` carried `0.074` for the crime panel.
That is the largest value of `|coef| / d_c` across the six categories, which
`src/did_measurement/bounds.py` computes as `breakdown_kappa`'s `point` value and
`results/empirics/crime/bounds.csv` reports as `breakdown_kappa_point`: motor
vehicle theft, `0.0216 / 0.2898 = 0.0744`. The range quoted as the institutional
bounds, 0.008 to 0.074, is the smallest and largest of that same column, larceny
to motor vehicle theft.

A breakdown value is a statement about the fitted post-treatment means, not about
the recording institution. Using it as the benchmark made the adequacy screen a
function of the outcome it was supposed to screen, which contradicts the claim
that the ratio is knowable without looking at the outcome, and it made the
resulting exclusion close to circular.

## Reproducing

    PYTHONPATH=src python scripts/oa_design.py

writes `results/inference/design_power.{csv,json}`. With the derived magnitude no
specification of any of the three designs is adequate, the best ratios being 11.8
for crime, 12.5 for the annual health sub design and 14.8 for the odd-year one.
