"""One call for the procedure the paper recommends.

The package exposes the pieces of the argument separately, which is what the
replication scripts need but not what a practitioner needs. This module is the
single entry point: give it a panel and the recording wedge the institution
implies, and it answers the two questions in the order the paper says to ask
them.

    1. Can this design detect the response the institution implies, before any
       outcome is examined?  That is the design calculus: the minimum detectable
       response against the institutional magnitude, the effective cluster
       count, and the size a nominal chi square test would actually have.

    2. If it can, does the falsification test reject?  The reported p value
       comes from a randomisation calibration, never from the chi square one,
       and the choice among them follows the paper's own recommendation.

Usage:

    from did_measurement import screen_design, falsification_test

    d = screen_design(panel, kappa_inst=0.074)
    if d.adequate:
        r = falsification_test(panel, kappa_inst=0.074)
        print(r.pvalue, r.calibration)
"""
from dataclasses import dataclass, field
from math import factorial
from typing import Optional

from .inference import (design_information, effective_clusters,
                        gradient_components, gradient_inference, mde_kappa,
                        predicted_wald_size)

__all__ = ["screen_design", "falsification_test", "DesignVerdict", "TestResult"]

# Below this many categories the two sided label permutation p value cannot fall
# under the level, whatever the data: the smallest attainable value is 2/C!.
_MIN_C_FOR_LABEL_PERMUTATION = 5


@dataclass
class DesignVerdict:
    """What the design can detect, computed without looking at an outcome."""
    adequate: bool
    kappa_mde: float
    kappa_inst: Optional[float]
    ratio: Optional[float]
    C: int
    G: int
    G_eff: float
    m: int
    I_k: float
    sum_delta2: float
    omega2: float
    chi2_size: float
    resolution_floor: float
    label_permutation_usable: bool
    notes: list = field(default_factory=list)

    def __str__(self):
        head = 'adequate' if self.adequate else 'NOT adequate'
        lines = ['design is %s' % head,
                 '  kappa_MDE      %.4f' % self.kappa_mde]
        if self.kappa_inst is not None:
            lines.append('  kappa_inst     %.4f  (ratio %.3f, adequate when <= 1)'
                         % (self.kappa_inst, self.ratio))
        lines += ['  C, G, m        %d, %d, %d' % (self.C, self.G, self.m),
                  '  G_eff          %.2f' % self.G_eff,
                  '  chi square size at nominal 0.05 would be %.3f' % self.chi2_size,
                  '  label permutation floor 2/C! = %.4f%s'
                  % (self.resolution_floor,
                     '' if self.label_permutation_usable else '  (unusable)')]
        return "\n".join(lines + ['  note: ' + n for n in self.notes])


@dataclass
class TestResult:
    """The falsification test, reported through a randomisation calibration."""
    pvalue: float
    calibration: str
    design: DesignVerdict
    all_pvalues: dict
    notes: list = field(default_factory=list)

    def __str__(self):
        lines = ['p = %.4f  (%s)' % (self.pvalue, self.calibration),
                 '  other calibrations: '
                 + ", ".join('%s %.4f' % (k, v)
                             for k, v in sorted(self.all_pvalues.items()))]
        return "\n".join(lines + ['  note: ' + n for n in self.notes])


def _components(df, **kw):
    keys = ('outcome', 'k_pre', 'k_post', 'bridge_cat', 'use_bridge',
            'cluster_col', 'loading')
    return gradient_components(df, **{k: v for k, v in kw.items() if k in keys})


def screen_design(df, kappa_inst=None, alpha=0.05, power=0.80, k=None, **kw):
    """Answer question 1: can this design see the response the institution implies?

    `kappa_inst` is the recording wedge the institution implies, on the scale of
    the outcome per unit of discretion. Without it the minimum detectable
    response is still reported, but adequacy cannot be decided and `adequate`
    is False with a note saying why.
    """
    comp = _components(df, **kw)
    info = design_information(comp, k=k)
    G, m, C = int(comp['G']), len(comp['post']), int(comp['C'])
    n, q = int(comp['n']), int(comp['q'])
    G_eff = effective_clusters(comp)
    omega2 = (n - 2 * q) / (n - 1)

    # price the minimum detectable response at the calibration a correctly sized
    # procedure uses, not at the chi square critical value the paper rejects
    kappa_mde = float(mde_kappa(comp, kind='wald', alpha=alpha, power=power,
                                k=k, calibration='hotelling',
                                G_eff=G_eff)['kappa'])
    chi2_size = float(predicted_wald_size(G_eff, m, alpha=alpha, omega2=omega2,
                                          calibration='chi2'))

    floor = 2.0 / factorial(C)
    usable = C >= _MIN_C_FOR_LABEL_PERMUTATION and floor <= alpha

    notes = []
    if not usable:
        notes.append('with C = %d the smallest attainable two sided label '
                     'permutation p value is %.4f, so that test cannot reject '
                     'at level %.2f; use the cluster level procedures'
                     % (C, floor, alpha))
    if m >= G_eff - 1:
        notes.append('m = %d against G_eff = %.2f leaves too few degrees of '
                     'freedom for the F calibration to be informative' % (m, G_eff))
    if kappa_inst is None:
        notes.append('no institutional magnitude supplied, so adequacy is undecided')

    ratio = None if kappa_inst is None else float(kappa_mde / kappa_inst)
    return DesignVerdict(
        adequate=bool(ratio is not None and ratio <= 1.0),
        kappa_mde=kappa_mde, kappa_inst=kappa_inst, ratio=ratio,
        C=C, G=G, G_eff=float(G_eff), m=m,
        I_k=float(info['I_k']), sum_delta2=float(info['sum_delta2']),
        omega2=float(omega2), chi2_size=chi2_size,
        resolution_floor=floor, label_permutation_usable=bool(usable),
        notes=notes)


def falsification_test(df, kappa_inst=None, alpha=0.05, power=0.80, k=None,
                       clustered_units=False, seed=0, **kw):
    """Answer question 2, after answering question 1.

    Runs the design screen first and carries it in the result, because a p value
    from a design that cannot detect the response is not evidence of absence.

    `clustered_units` should be True when a cluster holds many units rather than
    one. The directed cluster sign change test over rejects in that case, at
    measured sizes of 0.094 and 0.086 against a nominal 0.05, so it is not
    chosen as the reported calibration and a note says so.
    """
    design = screen_design(df, kappa_inst=kappa_inst, alpha=alpha, power=power,
                           k=k, **kw)

    methods = ('chi2', 'perm_wald', 'perm_dir', 'wcb_wald', 'signflip')
    res = gradient_inference(df, methods=methods, seed=seed,
                             **{kk: vv for kk, vv in kw.items()
                                if kk in ('outcome', 'k_pre', 'k_post',
                                          'bridge_cat', 'use_bridge',
                                          'cluster_col', 'loading')})
    pvals = {m: float(p) for m, p in res['p'].items()}

    notes = []
    if design.label_permutation_usable:
        pick, why = 'perm_wald', ('label permutation, exact in finite samples '
                                  'under category exchangeability')
    else:
        pick, why = 'wcb_wald', ('restricted wild cluster bootstrap, because '
                                 'the label permutation floor rules that test out')
    if clustered_units:
        notes.append('clusters hold several units, so the directed sign change '
                     'calibration is not reported; it over rejects in that design')
    if not design.adequate:
        notes.append('the design screen did not certify this specification, so a '
                     'non rejection is not evidence that no recording response '
                     'occurred')
    notes.append('the chi square p value is reported for comparison only; at '
                 'this design its null rejection rate is %.3f against a nominal '
                 '%.2f' % (design.chi2_size, alpha))

    return TestResult(pvalue=pvals[pick], calibration=why, design=design,
                      all_pvalues=pvals, notes=notes)
