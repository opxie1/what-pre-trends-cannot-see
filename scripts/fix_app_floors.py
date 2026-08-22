"""Realised permutation resolution floors and injection-reference p-values for
the Section 9 applications.

The min_attainable_p column of results/fixes/empirics/inference_gradient.csv was
written before the floor was computed from the realised orbit multiplicity, so
it records the naive 1/M.  This script recomputes, for every application design
and loading normalisation, (a) the realised floor mult/M from
permutation_test's orbit_multiplicity, (b) whether rejection at 5% was
attainable, and (c) the p-value under the ordered-injection reference set that
Assumption CE requires when the loading guard has dropped categories.

Run: PYTHONPATH=src python scripts/fix_app_floors.py
Output: results/fixes/empirics/app_floors_injection.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_empirics_lib import OUT, load_crime, load_medicaid, balanced_subset
from fix_empirics import crime_designs, CK_PRE, CK_POST, MK_PRE, MK_POST, \
    MED_FOCAL, MED_ANNUAL
from did_measurement.inference import (gradient_components, permutation_test,
                                       chi2_wald, effective_clusters)

SEED = 20260820
NPERM = 5040


def _rows(app, design, spec, comp, blocks):
    out = []
    for bname, idx in blocks:
        for ref in ("labels", "injection"):
            r = permutation_test(comp, "wald", NPERM, SEED, idx=idx,
                                 reference=ref)
            d = np.asarray(comp["d"], float)
            out.append({
                "application": app, "design": design, "spec": spec,
                "block": bname, "reference": ref,
                "C": comp["C"], "C_full": comp["C_full"],
                "G": comp["G"], "G_eff": round(effective_clusters(comp), 3),
                "pvalue": r["pvalue"], "n_perm": r["n_perm"],
                "exhaustive": r["exhaustive"],
                "orbit_multiplicity": r["orbit_multiplicity"],
                "min_pvalue_realised": r["min_pvalue"],
                "min_pvalue_naive": r["min_pvalue_naive"],
                "rejection_attainable_05": r["rejection_attainable"],
                "exact_under_CE": r["exact_under_CE"],
                "equivariant_loading": r["equivariant_loading"],
                "guard_fired": r["guard_fired"],
                "dropped": ";".join(str(x["cat"]) for x in comp["dropped"]),
                "sum_delta2": float(((d - d.mean()) ** 2).sum()),
                "delta_symmetric_about_zero": bool(
                    np.allclose(np.sort(np.round(d - d.mean(), 9)),
                                np.sort(np.round(-(d - d.mean()), 9)))),
                "chi2_pvalue": chi2_wald(comp, idx=idx)["pvalue"],
            })
    return out


def main():
    rows = []
    cr = load_crime()
    for tag, dat in crime_designs(cr):
        for loading, lname in (("mtest_iv", "adjusted"), ("none", "naive"),
                               ("triad", "triad")):
            comp = gradient_components(dat, outcome="y", k_pre=CK_PRE,
                                       k_post=CK_POST, bridge_cat=0,
                                       use_bridge=False, loading=loading)
            rows += _rows("crime", tag, lname, comp,
                          [("post", None), ("pre", comp["pre"])])
    md = load_medicaid()
    for tag, cats in (("annual5", MED_ANNUAL), ("odd7", MED_FOCAL)):
        dat = balanced_subset(md, cats)
        for outcome in ("Y", "ylog"):
            for loading, lname in (("mtest_iv", "adjusted"), ("none", "naive"),
                                   ("triad", "triad")):
                comp = gradient_components(dat, outcome=outcome,
                                           k_pre=MK_PRE, k_post=MK_POST,
                                           bridge_cat=0, use_bridge=False,
                                           loading=loading)
                rows += _rows("medicaid", tag, f"baseline|{outcome}|{lname}",
                              comp, [("post", None), ("pre", comp["pre"])])
    tab = pd.DataFrame(rows)
    path = os.path.join(OUT, "app_floors_injection.csv")
    tab.to_csv(path, index=False)
    print(f"wrote {path}: {len(tab)} rows")


if __name__ == "__main__":
    main()
