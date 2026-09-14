"""What design detects the recording response the institution implies?

Both applications were silent, and the minimum detectable response of the
specification each was run with exceeded the wedge its recording institution
implies: by a factor of about four in the crime panel and about nineteen in the
health panels.  Reporting that and stopping treats the gap as a fact about the
method.  It is a fact about the specification, and the specification has four
levers this paper already characterises.

  statistic    the directed statistic has a smaller minimum detectable response
               than the two sided Wald whenever the response path is signed
  pre window   I_k = k'([S^{-1}]_PP)^{-1} k rises sharply when the pre window is
               narrowed, and the minimum detectable response falls with its
               square root.  Corollary 1 is what makes this free: the pre period
               carries no information about a recording response, so horizons
               spent there buy nothing the test can use
  restrictions m enters the critical value, so testing the horizons the
               institution implicates rather than all of them lowers it
  categories   Sigma_delta^2 is the spread of the exposure index, and a category
               the recording system has no discretion over anchors its low end

The script computes the minimum detectable response over every combination in
all three application designs, compares it with the wedge each institution
implies, and runs on the data every specification the comparison certifies.

Writes results/inference/design_power.csv and .json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from did_measurement import inference as inf                 # noqa: E402
from oa_onefactor import (load_crime, load_medicaid,         # noqa: E402
                          balanced_subset)

OUT = os.path.join(ROOT, "results", "inference")
if not os.path.isdir(OUT):
    OUT = os.path.join(ROOT, "results", "theory", "inference")

ALPHA, POWER = 0.05, 0.80
N_PERM = N_SIGN = 999
N_BOOT = 1999

CRIME = [1, 2, 3, 4, 5, 6]
MED_ANNUAL = [1, 4, 5, 6, 7]
MED_FOCAL = [1, 2, 3, 4, 5, 6, 7]

# name -> (categories, outcome, (k_pre, k_post) as run, institutional wedge)
DESIGNS = {
    "crime":             (CRIME,      "y", (6, 8), 0.074),
    "medicaid annual":   (MED_ANNUAL, "Y", (5, 6), 0.052),
    "medicaid odd year": (MED_FOCAL,  "Y", (5, 6), 0.052),
}
PRE = {"as run": None, "one": 1}
POST = {"as run": None, "five": 5, "three": 3}
LOADINGS = ("none", "triad")
BRIDGE = {"excluded": False, "included": True}


def frame(name):
    if name == "crime":
        return load_crime()
    cats = MED_ANNUAL if name == "medicaid annual" else MED_FOCAL
    return balanced_subset(load_medicaid(), [0] + cats)


def mde(comp, kind):
    s2 = float(np.mean(comp["resid_var"]))
    return float(inf.mde_kappa(comp, kind=kind, alpha=ALPHA, power=POWER,
                               sigma2=s2,
                               wedge=lambda k: np.log1p(k * comp["d"]))["kappa"])


def main():
    rows = []
    for name, (cats, out, (kp0, kq0), wedge) in DESIGNS.items():
        df = frame(name)
        for pname, kp in PRE.items():
            for qname, kq in POST.items():
                for loading in LOADINGS:
                    for bname, ub in BRIDGE.items():
                        k_pre = kp0 if kp is None else kp
                        k_post = kq0 if kq is None else kq
                        try:
                            comp = inf.gradient_components(
                                df, outcome=out, k_pre=k_pre, k_post=k_post,
                                loading=loading, use_bridge=ub)
                        except Exception:
                            continue
                        di = inf.design_information(comp)
                        for kind in ("wald", "directed"):
                            try:
                                k = mde(comp, kind)
                            except Exception:
                                continue
                            rows.append(dict(
                                design=name, pre_window=pname, k_pre=k_pre,
                                post_window=qname, k_post=k_post,
                                loading=loading, bridge=bname,
                                statistic=kind, C=comp["C"],
                                m=len(comp["post"]),
                                sum_delta2=round(di["sum_delta2"], 4),
                                I_k=round(di["I_k"], 2),
                                G_eff=round(float(inf.effective_clusters(comp)), 2),
                                mde=round(k, 4), wedge=wedge,
                                ratio=round(k / wedge, 3),
                                powered=bool(k <= wedge)))
        best = min((r for r in rows if r["design"] == name),
                   key=lambda r: r["ratio"])
        print("   %-18s best ratio %.3f  (%s pre, %s post, %s, bridge %s, %s)  %s"
              % (name, best["ratio"], best["pre_window"], best["post_window"],
                 best["loading"], best["bridge"], best["statistic"],
                 "POWERED" if best["powered"] else "not powered"), flush=True)

    tab = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(os.path.join(OUT, "design_power.csv"), index=False)

    print("\n   running every certified specification on the data", flush=True)
    live = []
    for r in sorted([x for x in rows if x["powered"]], key=lambda x: x["ratio"]):
        cats, out, _, _ = DESIGNS[r["design"]]
        comp = inf.gradient_components(frame(r["design"]), outcome=out,
                                       k_pre=r["k_pre"], k_post=r["k_post"],
                                       loading=r["loading"],
                                       use_bridge=(r["bridge"] == "included"))
        st = r["statistic"]
        rec = dict(r)
        # the closed form understates the label permutation test's MDE; report
        # the permutation calibrated value too, since that test is the one the
        # paper recommends
        s2 = float(np.mean(comp["resid_var"]))
        mp = inf.mde_kappa_permutation(
            comp, alpha=ALPHA, power=POWER, sigma2=s2,
            wedge=lambda k: np.log1p(k * comp["d"]))
        rec["mde_perm"] = round(float(mp["kappa"]), 4)
        rec["e_perm"] = round(float(mp["e_perm"]), 4)
        rec["ratio_perm"] = round(float(mp["kappa"]) / r["wedge"], 3)
        rec["powered_perm"] = bool(mp["kappa"] <= r["wedge"])
        rec["p_chi2"] = round(float(inf.chi2_wald(comp)["pvalue"]), 4)
        pt = inf.permutation_test(comp, st, N_PERM, 7)
        rec["p_perm"] = round(float(pt["pvalue"]), 4)
        rec["perm_floor"] = round(float(pt["min_pvalue"]), 4)
        rec["p_signflip"] = round(float(
            inf.signflip_test(comp, N_SIGN, 7, st)["pvalue"]), 4)
        rec["p_wcb"] = round(float(
            inf.wild_cluster_bootstrap(comp, N_BOOT, 7, "wald")["pvalue"]), 4)
        live.append(rec)
        print("   %-18s %-8s ratio %.3f (perm %.3f)  chi2 %.4f  perm %.4f  "
              "sign %.4f  wcb %.4f"
              % (r["design"], st, r["ratio"], rec["ratio_perm"], rec["p_chi2"],
                 rec["p_perm"], rec["p_signflip"], rec["p_wcb"]), flush=True)

    with open(os.path.join(OUT, "design_power.json"), "w") as f:
        json.dump({"alpha": ALPHA, "power": POWER,
                   "n_specifications": len(rows),
                   "rows": rows, "certified_runs": live}, f, indent=2)
    print()
    for name in DESIGNS:
        s = tab[tab.design == name]
        print("%-18s specifications %3d   powered %2d   best ratio %.3f"
              % (name, len(s), int(s.powered.sum()), s.ratio.min()))


if __name__ == "__main__":
    main()
