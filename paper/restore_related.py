"""Put a compressed related work section back into the main paper.

Moving the section wholesale was too much.  A referee needs the paper's position
against the closest prior work in the paper itself, in particular what
Proposition 2 adds to the small sample cluster robust literature it sits in.
The extended discussion stays in the supplement; this is the part that has to be
in front of the reader.

Any bibitem the restored text cites is copied back from the pre-prune backup.

Run from paper2/.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main.tex')
BACKUP = r'C:\Users\xief\AppData\Local\Temp\main_b4prune.tex'

t = io.open(MAIN, encoding='utf-8').read()

SECTION = r"""\section{Related work}

Our exact null law concerns the cluster robust Wald statistic, which places it
in the few cluster literature. \citet{cameron2008bootstrap} propose the wild
cluster bootstrap with the null imposed and \citet{mackinnon2017wild}
characterize when it fails; neither treats the joint case in which the number of
restrictions is comparable to the number of clusters, which is where cross
outcome falsification tests live. \citet{carter2017cluster} define the effective
cluster count used in Section \ref{sec:anatomy}. \citet{canay2017randomization}
give asymptotic validity for a randomization test on a group of fixed
cardinality; the label permutation group of Section \ref{sec:perm} has
cardinality $C!$, which does not vary with the number of clusters. The finite
sample argument is the classical group invariance construction of
\citet{lehmann2005testing}.

We are not the first to correct both the variance estimator and the reference
distribution. \citet{bester2011cluster} hold the number of clusters fixed and
let cluster size grow, and the $F(m,G-m)$ shape of \eqref{eq:Fexact} is theirs
as a large cluster approximation. \citet{bell2002bias} remove the downward bias
of the cluster robust variance estimator and pair the correction with
Satterthwaite degrees of freedom, and \citet{imbens2016robust} give degrees of
freedom adjustments for the heteroskedasticity robust case. For a hypothesis
with several constraints \citet{tipton2015small} and \citet{pustejovsky2018small}
refer the statistic to an approximate Hotelling law whose denominator degrees of
freedom are matched from the data. Proposition \ref{prop:anatomy} does not
introduce that family. It states an algebraic identity and, under Assumption
\ref{as:I}, an exact law holding at every cluster count and every number of
restrictions with $G - m \ge 1$, rather than at matched degrees of freedom or in
a large cluster limit. What we add is the finite sample statement and the
explicit constant $\omega_2$. Corollary \ref{cor:asym} adds the rate at which
the chi square calibration stays valid.

The design that motivates the problem is the use of graded outcome categories to
detect a treatment induced change in recording. \citet{weisburd2024crs} run this
contrast across crime categories. \citet{hausman1998misclassification} formalize
outcome misclassification in a cross section, and \citet{fhs2019pretrends} use an
auxiliary series loading on the same latent confound to purge event study
coefficients of it, a device close to the normalization Assumption \ref{as:CE}
calls for and put to a different use. Falsification tests are also run as pre
trend tests, where \citet{roth2022pretest} shows that conditioning on passing one
distorts the sampling distribution of the effect estimate and
\citet{rambachan2023credible} replace the test with bounds. The supplement gives
the extended discussion. This paper is about the reference distribution.

"""

anchor = '\\section{What the falsification test is for}'
i = t.index(anchor)
t = t[:i] + SECTION + t[i:]

# restore
back = io.open(BACKUP, encoding='utf-8').read()
bib_all = back[back.index('\\begin{thebibliography}'):
               back.index('\\end{thebibliography}')]
items = re.findall(r'\\bibitem\[.*?(?=\\bibitem\[|\Z)', bib_all, flags=re.S)
by_key = {re.search(r'\]\{([^}]+)\}', x).group(1): x.rstrip() for x in items}

body = t[:t.index('\\begin{thebibliography}')]
keys = {k.strip() for g in re.findall(r'\\cite[a-z]*\{([^}]+)\}', body)
        for k in g.split(',')}
have = set(re.findall(r'\\bibitem\[[^\]]*\]\{([^}]+)\}', t))
need = sorted(keys - have)
missing = [k for k in need if k not in by_key]
assert not missing, 'no entry available for %s' % missing

a = t.index('\\begin{thebibliography}')
b = t.index('\\end{thebibliography}') + len('\\end{thebibliography}')
allk = sorted(have | set(need))
pool = dict(by_key)
for k in re.findall(r'\\bibitem\[.*?(?=\\bibitem\[|\\end\{thebibliography\})',
                    t[a:b], flags=re.S):
    kk = re.search(r'\]\{([^}]+)\}', k).group(1)
    pool[kk] = k.rstrip()
new_bib = ("\\begin{thebibliography}{99}\n\\itemsep 3pt\n\n"
           + "\n\n".join(pool[k] for k in allk if k in keys)
           + "\n\n\\end{thebibliography}")
t = t[:a] + new_bib + t[b:]

io.open(MAIN, 'w', encoding='utf-8', newline='').write(t)
print('related work restored: %d words' % len(SECTION.split()))
print('bibitems restored: %s' % (need or 'none'))
print('main bibliography now: %d entries' % len([k for k in allk if k in keys]))
