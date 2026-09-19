"""Move the conclusion's anomaly discussion to the supplement.

The evidence behind these three paragraphs is already in the supplement, so the
conclusion keeps the finding and the supplement keeps the working.

Run from paper2/.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main.tex')
SUPP = os.path.join(HERE, 'supplement.tex')

t = io.open(MAIN, encoding='utf-8').read()

i = t.index('Three findings in Section \\ref{sec:sim} that an earlier draft reported')
j = t.index('Lemma \\ref{lem:sep} requires a category balanced design')
block = t[i:j]
assert '\\begin{table}' not in block and '\\begin{figure}' not in block
print('   moved %d words from the conclusion' % len(block.split()))

SUMMARY = (
    'Three findings an earlier draft of Section \\ref{sec:sim} reported without '
    'a cause are resolved by replication, and the supplement gives the working. '
    'The decline in measured chi square size across the number of categories '
    'does not reproduce in an independent run and is not the loading '
    'normalization. The nonmonotone power at five categories is a plateau '
    'rather than a turning point: the realized signal is monotone in the '
    'response magnitude throughout, and what the permutation test has there is '
    'a ceiling set by a reference set of $120$ points, which is also why the '
    'large sample local power calculation overstates it. The Kolmogorov '
    'Smirnov rejection in the ideal experiment is what a correctly sized test '
    'does one time in twenty, at rejection rates of $0.060$, $0.045$, $0.035$ '
    'and $0.060$ over $200$ independent trials.\n\n'
)
t = t[:i] + SUMMARY + t[j:]
t = re.sub(r'\n{4,}', '\n\n', t)
io.open(MAIN, 'w', encoding='utf-8', newline='').write(t)

s = io.open(SUPP, encoding='utf-8').read()
add = ('\\section{Three findings resolved by replication}\n\n' + block.strip())
s = s.replace('\\begin{thebibliography}', add + "\n\n\\begin{thebibliography}", 1)
io.open(SUPP, 'w', encoding='utf-8', newline='').write(s)
print('   supplement updated')
