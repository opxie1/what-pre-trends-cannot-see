"""Move the nine-design robustness table and the health detail to the supplement.

Table 4 varies the crime design nine ways across three normalisations; it is
robustness evidence, and the main text needs only the baseline it already
reports in prose.  The health panels are screened out by the design calculus
before any test is run, so their normalisation-by-normalisation reporting
belongs with the other secondary evidence.

Run after move_blocks.py, from paper2/.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main.tex')
SUPP = os.path.join(HERE, 'supplement.tex')

t = io.open(MAIN, encoding='utf-8').read()
moved = []


def cut(start, end_marker, label, include_end=True):
    """Remove t[i:j] where i starts at `start` and j ends at `end_marker`."""
    global t
    i = t.find(start)
    assert i >= 0, 'start not found: %s' % label
    j = t.find(end_marker, i)
    assert j >= 0, 'end not found: %s' % label
    if include_end:
        j += len(end_marker)
    block = t[i:j]
    t = t[:i] + t[j:]
    moved.append((label, block))
    print('   moved %5d words  %s' % (len(block.split()), label))
    return block


# ---- 1. Table 4, the nine design robustness table, with its note
i = t.find('\\label{tab:appdesigns}')
assert i > 0
a = t.rfind('\\begin{table}', 0, i)
b = t.find('\\end{table}', i) + len('\\end{table}')
tab = t[a:b]
t = t[:a] + t[b:]
moved.append(('Table: crime application across nine designs', tab))
print('   moved %5d words  Table 4 with its note' % len(tab.split()))

# ---- 2. the health reporting, four paragraphs
health = cut('The second panel is a state by year by condition panel',
             'this is what it looks like in data.',
             'health panel reporting')

# ---- 3. two crime robustness paragraphs
cut('The chi square rejection is joint with the normalization.',
    'is conditional on the normalization.',
    'crime normalisation and placebo detail')

SUMMARY = (
    'The second panel is a state by year by condition panel of BRFSS crude '
    'prevalence covering $51$ jurisdictions from 2011 to 2024, clustered on '
    'state, with seven conditions graded by whether a clinical encounter is '
    'needed for the respondent to report the condition, and treatment given by '
    'a state\'s adoption of the ACA Medicaid expansion. Section '
    '\\ref{sec:designcalc} screens it out before any test is run, at best '
    'ratios of $12.5$ and $14.8$ in the two balanced sub designs, and two '
    'further features of it are reported in the supplement: Assumption '
    '\\ref{as:B} fails on the seven category stack because hypertension and '
    'high cholesterol are asked only in odd years, and every rejection '
    'anywhere in these panels comes from the one normalization Theorem '
    '\\ref{thm:exact} does not cover and none survives the correct reference '
    'set.\n'
)
k = t.find('The wedge each institution implies is what Section')
assert k > 0, 'anchor for the health summary not found'
t = t[:k] + SUMMARY + '\n' + t[k:]

t = re.sub(r'\n{4,}', '\n\n', t)
io.open(MAIN, 'w', encoding='utf-8', newline='').write(t)

s = io.open(SUPP, encoding='utf-8').read()
add = ['\\section{Additional application evidence}', '']
for label, block in moved:
    add.append(block.strip())
block = "\n\n".join(add)
anchor = '\\begin{thebibliography}'
if anchor in s:
    s = s.replace(anchor, block + "\n\n" + anchor, 1)
else:
    s = s.replace('\\end{document}', block + "\n\n\\end{document}", 1)
io.open(SUPP, 'w', encoding='utf-8', newline='').write(s)

print()
print('moved %d items, %d words' % (len(moved), sum(len(b.split()) for _, b in moved)))
