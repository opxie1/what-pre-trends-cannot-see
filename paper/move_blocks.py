"""Move secondary evidence out of the main text and into the supplement.

Each block is a \\paragraph unit lifted verbatim.  Extraction stops at the next
\\paragraph, \\section, \\appendix or float, so tables and figures that sit
inside a block stay in the main paper where they belong.

Run after build_supplement.py, from paper2/.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main.tex')
SUPP = os.path.join(HERE, 'supplement.tex')

# paragraph heading -> section of the supplement it joins
MOVE = [
    ('The exact null law in the ideal experiment.', 'sim'),
    ('Accuracy of the substitution of $G_{\\mathrm{eff}}$.', 'sim'),
    ('When exchangeability fails.', 'sim'),
    ('The price of label randomization.', 'sim'),
    ('Exposure spread, panel length and the pre window.', 'sim'),
    ('Robustness.', 'sim'),
    ('A bias reduced variance does not repair the calibration.', 'sim'),
]

STOP = re.compile(r'\\paragraph\{|\\section\{|\\section\*\{|\\subsection\{'
                  r'|\\appendix|\\begin\{table\}|\\begin\{figure\}')

t = io.open(MAIN, encoding='utf-8').read()
moved, taken = [], 0

for head, _ in MOVE:
    tag = '\\paragraph{' + head + '}'
    i = t.find(tag)
    if i < 0:
        print('   *** not found: %s' % head[:48])
        continue
    m = STOP.search(t, i + len(tag))
    j = m.start() if m else len(t)
    block = t[i:j].rstrip()
    # a float must never be swept along
    assert '\\begin{table}' not in block and '\\begin{figure}' not in block, head
    moved.append(block)
    taken += len(block.split())
    t = t[:i] + t[j:]
    print('   moved %5d words  %s' % (len(block.split()), head[:52]))

# tidy the blank lines the removals leave behind
t = re.sub(r'\n{4,}', '\n\n', t)

NOTE = (
    '\\paragraph{Further evidence.} The supplement reports five further blocks '
    'of this design: the exact null law checked in an experiment built to '
    'satisfy Assumption \\ref{as:I}, the accuracy of the substitution of '
    '$G_{\\mathrm{eff}}$, the size distortion when category exchangeability '
    'fails and what studentizing does to it, the price label randomization '
    'charges relative to a normal benchmark, the behaviour of the calibrations '
    'in four harder designs with clustered, unequal, serially correlated and '
    'heavy tailed errors, and a benchmark against the bias reduced variance of '
    '\\citet{bell2002bias}. None of them changes a conclusion drawn here.\n'
)
anchor = '\\begin{table}[htbp]'
k = t.find(anchor, t.find('\\section{Simulations}'))
assert k > 0, 'could not find the first simulation table'
t = t[:k] + NOTE + '\n' + t[k:]

io.open(MAIN, 'w', encoding='utf-8', newline='').write(t)

s = io.open(SUPP, encoding='utf-8').read()
add = ['\\section{Additional simulation evidence}', '']
add += [b for b in moved]
block = "\n\n".join(add)
anchor2 = '\\begin{thebibliography}'
if anchor2 in s:
    s = s.replace(anchor2, block + "\n\n" + anchor2, 1)
else:
    s = s.replace('\\end{document}', block + "\n\n\\end{document}", 1)
io.open(SUPP, 'w', encoding='utf-8', newline='').write(s)

print()
print('moved %d blocks, %d words out of the main text' % (len(moved), taken))
