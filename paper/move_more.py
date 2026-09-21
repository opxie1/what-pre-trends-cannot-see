"""Second round of moves: the one factor diagnostic and two sensitivity blocks.

Both are evidence a reader may want and neither carries the argument.  The main
text keeps a statement of what each found; the detail goes to the supplement.

Run after move_app.py, from paper2/.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main.tex')
SUPP = os.path.join(HERE, 'supplement.tex')

t = io.open(MAIN, encoding='utf-8').read()
moved = []


def cut(start, end, label):
    global t
    i = t.find(start)
    assert i >= 0, 'start not found: %s' % label
    j = t.find(end, i)
    assert j >= 0, 'end not found: %s' % label
    j += len(end)
    block = t[i:j]
    assert '\\begin{table}' not in block and '\\begin{figure}' not in block, label
    t = t[:i] + t[j:]
    moved.append(block)
    print('   moved %5d words  %s' % (len(block.split()), label))


# diagnostic
cut('The three designs of Section \\ref{sec:app} differ sharply on both.',
    'is a reason to report the cluster level procedures\nof Section \\ref{sec:cluster}, since they use neither the normalization nor\nAssumption \\ref{as:CE}.',
    'one factor diagnostic detail')

ONEFACTOR = (
    'In the three designs of Section \\ref{sec:app} the common factor carries '
    '$0.397$ of the median category\'s demeaned variance in the crime panel and '
    '$0.121$ and $0.065$ in the two health sub designs, and single triads '
    'disagree about $\\rho_c$ by factors of $1.23$, $2.39$ and $8.59$. The '
    'premise is a serviceable approximation in the crime design and is not one '
    'in the health designs. The supplement gives the two statistics, a note on '
    'why no $p$ value is attached to either, and the precise statement of what '
    'a failure costs: Lemma \\ref{lem:equi} is algebraic and survives, one '
    'factor is not among the hypotheses of Theorem \\ref{thm:exact}, and what '
    'goes is the reason to expect Assumption \\ref{as:CE} to hold after '
    'normalizing.\n'
)
k = t.find('\\begin{assumption}[category exchangeability]')
assert k > 0, 'anchor for the one factor summary not found'
t = t[:k] + ONEFACTOR + '\n' + t[k:]

# smoothness
cut('A second reading deserves an answer.',
    'neither \citet{low1997nonparametric} nor \citet{cailow2004adaptation}.',
    'smoothness class and second reading')

t = re.sub(r'\n{4,}', '\n\n', t)
io.open(MAIN, 'w', encoding='utf-8', newline='').write(t)

s = io.open(SUPP, encoding='utf-8').read()
block = "\n\n".join(['\\section{Additional identification and normalisation evidence}', '']
                    + [b.strip() for b in moved])
anchor = '\\begin{thebibliography}'
s = s.replace(anchor, block + "\n\n" + anchor, 1)
io.open(SUPP, 'w', encoding='utf-8', newline='').write(s)
print()
print('moved %d blocks, %d words' % (len(moved), sum(len(b.split()) for b in moved)))
