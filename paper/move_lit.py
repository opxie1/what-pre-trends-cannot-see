"""Third round: the related work section and the applied literature survey.

Both are discussion a reader may want and neither carries the argument.  The
main text keeps the claim each was supporting, in a few sentences, and the
survey itself goes to the supplement.  Each bibliography is then pruned to what
its own document cites, which is also where a good part of the remaining page
budget comes from.

Run after move_more.py, from paper2/.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main.tex')
SUPP = os.path.join(HERE, 'supplement.tex')

t = io.open(MAIN, encoding='utf-8').read()
moved = []

# ---- 1. the whole Related work section
a = t.index('\\section{Related work}')
b = t.index('\\section{What the falsification test is for}')
rel = t[a:b]
t = t[:a] + t[b:]
moved.append(rel)
print('   moved %5d words  Related work section' % len(rel.split()))

# ---- 2. the applied literature survey inside Section 3
i = t.index('The gap is a live one.')
j = t.index('That observation generalizes into the one constructive diagnostic')
survey = t[i:j]
t = t[:i] + t[j:]
moved.append(survey)
print('   moved %5d words  applied literature survey' % len(survey.split()))

SUMMARY = (
    'The gap is a live one, and the supplement documents it: work that warns '
    'about differential recording and then offers an event study as the remedy, '
    'work that reports a pre trend test as evidence of parallel trends while '
    'conceding that the outcome may reflect disclosure rather than incidence, '
    'and work that attributes an undetected violation to low power. Corollary '
    '\\ref{cor:blind} says that where the contamination is a recording response '
    'none of these is a matter of power. Where the threat is taken seriously '
    'the response is already the method proposed here: \\citet{harasztosi2019minwage} '
    'settle a reporting story not from the pre period but across outcomes, '
    'which is a cross outcome falsification test conducted informally, and '
    '\\citet{geruso2020upcoding} obtain a placebo with power only because an '
    'institutional lag supplies a window in which the coding response cannot '
    'yet have operated. The rest of this paper supplies the reference '
    'distribution such a test needs when the number of clusters is small.\n\n'
)
t = t[:i] + SUMMARY + t[i:]

t = re.sub(r'\n{4,}', '\n\n', t)
io.open(MAIN, 'w', encoding='utf-8', newline='').write(t)

s = io.open(SUPP, encoding='utf-8').read()
block = "\n\n".join(
    ['\\section{Related work and the applied literature}', '']
    + [b.strip().replace('\\section{Related work}', '\\subsection*{Related work}')
       for b in moved])
s = s.replace('\\begin{thebibliography}', block + "\n\n\\begin{thebibliography}", 1)
io.open(SUPP, 'w', encoding='utf-8', newline='').write(s)

print()
print('moved %d blocks, %d words' % (len(moved), sum(len(b.split()) for b in moved)))
