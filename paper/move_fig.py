"""Move Figure 1 to the supplement.

The figure plots the exact size implied by the F law against the number of
restrictions at five cluster counts.  It illustrates Corollary 2 and the design
rule; both are stated in the text with the numbers a reader needs, so the figure
is evidence rather than argument.

Run from paper2/.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main.tex')
SUPP = os.path.join(HERE, 'supplement.tex')

t = io.open(MAIN, encoding='utf-8').read()

i = t.index('\\label{fig:sizem}')
a = t.rfind('\\begin{figure}', 0, i)
b = t.index('\\end{figure}', i) + len('\\end{figure}')
fig = t[a:b]
t = t[:a] + t[b:]
print('   moved %d words of figure source' % len(fig.split()))

# pointed
refs = re.findall(r'[^.]*\\ref\{fig:sizem\}[^.]*\.', t)
for r in refs:
    print('   reference in main text: %s' % " ".join(r.split())[:100])

REPL = ('The supplement plots the exact size implied by \\eqref{eq:Fexact} '
        'against the number of restrictions at five cluster counts, with the '
        'measured rates of Table \\ref{tab:size} overlaid.')
for r in refs:
    t = t.replace(r, ' ' + REPL, 1)

t = re.sub(r'\n{4,}', '\n\n', t)
io.open(MAIN, 'w', encoding='utf-8', newline='').write(t)

s = io.open(SUPP, encoding='utf-8').read()
# supplement
if '\\usepackage{pgfplots}' not in s:
    s = s.replace('\\usepackage{xr}',
                  '\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n\\usepackage{xr}')
add = '\\section{The exact size against the number of restrictions}\n\n' + fig.strip()
s = s.replace('\\begin{thebibliography}', add + "\n\n\\begin{thebibliography}", 1)
io.open(SUPP, 'w', encoding='utf-8', newline='').write(s)
print('   supplement updated, pgfplots restored to its preamble')
