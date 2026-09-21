"""Split the manuscript into a main paper and an online supplement.

JBES counts abstract, main text, tables, figures and references against a cap of
35 pages.  The proofs and the secondary simulation and application evidence go
to a supplement, which is not counted.  Cross references from the supplement
into the main paper are resolved by the xr package, so every "Proof of Theorem
1" names the number the main paper actually prints.

Nothing is retyped: every block is lifted verbatim out of main.tex, so the
supplement cannot disagree with the paper it supports.

Run from paper2/:  python build_supplement.py
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main.tex')
SUPP = os.path.join(HERE, 'supplement.tex')

t = io.open(MAIN, encoding='utf-8').read()

# preamble
pre = t[:t.index('\\begin{document}')]
pre = pre.replace('\\usepackage{pgfplots}', '')
pre = pre.replace('\\pgfplotsset{compat=1.18}', '')
pre = re.sub(r'\\title\{.*?\n\}', '', pre, flags=re.S)
pre = re.sub(r'\\author\{.*?\n\}', '', pre, flags=re.S)
pre = re.sub(r'\\date\{[^}]*\}', '', pre)
assert '\\newtheorem' in pre, 'theorem environments lost from the preamble'
pre += '\n\\usepackage{xr}\n\\externaldocument{main}\n'

# proofs
a = t.index('\\section{Proofs}')
b = t.index('\\section*{Declarations}')
proofs = t[a:b].rstrip()
assert proofs.count('\\begin{proof}') == 12, (
    'expected 12 proofs, found %d' % proofs.count('\\begin{proof}'))

# bibliography
bibstart = t.index('\\begin{thebibliography}')
bib_all = t[bibstart:t.index('\\end{thebibliography}') + len('\\end{thebibliography}')]

body = proofs
keys = {k.strip() for g in re.findall(r'\\cite[a-z]*\{([^}]+)\}', body)
        for k in g.split(',')}
items = re.findall(r'\\bibitem\[.*?(?=\\bibitem\[|\\end\{thebibliography\})',
                   bib_all, flags=re.S)
kept = [i for i in items if re.search(r'\]\{([^}]+)\}', i).group(1) in keys]
missing = keys - {re.search(r'\]\{([^}]+)\}', i).group(1) for i in kept}
assert not missing, 'supplement cites entries not in the bibliography: %s' % missing

out = [
    pre,
    '\\begin{document}',
    '\\begin{center}',
    '{\\large\\bfseries Supplement to \\emph{What Pre-Trends Cannot See:}}\\\\[2pt]',
    '{\\large\\bfseries \\emph{Testing for Treatment-Induced Recording Change '
    'with Few Clusters}}\\\\[8pt]',
    'Ethan Xie',
    '\\end{center}',
    '\\vspace{6pt}',
    '\\noindent This supplement contains the proofs of every result in the '
    'paper. Numbering follows the paper: a reference to Theorem 1 here is '
    'Theorem 1 there.',
    '\\vspace{10pt}',
    '\\setcounter{section}{0}',
    '\\renewcommand{\\thesection}{S\\arabic{section}}',
    '\\numberwithin{equation}{section}',
    proofs.replace('\\section{Proofs}\n\\label{app:proofs}',
                   '\\section{Proofs}'),
]
if kept:
    out += ['\\begin{thebibliography}{99}', '\\itemsep 3pt']
    out += [i.rstrip() for i in kept]
    out += ['\\end{thebibliography}']
out += ['\\end{document}', '']

io.open(SUPP, 'w', encoding='utf-8', newline='').write("\n\n".join(out))
print('supplement written: %d proofs, %d bibliography entries'
      % (proofs.count('\\begin{proof}'), len(kept)))

# main
POINTER = (
    '\\appendix\n\n'
    '\\section*{Proofs}\n'
    '\\label{app:proofs}\n\n'
    '\\noindent Every proof is in the online supplement, which restates nothing '
    'and follows the numbering used here.\n'
)
a2 = t.index('\\appendix')
new_main = t[:a2] + POINTER + '\n' + t[b:]
io.open(MAIN, 'w', encoding='utf-8', newline='').write(new_main)
print('main.tex: proofs replaced by a pointer (%d words removed)'
      % (len(proofs.split())))
