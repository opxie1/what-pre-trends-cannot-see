"""Rebuild the supplement's bibliography from whatever the supplement cites.

The supplement's reference list was generated when it held only the proofs.
Moving the simulation and application blocks brought further citations with
them, so the list has to be regenerated from the current text.  Entries are
copied verbatim from main.tex so the two cannot disagree.

Run from paper2/ whenever blocks move.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
m = io.open(os.path.join(HERE, 'main.tex'), encoding='utf-8').read()
s = io.open(os.path.join(HERE, 'supplement.tex'), encoding='utf-8').read()

# what the supplement cites, outside its own reference list
body = s.split('\\begin{thebibliography}')[0]
keys = {k.strip() for g in re.findall(r'\\cite[a-z]*\{([^}]+)\}', body)
        for k in g.split(',')}

bib_all = m[m.index('\\begin{thebibliography}'):
            m.index('\\end{thebibliography}') + len('\\end{thebibliography}')]
items = re.findall(r'\\bibitem\[.*?(?=\\bibitem\[|\\end\{thebibliography\})',
                   bib_all, flags=re.S)
by_key = {re.search(r'\]\{([^}]+)\}', i).group(1): i.rstrip() for i in items}

missing = sorted(keys - set(by_key))
assert not missing, 'supplement cites entries absent from main: %s' % missing

kept = [by_key[k] for k in sorted(keys)]
new_bib = ("\\begin{thebibliography}{99}\n\\itemsep 3pt\n\n"
           + "\n\n".join(kept) + "\n\n\\end{thebibliography}")

a = s.find('\\begin{thebibliography}')
if a >= 0:
    b = s.index('\\end{thebibliography}') + len('\\end{thebibliography}')
    s = s[:a] + new_bib + s[b:]
else:
    s = s.replace('\\end{document}', new_bib + '\n\n\\end{document}')

io.open(os.path.join(HERE, 'supplement.tex'), 'w',
        encoding='utf-8', newline='').write(s)
print('supplement bibliography rebuilt: %d entries for %d cited keys'
      % (len(kept), len(keys)))
for k in sorted(keys):
    print('   %s' % k)
