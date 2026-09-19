"""Prune the main bibliography to what the main paper cites.

Moving the related work section and the applied survey to the supplement left
entries in the main reference list that the main text no longer cites.  The
supplement carries its own list, rebuilt by fix_supp_bib.py, so nothing is lost
to a reader of either document.

Run from paper2/ after any block move.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'main.tex')

t = io.open(MAIN, encoding='utf-8').read()
a = t.index('\\begin{thebibliography}')
b = t.index('\\end{thebibliography}') + len('\\end{thebibliography}')
bib = t[a:b]

body = t[:a]
keys = {k.strip() for g in re.findall(r'\\cite[a-z]*\{([^}]+)\}', body)
        for k in g.split(',')}

items = re.findall(r'\\bibitem\[.*?(?=\\bibitem\[|\\end\{thebibliography\})',
                   bib, flags=re.S)
by_key = {re.search(r'\]\{([^}]+)\}', i).group(1): i.rstrip() for i in items}

missing = sorted(keys - set(by_key))
assert not missing, 'main cites entries with no bibitem: %s' % missing

dropped = sorted(set(by_key) - keys)
kept_keys = sorted(keys)
new_bib = ("\\begin{thebibliography}{99}\n\\itemsep 3pt\n\n"
           + "\n\n".join(by_key[k] for k in kept_keys) + "\n\n\\end{thebibliography}")

io.open(MAIN, 'w', encoding='utf-8', newline='').write(t[:a] + new_bib + t[b:])
print('main bibliography: %d entries kept, %d dropped'
      % (len(kept_keys), len(dropped)))
print('dropped (now cited only in the supplement):')
for k in dropped:
    print('   %s' % k)
