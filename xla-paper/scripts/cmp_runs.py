#!/usr/bin/env python3
"""cmp_runs.py DIR_A DIR_B [THRESH]: per-case mean real_time (ms) of two results/mfem3/<tag>/<build> dirs, with the
B/A ratio; prints the cases whose ratio is outside [1/THRESH, THRESH] (default 1.07) and the geomean over all."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_table import load
a, b = load(sys.argv[1])[0], load(sys.argv[2])[0]
th = float(sys.argv[3]) if len(sys.argv) > 3 else 1.07
rows = []
for k in a:
    if k in b and a[k] and b[k]:
        ma, mb = sum(a[k]) / len(a[k]), sum(b[k]) / len(b[k])
        rows.append((k, ma, mb, mb / ma))
rows.sort(key=lambda r: r[3])
for k, ma, mb, r in rows:
    if r > th or r < 1 / th:
        print(f'{k[0]:16s} {k[1]:34s} {ma:9.4g} {mb:9.4g} {r:6.3f}')
if rows:
    print(f'{len(rows)} cases, geomean B/A {math.exp(sum(math.log(r[3]) for r in rows) / len(rows)):.4f}, '
          f'faster {sum(r[3] < 1 / th for r in rows)}, slower {sum(r[3] > th for r in rows)}')
