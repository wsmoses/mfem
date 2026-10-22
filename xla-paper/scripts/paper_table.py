#!/usr/bin/env python3
"""paper_table.py TAG [XLA_BUILD...]: the paper's MFEM table from results/mfem3/TAG/{native,<xla builds>}/bench_*.json.

Per case: every repetition's real_time (ms) for each column; mean, std, min; the XLA/native ratio of means.
Per suite: geometric mean of the per-case ratios, the number of cases, the number of runs per case.
Writes results/paper/TAG.json (everything) and results/paper/TAG.tex (the summary table), prints the summary.
Columns: native, then each XLA build dir given (default xla2), named by its directory or TAG_NAMES.
"""
import sys, os, json, glob, math, statistics
E = '/home/wmoses/ejwork'
UNIT = {'ns': 1e-6, 'us': 1e-3, 'ms': 1, 's': 1e3}
SUITES = {  # binary -> paper name
    'bench_assembly_levels_x0': 'assembly_levels', 'bench_assembly_levels': 'assembly_levels',
    'bench_ceed': 'ceed', 'bench_dg_amr': 'dg_amr', 'bench_elasticity': 'elasticity', 'bench_nlvc': 'nlvc',
    'bench_tmop': 'tmop', 'bench_vector': 'vector', 'bench_virtuals': 'virtuals', 'bench_bpl': 'bpl'}
ORDER = ['assembly_levels', 'ceed', 'dg_amr', 'elasticity', 'nlvc', 'tmop', 'vector', 'virtuals', 'bpl']

def load(d):
    """{(suite, case): [ms per repetition]} and the run status lines."""
    r = {}
    for f in sorted(glob.glob(os.path.join(d, 'bench_*.json'))):
        b = os.path.basename(f)[:-5]
        suite = SUITES.get(b)
        if not suite:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for x in data.get('benchmarks', []):
            if x.get('run_type') == 'aggregate':
                continue
            r.setdefault((suite, x['run_name']), []).append(x['real_time'] * UNIT[x.get('time_unit', 'ns')])
    status = {}
    sf = os.path.join(d, 'summary.txt')
    if os.path.exists(sf):
        for l in open(sf):
            p = l.strip().split('|')
            if len(p) == 3:
                status[SUITES.get(p[0], p[0])] = (p[1], int(p[2]))
    return r, status

def stats(xs):
    return {'n': len(xs), 'mean': statistics.fmean(xs), 'std': statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            'min': min(xs), 'max': max(xs), 'runs': xs}

def main():
    tag = sys.argv[1]
    builds = sys.argv[2:] or ['xla2']
    nat, natstat = load(f'{E}/results/mfem3/{tag}/native')
    cols = {b: load(f'{E}/results/mfem3/{tag}/{b}') for b in builds}
    out = {'tag': tag, 'builds': builds, 'suites': {}}
    for suite in ORDER:
        cases = sorted({c for (s, c) in nat if s == suite} | {c for b in builds for (s, c) in cols[b][0] if s == suite})
        if not cases:
            continue
        rows = []
        for c in cases:
            row = {'case': c, 'native': stats(nat[(suite, c)]) if (suite, c) in nat else None}
            for b in builds:
                row[b] = stats(cols[b][0][(suite, c)]) if (suite, c) in cols[b][0] else None
                if row['native'] and row[b]:
                    row[b]['ratio'] = row[b]['mean'] / row['native']['mean']
            rows.append(row)
        summary = {}
        for b in builds:
            ratios = [r[b]['ratio'] for r in rows if r['native'] and r[b]]
            summary[b] = {'cases': len(ratios), 'geomean_ratio': math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None,
                          'native_cases': sum(1 for r in rows if r['native']), 'xla_cases': sum(1 for r in rows if r[b]),
                          'runs_per_case': min((r[b]['n'] for r in rows if r[b]), default=0),
                          'status': cols[b][1].get(suite), 'native_status': natstat.get(suite),
                          'native_total_ms': sum(r['native']['mean'] for r in rows if r['native'] and r[b]),
                          'xla_total_ms': sum(r[b]['mean'] for r in rows if r['native'] and r[b])}
        out['suites'][suite] = {'cases': rows, 'summary': summary}
    os.makedirs(f'{E}/results/paper', exist_ok=True)
    json.dump(out, open(f'{E}/results/paper/{tag}.json', 'w'), indent=1)
    # the LaTeX summary: suite, cases, native total, per build total and geomean ratio
    lines = [r'\begin{tabular}{l r r ' + ' '.join('r r' for _ in builds) + '}', r'\toprule',
             'Suite & cases & native CUDA (ms) ' + ''.join(f'& {b} (ms) & {b}/native (geo.\\ mean)' for b in builds) + r' \\', r'\midrule']
    print(f"{'suite':16s} {'cases':>5s} {'native ms':>10s}" + ''.join(f" {b+' ms':>12s} {b+'/nat':>9s}" for b in builds))
    for suite in ORDER:
        if suite not in out['suites']:
            continue
        s = out['suites'][suite]['summary']
        first = s[builds[0]]
        line = f"{suite} & {first['native_cases']} & {first['native_total_ms']:.1f} "
        txt = f"{suite:16s} {first['native_cases']:5d} {first['native_total_ms']:10.1f}"
        for b in builds:
            g = s[b]['geomean_ratio']
            line += f"& {s[b]['xla_total_ms']:.1f} & {g:.2f} " if g else '& -- & -- '
            txt += f" {s[b]['xla_total_ms']:12.1f} {g:9.2f}" if g else f" {'--':>12s} {'--':>9s}"
        lines.append(line + r'\\')
        print(txt)
    lines += [r'\bottomrule', r'\end{tabular}']
    open(f'{E}/results/paper/{tag}.tex', 'w').write('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()
