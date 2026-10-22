#!/usr/bin/env python3
"""paper_tex.py TAG [BUILD]: the paper's two-page MFEM table (siunitx S columns) from results/paper/TAG.json.

Writes results/paper/TAG_cases_1.tex (assembly_levels..nlvc), TAG_cases_2.tex (tmop..bpl), TAG_summary.tex
(per-suite totals and geometric-mean speedup) and TAG_table.tex (the table* wrapper, [t] placement: acmart's
\\floatpagefraction is 0.9, so a [p] float of this size is deferred to the end of the document).
Speedup = native time / XLA time. BUILD is the XLA column (default: the first build of the json).
"""
import sys, json, math
E = '/home/wmoses/ejwork'
PAGES = [['assembly_levels', 'ceed', 'dg_amr', 'elasticity', 'nlvc'], ['tmop', 'vector', 'virtuals', 'bpl']]
HEAD = (r'\begin{tabular}{ll S[table-format=3.4e-1] S[table-format=1.2e-1] S[table-format=3.4e-1] S[table-format=1.2e-1] S[table-format=2.2]}' '\n'
        r'\toprule' '\n'
        r'Suite & case & {native CUDA (ms)} & {$\sigma$} & {XLA-GPU (ms)} & {$\sigma$} & {speedup} \\' '\n'
        r'\midrule' '\n')

def sig(x, n):
    """x to n significant digits: plain when it needs no exponent, else m.me-k."""
    if x == 0:
        return '0'
    e = math.floor(math.log10(abs(x)))
    if -2 <= e < 4:
        d = max(n - 1 - e, 0)
        return f'{x:.{d}f}'
    m = f'{x / 10 ** e:.{n - 1}f}'
    if m.startswith('10'):  # the mantissa rounded up to 10
        e += 1
        m = f'{x / 10 ** e:.{n - 1}f}'
    return f'{m}e{e}'

def tex(s):
    return r'\texttt{' + s.replace('_', r'\_') + '}'

def main():
    tag = sys.argv[1]
    j = json.load(open(f'{E}/results/paper/{tag}.json'))
    build = sys.argv[2] if len(sys.argv) > 2 else j['builds'][0]
    for i, page in enumerate(PAGES, 1):
        out = HEAD
        for suite in page:
            if suite not in j['suites']:
                continue
            for r in j['suites'][suite]['cases']:
                n, x = r['native'], r[build]
                cells = [sig(n['mean'], 3), sig(n['std'], 2)] if n else ['{---}'] * 2
                cells += [sig(x['mean'], 3), sig(x['std'], 2)] if x else ['{---}'] * 2
                cells.append(f"{n['mean'] / x['mean']:.2f}" if n and x else '{---}')
                out += f"{tex(suite)} & {tex(r['case'])} & " + ' & '.join(cells) + r' \\' + '\n'
        out += '\\bottomrule\n\\end{tabular}\n'
        open(f'{E}/results/paper/{tag}_cases_{i}.tex', 'w').write(out)
    # per-suite summary: cases, native and XLA totals over the cases both have, geomean speedup
    out = (r'\begin{tabular}{l S[table-format=3] S[table-format=4.3] S[table-format=4.3] S[table-format=1.2]}' '\n'
           r'\toprule' '\n' r'Suite & {cases} & {native CUDA (ms)} & {XLA-GPU (ms)} & {speedup} \\' '\n' r'\midrule' '\n')
    allr, nt, xt = [], 0.0, 0.0
    for suite in sum(PAGES, []):
        if suite not in j['suites']:
            continue
        rows = j['suites'][suite]['cases']
        both = [r for r in rows if r['native'] and r[build]]
        nat = sum(r['native']['mean'] for r in rows if r['native'])
        ratios = [r['native']['mean'] / r[build]['mean'] for r in both]
        allr += ratios
        if both:
            xla = sum(r[build]['mean'] for r in both)
            nt += sum(r['native']['mean'] for r in both)
            xt += xla
            g = math.exp(sum(map(math.log, ratios)) / len(ratios))
            out += f"{tex(suite)} & {len(rows)} & {sig(nat, 3)} & {sig(xla, 3)} & {g:.2f} \\\\\n"
        else:
            out += f"{tex(suite)} & {len(rows)} & {sig(nat, 3)} & {{---}} & {{---}} \\\\\n"
    g = math.exp(sum(map(math.log, allr)) / len(allr))
    out += f"\\midrule\nall & {len(allr)} & {sig(nt, 3)} & {sig(xt, 3)} & {g:.2f} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    open(f'{E}/results/paper/{tag}_summary.tex', 'w').write(out)
    size = r'\centering\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{0.95}'
    open(f'{E}/results/paper/{tag}_table.tex', 'w').write(
        '% needs \\usepackage{booktabs,siunitx} (siunitx aligns the S columns on the decimal marker; headers in braces)\n'
        '\\begin{table*}[t]\n' + size + '\n'
        '\\caption{Performance of MFEM across all benchmark suites when compiled with vanilla Clang for GPU and with Reactant for XLA-GPU (NVIDIA RTX 5090). '
        'Times are per benchmark iteration, mean of three repetitions with their standard deviation $\\sigma$; speedup is native time over XLA-GPU time.}\n'
        '\\label{tab:mfem-performance}\n' + f'\\input{{{tag}_cases_1}}\n' + '\\end{table*}\n'
        '\\begin{table*}[t]\n\\ContinuedFloat\n' + size + '\n\\caption{(continued)}\n' + f'\\input{{{tag}_cases_2}}\n' + '\\end{table*}\n')
    print(out)

if __name__ == '__main__':
    main()
