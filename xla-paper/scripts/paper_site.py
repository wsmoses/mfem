#!/usr/bin/env python3
"""paper_site.py TAG OUT.html [--meta META.json]: the paper's MFEM native-vs-XLA data page from results/paper/TAG.json
(written by paper_table.py). Every repetition of every case is on the page; the summary is the per-suite geometric
mean of per-case ratios of means. META.json: {"builds": {col: {"label", "note"}}, "hardware", "date", "notes": [...]}.
"""
import sys, os, json, math, html, statistics
E = '/home/wmoses/ejwork'
NAMES = {'assembly_levels': 'assembly_levels', 'ceed': 'ceed', 'dg_amr': 'dg_amr', 'elasticity': 'elasticity',
         'nlvc': 'nlvc', 'tmop': 'tmop', 'vector': 'vector', 'virtuals': 'virtuals', 'bpl': 'bpl'}
BLURB = {
    'assembly_levels': 'setup and action of the partial (PA), element (EA) and full (FA) assembly levels; 1,048,576 dofs, orders 2 and 4',
    'ceed': 'the CEED bake-off problems BP1–BP6 on a 96³ mesh, orders 2 and 4',
    'dg_amr': 'action of partially assembled DG convection on a nonconforming mesh, 1,048,576 elements',
    'elasticity': 'action of the Jacobian of a 3D Neo-Hookean hyperelastic model, 60³ elements',
    'nlvc': 'the nonlinear vector convection operator of Navier–Stokes, 2D and 3D',
    'tmop': 'partially assembled TMOP mesh optimization kernels',
    'vector': 'core vector and tensor operations, n = 1024',
    'virtuals': 'virtual-call overhead microbenchmark',
    'bpl': 'the BPL bake-off problems (libCEED-style matrix-free Laplacians)',
}

def fmt(v, nd=3):
    if v is None: return '—'
    if v == 0: return '0'
    if v < 0.01: return f'{v:.2e}'
    if v < 1: return f'{v:.{nd}f}'
    if v < 100: return f'{v:.2f}'
    return f'{v:.1f}'

def main():
    tag = sys.argv[1]; out = sys.argv[2]
    meta = {}
    if '--meta' in sys.argv:
        meta = json.load(open(sys.argv[sys.argv.index('--meta') + 1]))
    data = json.load(open(f'{E}/results/paper/{tag}.json'))
    builds = data['builds']
    labels = {b: meta.get('builds', {}).get(b, {}).get('label', b) for b in builds}
    notes = {b: meta.get('builds', {}).get(b, {}).get('note', '') for b in builds}
    suites = [s for s in data['suites']]
    # overall: geometric mean over every case with both numbers, per build
    overall = {}
    for b in builds:
        rs = [r[b]['ratio'] for s in suites for r in data['suites'][s]['cases'] if r['native'] and r[b]]
        overall[b] = (math.exp(sum(map(math.log, rs)) / len(rs)) if rs else None, len(rs),
                      sum(1 for r in rs if r <= 1.0))
    ncases = sum(1 for s in suites for r in data['suites'][s]['cases'] if r['native'])
    reps = min(r['native']['n'] for s in suites for r in data['suites'][s]['cases'] if r['native'])
    css = '''
<title>MFEM on XLA</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column for the method and summary; the data tables scroll inside their own frame */
:root { --bg:#f7f8f6; --fg:#1b2024; --mute:#5c6670; --line:#d5dad6; --head:#e9ede8; --accent:#1f6f78; --accent-ink:#ffffff;
  --faster:#2f7d4f; --faster-bg:#e4f2e8; --slower:#a4432b; --slower-bg:#f8e7e1; --mid:#8a6d1f; --mid-bg:#f5eedb; --code:#eef1ee;
  --display:"Source Serif 4", Georgia, "Times New Roman", serif; --body:"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono:"IBM Plex Mono", "SFMono-Regular", Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#15191c; --fg:#e4e8e6; --mute:#9aa5ad; --line:#2f373c; --head:#1e2529;
  --accent:#5fb8c0; --accent-ink:#0f1416; --faster:#7fd29c; --faster-bg:#1b3325; --slower:#f0957a; --slower-bg:#3b2219; --mid:#d9b454; --mid-bg:#3a3018; --code:#1e2428; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#15191c; --fg:#e4e8e6; --mute:#9aa5ad; --line:#2f373c; --head:#1e2529;
  --accent:#5fb8c0; --accent-ink:#0f1416; --faster:#7fd29c; --faster-bg:#1b3325; --slower:#f0957a; --slower-bg:#3b2219; --mid:#d9b454; --mid-bg:#3a3018; --code:#1e2428; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.5; padding-inline: 16px; padding-block: 28px 64px; }
.wrap { max-width: 1180px; margin-inline: auto; display: grid; gap: 36px; }
h1, h2, h3 { font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; letter-spacing: -0.01em; }
h1 { font-size: 2.2rem; line-height: 1.1; } h2 { font-size: 1.45rem; } h3 { font-size: 1.15rem; }
p { margin: 0; max-width: 68ch; } .lede { font-size: 1.1rem; color: var(--mute); }
.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent); }
header { display: grid; gap: 10px; }
section { display: grid; gap: 14px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.kpi { border-top: 2px solid var(--accent); padding-top: 8px; display: grid; gap: 2px; }
.kpi .v { font-family: var(--mono); font-size: 1.6rem; font-weight: 500; font-variant-numeric: tabular-nums; }
.kpi .l { color: var(--mute); font-size: 0.85rem; }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 4px; min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; font-variant-numeric: tabular-nums; }
th, td { padding: 5px 10px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; white-space: nowrap; }
thead th { background: var(--head); font-weight: 600; position: sticky; top: 0; }
th.num, td.num { text-align: right; font-family: var(--mono); }
td.case { font-family: var(--mono); }
tbody tr:last-child td { border-bottom: 0; }
.runs { color: var(--mute); font-size: 0.78rem; }
.ratio { display: inline-block; min-width: 4.2em; text-align: right; padding: 0 6px; border-radius: 3px; }
.ratio.faster { background: var(--faster-bg); color: var(--faster); }
.ratio.slower { background: var(--slower-bg); color: var(--slower); }
.ratio.mid { background: var(--mid-bg); color: var(--mid); }
tfoot td { background: var(--head); font-weight: 600; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 18px; max-width: 80ch; margin: 0; }
dt { color: var(--mute); } dd { margin: 0; min-width: 0; }
code, pre { font-family: var(--mono); font-size: 0.82rem; }
code { background: var(--code); padding: 1px 4px; border-radius: 3px; }
pre { background: var(--code); padding: 12px 14px; border-radius: 4px; overflow-x: auto; margin: 0; line-height: 1.4; }
.chart { width: 100%; max-width: 760px; height: auto; display: block; }
.chart text { font-family: var(--mono); font-size: 11px; fill: var(--fg); }
.chart .axis { stroke: var(--line); } .chart .one { stroke: var(--fg); stroke-dasharray: 3 3; }
.chart .bar-off { fill: var(--accent); } .chart .bar-on { fill: var(--mute); opacity: 0.7; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.85rem; color: var(--mute); }
.legend span::before { content: ""; display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; border-radius: 2px; background: var(--accent); }
.legend span.on::before { background: var(--mute); opacity: 0.7; }
button { font: inherit; font-size: 0.8rem; padding: 4px 10px; border: 1px solid var(--line); background: var(--head); color: var(--fg); border-radius: 3px; cursor: pointer; }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.toc { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.9rem; }
a { color: var(--accent); }
details summary { cursor: pointer; color: var(--accent); font-size: 0.9rem; }
@media (max-width: 640px) { h1 { font-size: 1.7rem; } body { font-size: 14px; } }
</style>'''
    def rcls(r):
        return 'faster' if r <= 1.0 else ('mid' if r <= 1.5 else 'slower')
    # summary table + chart
    srows = []
    for s in suites:
        sm = data['suites'][s]['summary']
        first = sm[builds[0]]
        cells = f'<td><a href="#{s}">{html.escape(NAMES[s])}</a></td><td class="num">{first["native_cases"]}</td><td class="num">{fmt(first["native_total_ms"])}</td>'
        for b in builds:
            g = sm[b]['geomean_ratio']
            cells += f'<td class="num">{fmt(sm[b]["xla_total_ms"]) if g else "—"}</td><td class="num">' + (f'<span class="ratio {rcls(g)}">{g:.2f}×</span>' if g else '—') + '</td>'
        srows.append(f'<tr>{cells}</tr>')
    foot = f'<td>all suites (geometric mean over cases)</td><td class="num">{ncases}</td><td></td>'
    for b in builds:
        g, n, nf = overall[b]
        foot += f'<td></td><td class="num">' + (f'<span class="ratio {rcls(g)}">{g:.2f}×</span>' if g else '—') + '</td>'
    shead = ''.join(f'<th class="num">{html.escape(labels[b])} total ms</th><th class="num">{html.escape(labels[b])} / native</th>' for b in builds)
    summary_table = f'<div class="tablewrap"><table><thead><tr><th>suite</th><th class="num">cases</th><th class="num">native total ms</th>{shead}</tr></thead><tbody>{"".join(srows)}</tbody><tfoot><tr>{foot}</tr></tfoot></table></div>'
    # chart: horizontal bars of geomean ratio per suite, log scale from 0.25 to 4
    def chart():
        W, H = 760, 26 * len(suites) + 50; L, R = 150, 30
        lo, hi = math.log(0.25), math.log(4)
        x = lambda v: L + (math.log(max(0.25, min(4, v))) - lo) / (hi - lo) * (W - L - R)
        g = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="geometric-mean XLA/native ratio per suite">']
        for t in (0.25, 0.5, 1, 2, 4):
            g.append(f'<line class="axis{" one" if t == 1 else ""}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="18" y2="{H - 28}" stroke-width="1"/>')
            g.append(f'<text x="{x(t):.1f}" y="{H - 12}" text-anchor="middle">{t}×</text>')
        g.append(f'<text x="{x(1):.1f}" y="12" text-anchor="middle">native = 1×; left is faster on XLA</text>')
        for i, s in enumerate(suites):
            y = 24 + i * 26
            g.append(f'<text x="{L - 8}" y="{y + 15}" text-anchor="end">{html.escape(NAMES[s])}</text>')
            nb = len(builds); bh = 18 / nb
            for k, b in enumerate(builds):
                r = data['suites'][s]['summary'][b]['geomean_ratio']
                if not r: continue
                x0, x1 = sorted((x(1), x(r)))
                g.append(f'<rect class="bar-{"off" if k == 0 else "on"}" x="{x0:.1f}" y="{y + 2 + k * bh:.1f}" width="{max(1, x1 - x0):.1f}" height="{bh - 1:.1f}"/>')
                g.append(f'<text x="{(x1 + 4) if r >= 1 else (x0 - 4):.1f}" y="{y + 2 + k * bh + bh * 0.75:.1f}" text-anchor="{"start" if r >= 1 else "end"}">{r:.2f}</text>')
        g.append('</svg>')
        return ''.join(g)
    legend = '<div class="legend">' + ''.join(f'<span class="{"on" if k else ""}">{html.escape(labels[b])}</span>' for k, b in enumerate(builds)) + '</div>'
    # per-suite tables with every run
    sections = []
    for s in suites:
        rows = []
        for r in data['suites'][s]['cases']:
            n = r['native']
            cells = f'<td class="case">{html.escape(r["case"])}</td>'
            cells += (f'<td class="num">{fmt(n["mean"])}</td><td class="num runs">± {fmt(n["std"])}<br>{" / ".join(fmt(v) for v in n["runs"])}</td>' if n else '<td class="num">—</td><td></td>')
            for b in builds:
                xb = r[b]
                if xb:
                    cells += f'<td class="num">{fmt(xb["mean"])}</td><td class="num runs">± {fmt(xb["std"])}<br>{" / ".join(fmt(v) for v in xb["runs"])}</td>'
                    cells += f'<td class="num">' + (f'<span class="ratio {rcls(xb["ratio"])}">{xb["ratio"]:.2f}×</span>' if 'ratio' in xb else '—') + '</td>'
                else:
                    cells += '<td class="num">—</td><td></td><td class="num">—</td>'
            rows.append(f'<tr>{cells}</tr>')
        sm = data['suites'][s]['summary']
        head = '<th>case</th><th class="num">native ms</th><th class="num">σ · runs</th>' + ''.join(f'<th class="num">{html.escape(labels[b])} ms</th><th class="num">σ · runs</th><th class="num">/ native</th>' for b in builds)
        geo = ' · '.join(f'{html.escape(labels[b])} geo-mean {sm[b]["geomean_ratio"]:.2f}× over {sm[b]["cases"]} cases' for b in builds if sm[b]['geomean_ratio'])
        status = ''
        for b in builds:
            st = sm[b]['status']
            if st and st[0] != '0':
                status += f' {html.escape(labels[b])}: run status {html.escape(str(st[0]))}.'
        sections.append(f'<section id="{s}"><h3>{html.escape(NAMES[s])}</h3><p class="lede">{html.escape(BLURB.get(s, ""))}</p><p>{geo}.{status}</p><div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>')
    # LaTeX
    tex = open(f'{E}/results/paper/{tag}.tex').read()
    hw = meta.get('hardware', '')
    method = ''.join(f'<dt>{html.escape(k)}</dt><dd>{v}</dd>' for k, v in meta.get('method', []))
    notes_html = ''.join(f'<li>{v}</li>' for v in meta.get('notes', []))
    kpis = f'<div class="kpi"><span class="v">{ncases}</span><span class="l">benchmark cases, {len(suites)} suites</span></div>'
    kpis += f'<div class="kpi"><span class="v">{reps}×</span><span class="l">repetitions per case and configuration</span></div>'
    for b in builds:
        g, n, nf = overall[b]
        if g:
            kpis += f'<div class="kpi"><span class="v">{g:.2f}×</span><span class="l">{html.escape(labels[b])} / native, geometric mean over {n} cases; XLA at or below native on {nf}</span></div>'
    body = f'''{css}
<div class="wrap">
<header>
<span class="eyebrow">MFEM · native CUDA vs Reactant XLA-GPU · {html.escape(meta.get("date", ""))}</span>
<h1>MFEM on XLA</h1>
<p class="lede">Every MFEM benchmark suite, compiled once with Clang for CUDA and once through Reactant to XLA, run on the same GPU on the same day. All repetitions are on this page; the summary is the geometric mean of per-case ratios of means.</p>
<nav class="toc">{" ".join(f'<a href="#{s}">{html.escape(NAMES[s])}</a>' for s in suites)} <a href="#method">method</a> <a href="#latex">LaTeX</a></nav>
</header>
<section><div class="kpis">{kpis}</div></section>
<section id="summary"><h2>Summary by suite</h2>{summary_table}{legend}{chart()}</section>
<section id="method"><h2>Method</h2><dl>{method}</dl><ul>{notes_html}</ul></section>
<section id="data"><h2>All measurements</h2><p>Times are Google Benchmark <code>real_time</code> in milliseconds per iteration; the three repetitions of each case are listed under the mean with their population standard deviation σ. Ratios are XLA mean over native mean; green ≤ 1×, amber ≤ 1.5×, red above.</p>
{"".join(sections)}</section>
<section id="latex"><h2>LaTeX summary table</h2><p>Columns: suite, cases, native total of per-case means (ms), then per configuration the total (ms) and the geometric mean of the per-case ratios.</p>
<div><button id="copytex" type="button">Copy LaTeX</button></div><pre id="tex">{html.escape(tex)}</pre></section>
</div>
<script>
document.getElementById('copytex').addEventListener('click', function () {{
  var t = document.getElementById('tex').textContent, b = this;
  function fb() {{ var r = document.createRange(); r.selectNodeContents(document.getElementById('tex')); var s = getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = 'Selected'; }}
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(function () {{ b.textContent = 'Copied'; }}, fb); else fb();
}});
</script>'''
    open(out, 'w').write(body)
    print(out, len(body))

if __name__ == '__main__':
    main()
