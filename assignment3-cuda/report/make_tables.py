#!/usr/bin/env python3
"""Turn ../results/*.csv into LaTeX tables (gen/*.tex) and macros (gen/macros.tex).
Only the Python standard library is used."""
import csv, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
GEN = os.path.join(HERE, 'gen')
os.makedirs(GEN, exist_ok=True)


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def p2s(v):
    """power-of-two pretty print: 1048576 -> 2^{20}"""
    v = int(v)
    if v > 0 and v & (v - 1) == 0 and v >= 1024:
        return '$2^{%d}$' % (v.bit_length() - 1)
    return str(v)


def ms(v):
    v = float(v)
    if v >= 1000:
        return '%.0f' % v
    if v >= 100:
        return '%.1f' % v
    if v >= 10:
        return '%.2f' % v
    return '%.3f' % v


def write(name, text):
    with open(os.path.join(GEN, name), 'w') as f:
        f.write(text)


def table(cols, header, rows, caption, label):
    s = ['\\begin{table}[H]\\centering\\small',
         '\\caption{%s}\\label{%s}' % (caption, label),
         '\\begin{tabular}{%s}' % cols, '\\toprule',
         ' & '.join(header) + ' \\\\', '\\midrule']
    for r in rows:
        s.append('\\midrule' if r == 'MID' else ' & '.join(r) + ' \\\\')
    s += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(s)


macros = []


def macro(name, val):
    macros.append('\\newcommand{\\%s}{%s}' % (name, val))


def best_of(rows, key='avg_ms'):
    return min(rows, key=lambda r: float(r[key])) if rows else None


def matrix(rows, rkey, ckey, caption, label, rname, cname, fmt=ms):
    """rows -> 2-D table of min avg_ms over everything else; best cell in bold."""
    cell = defaultdict(lambda: None)
    rvals, cvals = set(), set()
    for r in rows:
        a, b = r[rkey], r[ckey]
        rvals.add(a); cvals.add(b)
        if cell[(a, b)] is None or float(r['avg_ms']) < float(cell[(a, b)]['avg_ms']):
            cell[(a, b)] = r
    rv = sorted(rvals, key=lambda v: (float(v) if v.replace('.', '').isdigit() else 0, v))
    cv = sorted(cvals, key=float)
    allbest = best_of([c for c in cell.values() if c])
    out = []
    for a in rv:
        line = [p2s(a) if a.isdigit() else a]
        for b in cv:
            c = cell[(a, b)]
            if c is None:
                line.append('--')
            else:
                t = fmt(c['avg_ms'])
                line.append('\\textbf{%s}' % t if c is allbest else t)
        out.append(line)
    return table('l' + 'r' * len(cv), ['%s $\\backslash$ %s' % (rname, cname)] + [p2s(b) for b in cv],
                 out, caption, label)


# ------------------------------------------------------------------ device
dev = {}
p = os.path.join(RES, 'devinfo.txt')
if os.path.exists(p):
    for line in open(p):
        if ',' in line:
            k, v = line.strip().split(',', 1)
            dev[k] = v
for k, v in dev.items():
    macro('dev' + ''.join(c for c in k if c.isalpha()), v)

# ------------------------------------------------------------------ Q1
p1t = load('q1_p1_tune.csv')
dyn = load('q1_p1_dyn.csv')
if p1t:
    for s in ('static', 'cyclic'):
        rows = [r for r in p1t if r['sched'] == s]
        write('q1_p1_%s.tex' % s, matrix(rows, 'threads', 'blocksize',
              'P1, %s assignment: time (ms) for $2^{22}$ inserts vs.\\ total threads and block size.' % s,
              'tab:p1%s' % s, 'threads', 'block size'))
    rows = []
    for s, src in (('static', p1t), ('cyclic', p1t), ('dynamic', dyn)):
        b = best_of([r for r in src if r['sched'] == s])
        if b is None:
            continue
        rows.append([s + (' (chunk %s)' % b['param'] if s == 'dynamic' else ''), p2s(b['threads']), b['blocks'],
                     b['blocksize'], ms(b['avg_ms']), '%.2f' % (int(b['nops']) / float(b['avg_ms']) / 1e3)])
        macro('pone%sms' % s, ms(b['avg_ms']))
    write('q1_p1_sched.tex', table('lrrrrr', ['assignment', 'threads', 'blocks', 'block size', 'time (ms)', 'Mops/s'],
          rows, 'P1: best configuration of each task-assignment scheme ($2^{22}$ inserts).', 'tab:p1sched'))

ch = load('q1_p1_chunk.csv')
if ch:
    write('q1_p1_chunk.tex', matrix(ch, 'param', 'threads',
          'P1 dynamic assignment: time (ms) vs.\\ chunk size and total threads (block size 256, $2^{22}$ inserts).',
          'tab:p1chunk', 'chunk', 'threads'))
    b = best_of(ch)
    macro('ponechunk', b['param']); macro('ponechunkthreads', p2s(b['threads']))

if dyn:
    write('q1_p1_dyn.tex', matrix(dyn, 'threads', 'blocksize',
          'P1 dynamic assignment with the best chunk size: time (ms) vs.\\ threads and block size ($2^{22}$ inserts).',
          'tab:p1dyn', 'threads', 'block size'))
    b = best_of(dyn)
    macro('ponebs', b['blocksize']); macro('ponebestthreads', p2s(b['threads'])); macro('ponebestms', ms(b['avg_ms']))

grp = load('q1_p2_group.csv')
if grp:
    write('q1_p2_group.tex', matrix(grp, 'param', 'threads',
          'P2: time (ms) vs.\\ thread-group size and total threads (block size 256, $2^{22}$ inserts).',
          'tab:p2group', 'group size', 'threads'))
    b = best_of(grp)
    macro('ptwogroup', b['param'])
p2t = load('q1_p2_tune.csv')
if p2t:
    write('q1_p2_tune.tex', matrix(p2t, 'threads', 'blocksize',
          'P2 with the best group size: time (ms) vs.\\ threads and block size ($2^{22}$ inserts).',
          'tab:p2tune', 'threads', 'block size'))
    b = best_of(p2t)
    macro('ptwobs', b['blocksize']); macro('ptwobestthreads', p2s(b['threads'])); macro('ptwobestms', ms(b['avg_ms']))

fin = load('q1_final.csv')
dfin = load('q1_p2dyn_final.csv')
if fin:
    groups = defaultdict(list)
    for r in fin + [dict(r, prog='P2D') for r in dfin]:
        nops = int(r['nops'])
        key = (nops, round(100 * int(r['ins']) / nops), round(100 * int(r['del']) / nops), round(100 * int(r['srch']) / nops))
        groups[key].append(r)
    order = ['100/0/0', '50/0/50', '50/50/0', '80/20/0', '80/0/20']
    rows, speedups, dspeed, dvs2, last = [], [], [], [], None
    for key in sorted(groups, key=lambda k: (k[0], order.index('%d/%d/%d' % k[1:]) if '%d/%d/%d' % k[1:] in order else 9)):
        g = groups[key]
        b1 = best_of([r for r in g if r['prog'] == 'P1'])
        b2 = best_of([r for r in g if r['prog'] == 'P2'])
        bd = best_of([r for r in g if r['prog'] == 'P2D'])
        if last is not None and key[0] != last:
            rows.append('MID')
        last = key[0]
        sp = float(b1['avg_ms']) / float(b2['avg_ms'])
        speedups.append(sp)
        ok = 'OK' if all(r['verify'] == 'OK' for r in g) else 'FAIL'
        row = [p2s(key[0]), '%d/%d/%d' % key[1:], p2s(b1['threads']), ms(b1['avg_ms']),
               p2s(b2['threads']), ms(b2['avg_ms']), '%.2f' % sp]
        if bd:
            dspeed.append(float(b1['avg_ms']) / float(bd['avg_ms']))
            dvs2.append(float(b2['avg_ms']) / float(bd['avg_ms']))
            row += [p2s(bd['threads']), ms(bd['avg_ms']), '%.2f' % dspeed[-1], '%.2f' % dvs2[-1]]
        rows.append(row + [ok])
    if dfin:
        write('q1_final.tex', '\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3.5pt}\n'
              '\\caption{P1 (dynamic, chunk \\ponechunk, block size \\ponebs) vs.\\ P2 with static (cyclic) group '
              'assignment (group \\ptwogroup, block size \\ptwobs) and P2 with dynamic group assignment (chunk '
              '\\ptwodynchunk, group \\ptwodyngroup, block size 128), each at its best thread count. '
              'Ratios $>1$ mean the second program is faster.}\\label{tab:q1final}\n'
              '\\begin{tabular}{ll|rr|rrr|rrrr|c}\\toprule\n'
              '& & \\multicolumn{2}{c|}{P1} & \\multicolumn{3}{c|}{P2 static} & \\multicolumn{4}{c|}{P2 dynamic} & \\\\\n'
              'ops & mix (i/d/s) & thr. & ms & thr. & ms & P1/P2 & thr. & ms & P1/P2d & P2/P2d & verified \\\\\\midrule\n' +
              '\n'.join('\\midrule' if r == 'MID' else ' & '.join(r) + ' \\\\' for r in rows) +
              '\n\\bottomrule\\end{tabular}\\end{table}\n')
        macro('qonedynspeedupmin', '%.2f' % min(dspeed)); macro('qonedynspeedupmax', '%.2f' % max(dspeed))
        macro('qonedynvsstaticmin', '%.2f' % min(dvs2)); macro('qonedynvsstaticmax', '%.2f' % max(dvs2))
    else:
        write('q1_final.tex', table('lllrlrrc', ['ops', 'ins/del/srch \\%', 'P1 threads', 'P1 ms', 'P2 threads', 'P2 ms',
                                                 'P1/P2', 'verified'], rows,
              'P1 (dynamic, chunk \\ponechunk, block size \\ponebs) vs.\\ P2 (group \\ptwogroup, block size \\ptwobs), '
              'each at its best thread count. P1/P2 $>1$ means P2 is faster.', 'tab:q1final'))
    macro('qonespeedupmin', '%.2f' % min(speedups)); macro('qonespeedupmax', '%.2f' % max(speedups))
    macro('qoneallverified', 'All' if all(r['verify'] == 'OK' for r in fin + dfin) else 'NOT all')

dch = load('q1_p2dyn_chunk.csv')
if dch:
    for r in dch:
        r['chunk'] = r['sched'].replace('dynamic-', '')
    write('q1_p2dyn_chunk.tex', matrix(dch, 'chunk', 'threads',
          'P2 with dynamic group assignment: time (ms) vs.\\ chunk size and total threads (group 32, block size 128, '
          '$2^{22}$ inserts). Compare the static column of Table~\\ref{tab:p2group}.', 'tab:p2dynchunk', 'chunk', 'threads'))
    b = best_of(dch)
    macro('ptwodynchunk', b['chunk']); macro('ptwodynbestms', ms(b['avg_ms'])); macro('ptwodynbestthreads', p2s(b['threads']))
dgr = load('q1_p2dyn_group.csv')
if dgr:
    rows = [[r['param'], ms(r['avg_ms'])] for r in sorted(dgr, key=lambda r: int(r['param']))]
    write('q1_p2dyn_group.tex', table('rr', ['group size', 'time (ms)'], rows,
          'P2 dynamic: group size at $2^{20}$ threads, best chunk ($2^{22}$ inserts).', 'tab:p2dyngroup'))
    macro('ptwodyngroup', best_of(dgr)['param'])

sd = load('q1_seeds.csv')
if sd:
    dd = defaultdict(dict); nn = {}
    for r in sd:
        nops = int(r['nops'])
        mix = '%d/%d/%d' % tuple(round(100 * int(r[k]) / nops) for k in ('ins', 'del', 'srch'))
        prog = r['prog'] + ('d' if r['prog'] == 'P2' and r['sched'].startswith('dynamic') else '')
        dd[(mix, r['seed'])][prog] = float(r['avg_ms'])
    seeds_ = ['12345', '2', '3']
    order = ['100/0/0', '50/0/50', '50/50/0', '80/20/0', '80/0/20']
    rows, r12, r2d, same = [], [], [], 0
    for mix in order:
        v = {s: dd[(mix, s)] for s in seeds_ if (mix, s) in dd}
        if len(v) < 3:
            continue
        a12 = [v[s]['P1'] / v[s]['P2'] for s in seeds_]; a2d = [v[s]['P2'] / v[s]['P2d'] for s in seeds_]
        r12 += a12; r2d += a2d
        same += sum(1 for s in seeds_ if v[s]['P2d'] < v[s]['P2'] < v[s]['P1'])
        fmt = lambda prog: ' / '.join('%.0f' % v[s][prog] for s in seeds_)
        rows.append([mix, fmt('P1'), fmt('P2'), fmt('P2d'),
                     '%.2f--%.2f' % (min(a12), max(a12)), '%.2f--%.2f' % (min(a2d), max(a2d))])
    write('q1_seeds.tex', table('lrrrrr', ['mix (i/d/s)', 'P1 ms', 'P2 static ms', 'P2 dynamic ms', 'P1/P2', 'P2/P2d'], rows,
          'Robustness: $2^{22}$ operations generated with three different random seeds (12345, used everywhere '
          'else in this report, 2 and 3; times listed in that order). P1: dynamic chunk 1, $2^{20}$ threads; '
          'P2 static: group 32, $2^{26}$ threads; P2 dynamic: chunk 1, group 32, $2^{18}$ threads.', 'tab:q1seeds'))
    macro('qoneseedratioonemin', '%.1f' % min(r12)); macro('qoneseedratioonemax', '%.1f' % max(r12))
    macro('qoneseedratiotwomin', '%.2f' % min(r2d)); macro('qoneseedratiotwomax', '%.2f' % max(r2d))
    macro('qoneseedsameorder', str(same))

# ------------------------------------------------------------------ Q2
bas = load('q2_basic.csv')
if bas:
    byt = defaultdict(list)
    for r in bas:
        byt[int(r['t'])].append(r)
    rows = []
    allb = best_of(bas)
    for t in sorted(byt):
        b = best_of(byt[t])
        cell = ms(b['avg_ms'])
        rows.append([p2s(t), b['blocksize'], b['blocks'], '\\textbf{%s}' % cell if b is allb else cell, b['gbps'],
                     '%.1e' % float(b['absdiff'])])
    write('q2_basic.tex', table('rrrrrr', ['threads $t$', 'best block size', 'blocks', 'time (ms)', 'GB/s', '$|$diff$|$'],
          rows, 'Basic program ($m=n=2^{14}$): time vs.\\ thread count, each with its best block size.', 'tab:q2basic'))
    write('q2_basic_bs.tex', matrix(bas, 't', 'blocksize',
          'Basic program ($m=n=2^{14}$): time (ms) for every thread count and block size.', 'tab:q2basicbs',
          '$t$', 'block size'))
    macro('qtwobasicbestt', p2s(allb['t'])); macro('qtwobasicbestbs', allb['blocksize']); macro('qtwobasicbestms', ms(allb['avg_ms']))

ver = load('q2_versions.csv')
names = [('v1_basic', 'basic (contiguous chunks)'), ('v2_coalesced', '+ coalescing'),
         ('v3_tree', '+ tree reduction (global mem.)'), ('v4_shmem', '+ shared memory'),
         ('v5_opt', '+ float4, ILP, warp shuffle')]
VW = {'v1_basic': 'vone', 'v2_coalesced': 'vtwo', 'v3_tree': 'vthree', 'v4_shmem': 'vfour', 'v5_opt': 'vfive'}
if ver and bas:
    bt = allb['t']
    rows = []
    base = None
    for v, label in names:
        at = best_of([r for r in ver if r['version'] == v and r['t'] == bt])
        own = best_of([r for r in ver if r['version'] == v])
        if at is None:
            continue
        if base is None:
            base = float(at['avg_ms'])
        rows.append([label, at['blocksize'], ms(at['avg_ms']), at['gbps'], '%.2f' % (base / float(at['avg_ms'])),
                     p2s(own['t']), own['blocksize'], ms(own['avg_ms']), own['gbps']])
        macro('qtwo%sms' % VW[v], ms(own['avg_ms']))
        macro('qtwo%sgbps' % VW[v], own['gbps'])
    write('q2_versions.tex', table('lrrrr|rrrr',
          ['version', 'bs', 'ms', 'GB/s', 'speedup', 'best $t$', 'bs', 'ms', 'GB/s'], rows,
          'Effect of each optimisation ($m=n=2^{14}$). Left: at the best thread count of the basic program '
          '($t=$\\,\\qtwobasicbestt, best block size per version). Right: each version at its own best $(t$, block size$)$.',
          'tab:q2versions'))
    # trend: version x t (best over block size)
    vr = [dict(r, version=dict(names)[r['version']].split(' (')[0]) for r in ver]
    order = {lab.split(' (')[0]: i for i, (_, lab) in enumerate(names)}
    cell = defaultdict(lambda: None)
    ts = sorted({int(r['t']) for r in ver})
    for r in vr:
        k = (r['version'], int(r['t']))
        if cell[k] is None or float(r['avg_ms']) < float(cell[k]['avg_ms']):
            cell[k] = r
    rows = []
    for vname in sorted(order, key=order.get):
        rows.append([vname] + [ms(cell[(vname, t)]['avg_ms']) if cell[(vname, t)] else '--' for t in ts])
    write('q2_trend.tex', '\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n'
          '\\caption{Time (ms) of every version vs.\\ thread count $t$ ($m=n=2^{14}$, best block size for each cell).}'
          '\\label{tab:q2trend}\n\\begin{tabular}{l' + 'r' * len(ts) + '}\\toprule\n'
          'version & ' + ' & '.join(p2s(t) for t in ts) + ' \\\\\\midrule\n' +
          '\n'.join(' & '.join(r) + ' \\\\' for r in rows) + '\n\\bottomrule\\end{tabular}\\end{table}\n')

cmp_ = load('q2_compare.csv')
if cmp_:
    keys = []
    for r in cmp_:
        k = (int(r['m']), int(r['n']))
        if k not in keys:
            keys.append(k)
    rows, omp_rows = [], []
    sps = []
    for k in keys:
        g = best_of([r for r in cmp_ if (int(r['m']), int(r['n'])) == k and r['version'] != 'omp'])
        o = best_of([r for r in cmp_ if (int(r['m']), int(r['n'])) == k and r['version'] == 'omp'])
        sp = float(o['avg_ms']) / float(g['avg_ms'])
        sps.append(sp)
        rows.append([p2s(k[0]), p2s(k[1]), '%.0f' % (k[0] * k[1] * 4 / 2**20), p2s(g['t']), g['blocksize'], ms(g['avg_ms']),
                     g['gbps'], '%.1e' % float(g['absdiff']), o['t'], ms(o['avg_ms']), o['gbps'], '%.1f' % sp])
        omp = sorted([r for r in cmp_ if (int(r['m']), int(r['n'])) == k and r['version'] == 'omp'], key=lambda r: int(r['t']))
        omp_rows.append([p2s(k[0]), p2s(k[1])] + [ms(r['avg_ms']) for r in omp])
        omp_ts = [r['t'] for r in omp]
    write('q2_compare.tex', '\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n'
          '\\caption{Best CUDA program (v5) vs.\\ best OpenMP program. Speedup = OpenMP time / CUDA time.}\\label{tab:q2compare}\n'
          '\\begin{tabular}{rrr|rrrrr|rrr|r}\\toprule\n'
          '$m$ & $n$ & A (MB) & $t$ & bs & CUDA ms & GB/s & $|$diff$|$ & OMP thr. & OMP ms & GB/s & speedup \\\\\\midrule\n' +
          '\n'.join(' & '.join(r) + ' \\\\' for r in rows) + '\n\\bottomrule\\end{tabular}\\end{table}\n')
    write('q2_omp.tex', table('rr' + 'r' * len(omp_ts), ['$m$', '$n$'] + ['%s thr.' % t for t in omp_ts], omp_rows,
          'OpenMP program: time (ms) vs.\\ number of threads.', 'tab:q2omp'))
    macro('qtwospeedupmin', '%.1f' % min(sps)); macro('qtwospeedupmax', '%.1f' % max(sps))

write('macros.tex', '\n'.join(macros) + '\n')
print('Wrote tables to', GEN)
