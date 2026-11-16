# Builds the Markdown result tables for the report from results/*.csv.
import csv, re, sys
from collections import defaultdict

out = []

# ---------------- Q1 ----------------
rows = list(csv.DictReader(open('results/q1_times.csv')))
by = defaultdict(dict)
for r in rows:
    by[r['graph']][int(r['threads'])] = r
names = {
    'g1_random_sparse': 'G1: sparse random + planted triangles',
    'g2_random_dense': 'G2: denser random + planted triangles',
    'g3_clustered': 'G3: clustered (communities of 20, p = 0.5)',
    'g4_powerlaw': 'G4: power-law (Holme-Kim, m = 5, p_t = 0.5)',
    'g5_random_large': 'G5: large sparse random + planted triangles',
}
for g, per in by.items():
    info = open(f'results/q1_info_{g}.txt').read()
    n, m, tri, eh = re.search(r'n = (\d+), m = (\d+), triangles = (\d+), \|E\(H\)\| = (\d+)', info).groups()
    k0, k, lb = re.search(r'initial = (\d+), final = (\d+), lower bound = (\d+)', info).groups()
    jp = open(f'results/q1_jp_{g}.txt').read()
    jk0, jk = re.search(r'initial = (\d+), final = (\d+)', jp).groups()
    out.append(f"\n#### {names.get(g, g)}\n")
    out.append(f"n = {int(n):,} vertices, m = {int(m):,} edges, {int(tri):,} triangles; "
               f"conflict graph H: {int(n) + int(m):,} nodes, {int(eh):,} edges. "
               f"Colors: {k} (DSATUR {k0} before elimination; clique lower bound {lb}).\n")
    out.append("| Threads | Total time (s) | Speedup | Build H (s) | Coloring (s) | Improve (s) |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    base = float(per[1]['median_total_s'])
    for t in sorted(per):
        r = per[t]
        tt = float(r['median_total_s'])
        out.append(f"| {t} | {tt:.3f} | {base / tt:.2f} | {float(r['build_s']):.3f} | {float(r['color_s']):.3f} | {float(r['improve_s']):.3f} |")
    out.append("")

out.append("\n#### Coloring quality: DSATUR vs largest-degree-first (Jones-Plassmann)\n")
out.append("| Graph | JP-LF | JP-LF + elimination | Batched DSATUR | DSATUR + elimination | Lower bound |")
out.append("|---|---:|---:|---:|---:|---:|")
for g in by:
    info = open(f'results/q1_info_{g}.txt').read()
    k0, k, lb = re.search(r'initial = (\d+), final = (\d+), lower bound = (\d+)', info).groups()
    jk0, jk = re.search(r'initial = (\d+), final = (\d+)', open(f'results/q1_jp_{g}.txt').read()).groups()
    out.append(f"| {g.split('_')[0].upper()} | {jk0} | {jk} | {k0} | {k} | {lb} |")

# ---------------- Q2 ----------------
rows = list(csv.DictReader(open('results/q2_times.csv')))
q = defaultdict(dict)
for r in rows:
    q[(r['algo'], int(r['n']))][int(r['threads'])] = float(r['median_s'])
ns = sorted({n for (_, n) in q})
for n in ns:
    p, b = q[('pipe', n)], q[('barrier', n)]
    mb = 2 * (n + 1) * (n + 2) * 8 / 1e6
    out.append(f"\n#### n = {n:,} (matrix {2 * n + 1:,} x {2 * n + 1:,}, {mb:,.0f} MB of non-zeros)\n")
    out.append("| Threads | Pipelined (ms) | Speedup | GB/s | Barrier version (ms) | Speedup |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    for t in sorted(p):
        out.append(f"| {t} | {p[t] * 1e3:.2f} | {p[1] / p[t]:.2f} | {mb / 1e3 / p[t]:.1f} | {b[t] * 1e3:.2f} | {b[1] / b[t]:.2f} |")

open('results/tables.md', 'w').write("\n".join(out) + "\n")
print("\n".join(out))
