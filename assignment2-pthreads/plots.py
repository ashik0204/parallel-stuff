#!/usr/bin/env python3
"""Turns results/q1.csv and results/q2.csv into figures and LaTeX tables in report/."""
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "report", "figs")
TAB = os.path.join(HERE, "report", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
STYLE = {1: ("tab:blue", "o"), 2: ("tab:orange", "s"), 3: ("tab:green", "^"),
         4: ("tab:red", "v"), 5: ("tab:purple", "D"), 6: ("tab:brown", "P")}


TIMEOUT = float("inf")


def medians(path, keys):
    """Median time per configuration; a run that hit the time limit (-1) counts as infinite."""
    groups = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            t = int(row["time_us"])
            groups[tuple(row[k] for k in keys)].append(TIMEOUT if t < 0 else t)
    return {k: statistics.median(v) for k, v in groups.items()}


# ---------------------------------------------------------------- Q1
Q1_NAMES = {1: "P1 lock-free FIFO sem", 2: "P2 spin-lock sem",
            3: "P3 Linux sem_t", 4: "P4 cond. variable"}
q1 = medians(os.path.join(HERE, "results", "q1.csv"), ["version", "N", "P", "n"])
TOTAL = 4000000

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax, N in zip(axes, [8, 64, 1024]):
    for v in range(1, 5):
        Ps = sorted({int(k[2]) for k in q1 if k[0] == str(v) and k[1] == str(N) and int(k[3]) == TOTAL // int(k[2])})
        ys = [q1[(str(v), str(N), str(P), str(TOTAL // P))] / 1000 for P in Ps]
        c, m = STYLE[v]
        ok = [(P, y) for P, y in zip(Ps, ys) if y != TIMEOUT]
        ax.plot([p for p, _ in ok], [y for _, y in ok], marker=m, color=c, label=Q1_NAMES[v], ms=4)
        for P, y in zip(Ps, ys):
            if y == TIMEOUT:
                ax.plot([P], [120000], marker="x", color=c, ms=7, ls="none",
                        label="timed out (> 120 s)" if N == 8 and P == 23 else None)
    ax.axvline(15.5, color="gray", ls=":", lw=1)
    ax.set_title(f"N = {N}")
    ax.set_xlabel("producers P (nP = 4M)")
    ax.set_yscale("log")
axes[0].set_ylabel("time (ms, log scale)")
axes[0].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "q1_sweep.pdf"))

fig, ax = plt.subplots(figsize=(4, 2.8))
for v in range(1, 5):
    ns = sorted({int(k[3]) for k in q1 if k[0] == str(v) and k[1] == "64" and k[2] == "8"})
    ys = [q1[(str(v), "64", "8", str(n))] / 1000 for n in ns]
    c, m = STYLE[v]
    ax.plot([n * 8 / 1e6 for n in ns], ys, marker=m, color=c, label=Q1_NAMES[v], ms=4)
ax.set_xlabel("total values nP (millions), N = 64, P = 8")
ax.set_ylabel("time (ms)")
ax.legend(fontsize=7)
fig.savefig(os.path.join(FIG, "q1_n.pdf"))

with open(os.path.join(TAB, "q1.tex"), "w") as f:
    Ps = [1, 2, 4, 8, 12, 15, 23, 31]
    f.write("\\begin{tabular}{rl" + "r" * len(Ps) + "}\n\\toprule\n")
    f.write("$N$ & Version & " + " & ".join(f"$P{{=}}{P}$" for P in Ps) + " \\\\\n\\midrule\n")
    for N in [8, 64, 1024]:
        for v in range(1, 5):
            cells = []
            for P in Ps:
                t = q1[(str(v), str(N), str(P), str(TOTAL // P))]
                best = min(q1[(str(u), str(N), str(P), str(TOTAL // P))] for u in range(1, 5))
                s = "$>$120k" if t == TIMEOUT else f"{t/1000:.0f}"
                cells.append(f"\\textbf{{{s}}}" if t == best else s)
            f.write((f"{N}" if v == 1 else "") + f" & P{v} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\midrule\n" if N != 1024 else "\\bottomrule\n")
    f.write("\\end{tabular}\n")

# ---------------------------------------------------------------- Q2
Q2_NAMES = {1: "P1 global mutex", 2: "P2 mutex/list", 3: "P3 mutex/node",
            4: "P4 rwlock/node", 5: "P5 CAS lock/node", 6: "P6 optimistic"}
RATIOS = [("80_10_10", "80% ins / 10% del / 10% search"),
          ("30_30_40", "30% ins / 30% del / 40% search"),
          ("10_10_80", "10% ins / 10% del / 80% search")]
OPS = [1000000, 2000000, 4000000]
q2 = medians(os.path.join(HERE, "results", "q2.csv"), ["version", "ratio", "ops", "threads"])
THREADS = sorted({int(k[3]) for k in q2})


def tput(v, r, ops, T):
    return ops / q2[(str(v), r, str(ops), str(T))]  # ops per microsecond = Mops/s


for ops in OPS:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, (r, title) in zip(axes, RATIOS):
        for v in range(1, 7):
            c, m = STYLE[v]
            ax.plot(THREADS, [tput(v, r, ops, T) for T in THREADS], marker=m, color=c, label=Q2_NAMES[v], ms=4)
        ax.axvline(16, color="gray", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("threads")
        ax.set_xticks(THREADS)
    axes[0].set_ylabel(f"throughput (Mops/s), {ops//1000000}M ops")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(os.path.join(FIG, f"q2_tput_{ops//1000000}M.pdf"))

with open(os.path.join(TAB, "q2.tex"), "w") as f:
    f.write("\\begin{tabular}{ll" + "r" * len(THREADS) + "}\n\\toprule\n")
    f.write("Mix & Ops & " + " & ".join(f"$T{{=}}{T}$" for T in THREADS) + " \\\\\n\\midrule\n")
    for r, _ in RATIOS:
        for ops in OPS:
            cells = []
            for T in THREADS:
                best = max(range(1, 7), key=lambda v: tput(v, r, ops, T))
                bt = tput(best, r, ops, T)
                cells.append(f"P{best} ({bt:.1f})")
            f.write(f"{r.replace('_', '/')} & {ops//1000000}M & " + " & ".join(cells) + " \\\\\n")
        f.write("\\midrule\n" if r != RATIOS[-1][0] else "\\bottomrule\n")
    f.write("\\end{tabular}\n")

with open(os.path.join(TAB, "q2_full.tex"), "w") as f:
    for r, title in RATIOS:
        f.write("\\begin{tabular}{ll" + "r" * len(THREADS) + "}\n\\toprule\n")
        f.write(f"\\multicolumn{{{len(THREADS)+2}}}{{c}}{{\\textbf{{{title.replace("%", r"\%")}}} (time in ms)}} \\\\\n\\midrule\n")
        f.write("Ops & Ver. & " + " & ".join(f"$T{{=}}{T}$" for T in THREADS) + " \\\\\n\\midrule\n")
        for ops in OPS:
            for v in range(1, 7):
                cells = []
                for T in THREADS:
                    t = q2[(str(v), r, str(ops), str(T))]
                    best = min(q2[(str(u), r, str(ops), str(T))] for u in range(1, 7))
                    s = f"{t/1000:.0f}"
                    cells.append(f"\\textbf{{{s}}}" if t == best else s)
                f.write((f"{ops//1000000}M" if v == 1 else "") + f" & P{v} & " + " & ".join(cells) + " \\\\\n")
            f.write("\\midrule\n" if ops != OPS[-1] else "\\bottomrule\n")
        f.write("\\end{tabular}\n\\vspace{1em}\n\n")

print("figures in", FIG, "and tables in", TAB)
