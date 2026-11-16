# Brute-force minimum colors for tiny graphs (independent of q1.cpp's search).
import sys, itertools
lines = open(sys.argv[1]).read().split('\n')
import re
n = int(re.findall(r'\d+', lines[0])[0]); E = set()
for k in range(n):
    if k + 1 < len(lines):
        for t in re.findall(r'\d+', lines[k + 1]):  # any non-digit is a separator
            j = int(t)
            if j != k and j < n:
                E.add((min(k, j), max(k, j)))  # duplicates / reversed pairs = one edge
E = sorted(E)
Es = set(E); nodes = n + len(E); adj = [set() for _ in range(nodes)]
def link(a, b): adj[a].add(b); adj[b].add(a)
eid = {e: n + i for i, e in enumerate(E)}
for (u, v) in E: link(u, v); link(u, eid[(u, v)]); link(v, eid[(u, v)])
for a, b, c in itertools.combinations(range(n), 3):
    if (a, b) in Es and (b, c) in Es and (a, c) in Es:
        link(eid[(a, b)], eid[(b, c)]); link(eid[(a, b)], eid[(a, c)]); link(eid[(b, c)], eid[(a, c)])
def ok(k):
    col = [-1] * nodes
    def go(i):
        if i == nodes: return True
        for c in range(min(k, max(col[:i] + [-1]) + 2)):
            if all(col[j] != c for j in adj[i]):
                col[i] = c
                if go(i + 1): return True
        col[i] = -1
        return False
    return go(0)
k = 1 if n else 0
while n and not ok(k): k += 1
print(k)
