# CS433 Assignment #1 — Report

**Group 1** — *Ashik Stenny*, *Spandan Pati*, *Vedhanth Balasubhramanian*

Both programs are written in C++ with **OpenMP** (no POSIX threads, no other libraries).

| | Question 1 | Question 2 |
|---|---|---|
| Source file | `q1.cpp` | `q2.cpp` |
| Parallel API | OpenMP | OpenMP |
| Compile | `g++ -O3 -fopenmp q1.cpp -o q1` | `g++ -O3 -fopenmp q2.cpp -o q2` |
| Run | `./q1 <input> <output> <threads>` | `./q2 <input> <output> <threads>` |

**Test machine.** AMD Ryzen 7 5800HS laptop: 8 cores / 16 hardware threads (SMT), 16 MB L3, dual-channel DDR4, 16 GB RAM; Ubuntu (Linux 6.17), g++ 15.2. Both programs were compiled with `-O3 -fopenmp`, and all timings were taken on AC power.

**Timing.** Every program measures its own time with `omp_get_wtime()` around the computation only. Reading the input, allocating memory, initializing arrays, and writing the output are not included. In Q2 the timed region is exactly the solve. Q1 has allocations between its phases (for example, the conflict graph's size is known only after the counting pass), so it uses a clock that is paused around every allocation and array initialization and resumed afterwards. Each configuration was run several times (Q1: 3 runs, Q2: 5 runs), and the tables report the **median**.

---

## Question 1: Coloring vertices and edges with triangle constraints

### 1.1 Reformulation as graph coloring

The rules say that two objects must get different colors when they are:
(i) the two end points of an edge; (ii) an edge and one of its end points; (iii) two edges of the same triangle.

We build a **conflict graph H**. Its nodes are the *n* vertices **and** the *m* edges of the input graph G, and two nodes of H are adjacent exactly when one of rules (i)–(iii) applies to them:

* vertex *u* — vertex *v* if (u, v) ∈ E,
* vertex *u* — edge *e* if *u* is an end point of *e*,
* edge *e* — edge *f* if *e* and *f* lie on a common triangle.

A proper vertex coloring of H is exactly a valid coloring for the assignment, so we must find the chromatic number of H. This problem is NP-hard. We therefore (a) compute a high-quality coloring with a parallel heuristic, (b) compute a lower bound, which proves optimality when the two meet, and (c) for small graphs, run an exact branch and bound. Any graph with an edge needs at least 3 colors (u, v, and (u, v)). A complete graph K₄ (the sample input) needs 5, which our exact search proves.

### 1.2 Algorithm and parallelization

**Input.** The whole file is read into memory. Line *k* (after the first) gives the neighbors *i > k* of vertex *k*, and empty or missing lines are allowed. Edges are sorted and de-duplicated, and G is stored in CSR form with sorted neighbor lists. Each neighbor entry also stores the id of the connecting edge.

**Step 1 — build H (triangle listing), parallel.**
For an edge *e = (u, v)*, the triangles through *e* are the common neighbors *w* of *u* and *v*. Because the lists are sorted, a linear merge of adj(u) and adj(v) finds them. The build has two parallel passes over all nodes of H:
1. **Count pass:** compute the degree of every node in H (vertex *u*: 2·deg(u); edge *e*: 2 + 2·#triangles(e)).
2. A parallel prefix sum gives the CSR offsets.
3. **Fill pass:** every node writes its own adjacency list. Edge *e* writes its end points and, for each common neighbor *w*, the edges (u, w) and (v, w).

Each node writes only its own slice of the array, so no locks or atomics are needed. `schedule(dynamic)` balances the very uneven merge costs, since degrees are skewed in the power-law graph.

In this scheme every triangle is found three times per pass (once from each of its edges). For **dense graphs (average degree ≥ 16)** the program instead finds every triangle exactly once:
* Vertices are ranked by (degree, number), and every vertex keeps only its higher-ranked neighbors ("forward" lists). A triangle is found only from its lowest-ranked vertex.
* It then appends entries to the lists of all three of its edges through atomic cursors.
* Finally each edge's list is sorted by the third vertex, so H, and therefore the whole output, is exactly the same as in the other mode.

We chose the threshold by measurement (see 1.3.1). On dense graphs this halves the time of this step. On sparse graphs the merges it saves are short, and its extra passes and scattered atomic updates cost more.

**Step 2 — batched parallel DSATUR.**
We first tried the classic parallel greedy algorithm (Jones–Plassmann with largest-degree-first priority). It was fast but used 1–3 more colors than sequential DSATUR [Brélaz 1979] (see the quality table in 1.4). The reason: the answer is dominated by how well the *vertices* are colored, and on H, DSATUR's "most-constrained node first" rule is much better than degree order. We therefore parallelized DSATUR itself:

* The *saturation* of a node is the number of distinct colors among its colored neighbors. Sequential DSATUR always colors next an uncolored node of maximum saturation.
* In every **round**, let *L* be the current maximum saturation. Among the uncolored nodes of saturation *L*, we select every node that beats all of its uncolored neighbors of saturation *L* in a fixed (degree, hash) order, i.e. a local maximum, as in Luby / Jones–Plassmann. **The selected nodes are pairwise non-adjacent**, so all of them are colored **in parallel** (smallest color not used by a neighbor) with no conflicts and no locks.
* The colors are then pushed to the uncolored neighbors in parallel. Each node keeps a 128-bit bitset of the colors it has seen, updated with an atomic `fetch-or`; the saturation grows only when a bit is newly set. Larger colors fall back to a recount.
* Nodes are kept in **buckets by saturation level**. An entry goes stale when its node is colored or rises to a higher level, and stale entries are dropped lazily when the bucket is scanned. New entries are appended in parallel: each thread counts per level, one thread reserves space, then all threads copy.
* When no uncolored node has a colored neighbor (at the start, or when a new connected component begins), the uncolored node of largest degree is colored alone, exactly like sequential DSATUR. It is taken from a degree-sorted list, so this costs O(1) amortized.
* **Small rounds run on one thread.** Rounds with fewer than 128 candidates have too little work to pay for about nine barriers, so one thread executes them inside a `single` section (one barrier).
* A subtle race had to be avoided: an `omp single` does not wait for the other threads at its start. So a fast thread may already be modifying the shared state of round *r+1* while a slow thread is still deciding what kind of round *r* is. The decision is therefore double-buffered by round parity.

This batched version picks many nodes per round while respecting DSATUR's order across saturation levels. On our graphs it gives the same or nearly the same number of colors as sequential DSATUR.

**Step 3 — parallel top-class elimination (local search).**
Let *t = K − 1* be the highest color. For every node *v* of color *t* (these nodes are independent, so all are handled at once), we look for
* a color *c < t* that no neighbor of *v* uses (move *v* to *c*), or
* a color *c < t* used by only *k ≤ 4* neighbors *w₁..w_k* of *v*, each of which can switch to another color *d_i < t*. We move the *w_i*, then move *v* to *c*. The *w_i* all have color *c*, so they are non-adjacent and can move together.

Proposals are computed in parallel. Two proposals conflict if a node changed by one equals, or is adjacent to, a node changed by the other. Each proposal therefore **claims** the closed neighborhoods of its changed nodes with an atomic minimum of its index, and it is applied only if it owns all its claims. Accepted proposals never interfere, and the smallest proposal always wins, so every pass makes progress. When class *t* becomes empty, *K* decreases and the next class is tried. The step stops when no move is possible or when *K* reaches the lower bound.

**Lower bound and exact search.** A greedy clique search in H, started from the 2048 highest-degree nodes in parallel, gives a lower bound. If the coloring reaches it, the coloring is optimal. For small instances (|V(H)| ≤ 400) an exact DSATUR branch and bound is run, with the heuristic result as its initial upper bound. On all 200+ tiny random graphs we tested, the answer matched an independent brute-force search.

**Determinism.** None of the steps depends on the scheduling of threads: selection uses fixed priorities, and conflicts are resolved by fixed indices. **The output is identical for every thread count.** We verified this with byte-for-byte comparison on hundreds of graphs. As a result, the timing tables below compare exactly the same work at each thread count.

### 1.3 Optimizations (summary)

* Reformulation as coloring of one conflict graph: all three rules are handled uniformly by the same coloring code.
* Sorted CSR with edge ids and merge-based triangle listing: fully parallel, with no locks and no hash tables.
* **Adaptive triangle listing:** on dense graphs (average degree ≥ 16) each triangle is found once instead of three times, with identical output. On G2 this cut the time to build H at 12 threads from 0.31 s to 0.17 s.
* Two-pass (count/fill) construction with a parallel prefix sum, so every thread writes a disjoint slice.
* Conflict-free parallel coloring rounds (independent sets), so there are no speculative recolorings and no atomics on colors.
* Per-node color bitsets with atomic `fetch-or`, preceded by a plain atomic read that skips most atomic writes.
* Lazily cleaned saturation buckets, degree-sorted start list, and a single-thread path for small rounds.
* **Coloring and saturation update fused into one pass.** Picked nodes are never adjacent, so a node's neighbors can be updated right after it is colored, while its adjacency list is still in cache. This saves one pass and one barrier per round, and made the coloring phase 3–6% faster at 8–12 threads.
* Dynamic scheduling everywhere the work per item is irregular, with chunks of 64–1024 iterations so that consecutive array entries written by one thread (e.g. degree counts) stay in the same cache lines and are not falsely shared.
* **No false sharing on per-thread lists.** Every thread appends selected nodes to its own `std::vector`. The vectors' 24-byte headers, which each `push_back` writes, were originally adjacent in memory, so 2–3 threads' headers shared a 64-byte cache line and invalidated each other's caches on every append. Padding each header to more than 64 bytes removed this. It sped up the coloring phase by 5–10% at 8 threads (G3: 0.74 s → 0.67 s) with no effect on 1 thread, as expected.
* Parallel, deterministic local search (claim/accept) that removes whole color classes.

#### 1.3.1 Optimizations we measured (kept and rejected)

For the conflict-graph construction we implemented four further ideas, each behind a switch, and timed each one alone on all five graphs (median of 3 runs). The table shows the change in the time to build H relative to the plain version. Each idea that should not change the result was checked to give byte-identical output.

| Idea | 1 thread | 12 threads | Decision |
|---|---|---|---|
| 32-bit instead of 64-bit per-node counts (less memory traffic) | −5% … +9% | −9% … +6% | rejected: within noise |
| "Galloping" intersection (binary search when one list is ≥ 8× longer) | −4% … +1% | −3% … +3% | rejected: lists are never lopsided enough |
| Each triangle found once (degree-ordered forward lists, atomic cursors) | −25% … −55% | G2 (dense) **−47%**; sparse graphs +9% … +49% | **kept, for average degree ≥ 16** |
| Renumbering vertices for locality (BFS order or decreasing degree) | 0% … +52% | +16% … +100% | rejected: slower |

We did the same for two ideas in the coloring phase (change in coloring time, 8 and 12 threads, interleaved runs):

| Idea | 8–12 threads | Decision |
|---|---|---|
| Color and update neighbors' saturation in one pass (see 1.3) | −3% … −6% | **kept** |
| Pack each node's color, saturation, degree, tie-break key and color bitset into one 32-byte record | **+7% … +20%** (slower) | rejected |

Lessons:
1. **"Triangles once" does less work but more communication.** Its atomic updates go to random edges, so cache lines move between cores. It therefore wins in parallel only when the merges it saves are long. On extra random graphs of average degree 16, 24, 32 and 44 it reduced this step's time at 12 threads by 23%, 33%, 38% and 45%; on our sparse graphs (average degree 7–11.5) it was slower. We therefore enable it at average degree ≥ 16.
2. **Packing data per node can backfire.** The idea was one cache miss per neighbor instead of several, but it failed for two reasons:
   * The field read most often, the color, grew from a compact 4-byte array (20 MB for G1, close to fitting in the 16 MB L3) into 32-byte records (157 MB), so color lookups miss the cache more often.
   * Each record mixed fields written with atomics (saturation, color bitset) with fields other threads only read (color, degree), so every atomic update invalidated the other cores' copies of the whole line.
3. **Renumbering can't create locality that isn't there.** The random graphs have none to recover, and the clustered and power-law graphs are already numbered community by community, so reordering only added a sequential pass.

### 1.4 Results

Test graphs were produced by our generator `tools/gen_graph.cpp`. It builds random graphs with planted random triangles (random triplets of vertices connected by three edges, as suggested in the assignment), a community (clustered) graph with very many triangles, and a power-law graph (Holme–Kim preferential attachment with triad formation). Every output was checked by an independent checker (`tools/check_coloring.cpp`), which verifies rules (i)–(iii) directly on G, and checks that every vertex and edge is listed once and that the colors are exactly 0..K−1.

Times are in seconds (median of 3 runs), computation only: allocation and array initialization are excluded. *Build H* = triangle listing and conflict graph construction; *Coloring* = batched DSATUR; *Improve* = lower bound and top-class elimination.

#### G1: sparse random + planted triangles

n = 1,000,000 vertices, m = 3,899,986 edges, 300,091 triangles; conflict graph H: 4,899,986 nodes, 12,600,231 edges. Colors: 6 (DSATUR 7 before elimination; clique lower bound 3).

| Threads | Total time (s) | Speedup | Build H (s) | Coloring (s) | Improve (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.845 | 1.00 | 1.107 | 2.716 | 0.022 |
| 2 | 2.135 | 1.80 | 0.569 | 1.551 | 0.016 |
| 4 | 1.337 | 2.88 | 0.284 | 1.038 | 0.015 |
| 6 | 1.098 | 3.50 | 0.197 | 0.885 | 0.015 |
| 8 | 0.982 | 3.92 | 0.160 | 0.807 | 0.015 |
| 12 | 0.948 | 4.06 | 0.130 | 0.802 | 0.016 |
| 16 | 1.091 | 3.52 | 0.118 | 0.953 | 0.019 |


#### G2: denser random + planted triangles

n = 100,000 vertices, m = 3,099,044 edges, 239,867 triangles; conflict graph H: 3,199,044 nodes, 10,016,733 edges. Colors: 19 (DSATUR 19 before elimination; clique lower bound 3).

| Threads | Total time (s) | Speedup | Build H (s) | Coloring (s) | Improve (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.693 | 1.00 | 1.456 | 1.225 | 0.012 |
| 2 | 1.548 | 1.74 | 0.778 | 0.756 | 0.014 |
| 4 | 0.972 | 2.77 | 0.399 | 0.563 | 0.009 |
| 6 | 0.755 | 3.57 | 0.273 | 0.468 | 0.014 |
| 8 | 0.660 | 4.08 | 0.216 | 0.434 | 0.010 |
| 12 | 0.638 | 4.22 | 0.174 | 0.444 | 0.020 |
| 16 | 0.676 | 3.98 | 0.169 | 0.485 | 0.023 |


#### G3: clustered (communities of 20, p = 0.5)

n = 500,000 vertices, m = 2,874,567 edges, 3,560,517 triangles; conflict graph H: 3,374,567 nodes, 19,305,252 edges. Colors: 10 (DSATUR 11 before elimination; clique lower bound 8).

| Threads | Total time (s) | Speedup | Build H (s) | Coloring (s) | Improve (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.519 | 1.00 | 0.524 | 1.986 | 0.009 |
| 2 | 1.523 | 1.65 | 0.287 | 1.227 | 0.008 |
| 4 | 0.985 | 2.56 | 0.150 | 0.827 | 0.008 |
| 6 | 0.805 | 3.13 | 0.107 | 0.689 | 0.008 |
| 8 | 0.718 | 3.51 | 0.084 | 0.626 | 0.009 |
| 12 | 0.695 | 3.62 | 0.069 | 0.617 | 0.009 |
| 16 | 0.740 | 3.40 | 0.060 | 0.671 | 0.010 |


#### G4: power-law (Holme-Kim, m = 5, p_t = 0.5)

n = 500,000 vertices, m = 2,499,985 edges, 1,004,155 triangles; conflict graph H: 2,999,985 nodes, 10,512,420 edges. Colors: 8 (DSATUR 8 before elimination; clique lower bound 6).

| Threads | Total time (s) | Speedup | Build H (s) | Coloring (s) | Improve (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.572 | 1.00 | 0.857 | 1.704 | 0.012 |
| 2 | 1.458 | 1.76 | 0.451 | 0.998 | 0.009 |
| 4 | 0.906 | 2.84 | 0.229 | 0.669 | 0.007 |
| 6 | 0.722 | 3.56 | 0.159 | 0.555 | 0.008 |
| 8 | 0.636 | 4.04 | 0.126 | 0.500 | 0.010 |
| 12 | 0.613 | 4.20 | 0.108 | 0.490 | 0.015 |
| 16 | 0.636 | 4.05 | 0.095 | 0.521 | 0.020 |


#### G5: large sparse random + planted triangles

n = 2,000,000 vertices, m = 6,999,983 edges, 1,000,052 triangles; conflict graph H: 8,999,983 nodes, 24,000,105 edges. Colors: 6 (DSATUR 6 before elimination; clique lower bound 3).

| Threads | Total time (s) | Speedup | Build H (s) | Coloring (s) | Improve (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.895 | 1.00 | 2.161 | 5.698 | 0.036 |
| 2 | 4.518 | 1.75 | 1.099 | 3.374 | 0.044 |
| 4 | 3.003 | 2.63 | 0.593 | 2.367 | 0.044 |
| 6 | 2.476 | 3.19 | 0.418 | 2.016 | 0.042 |
| 8 | 2.254 | 3.50 | 0.332 | 1.879 | 0.043 |
| 12 | 2.145 | 3.68 | 0.289 | 1.810 | 0.046 |
| 16 | 2.339 | 3.37 | 0.261 | 2.030 | 0.048 |


#### Coloring quality: DSATUR vs largest-degree-first (Jones-Plassmann)

| Graph | JP-LF | JP-LF + elimination | Batched DSATUR | DSATUR + elimination | Lower bound |
|---|---:|---:|---:|---:|---:|
| G1 | 7 | 6 | 7 | 6 | 3 |
| G2 | 21 | 19 | 19 | 19 | 3 |
| G3 | 13 | 11 | 11 | 10 | 8 |
| G4 | 8 | 8 | 8 | 8 | 6 |
| G5 | 7 | 6 | 6 | 6 | 3 |



### 1.5 Discussion

**Overall speedup.** With 12 threads, the measured time drops by a factor of **3.6–4.2**, e.g. from 3.85 s to 0.95 s on G1 and from 7.90 s to 2.15 s on G5. With 8 threads (one per physical core) the speedup is 3.5–4.1×. Every thread count produces exactly the same coloring, so these speedups compare identical work.

**Building H scales very well: 8.3–9.3× at 16 threads.** Every node of H is processed independently: a merge of two sorted adjacency lists, then a write into a private slice of the output. There are no atomics (in the sparse mode), only a few barriers, and a prefix sum. This phase even gains beyond the 8 physical cores, because two SMT threads on one core hide each other's cache-miss latency.

**G2 and the "triangles once" mode.** G2 is the only dense graph (average degree 62), so it uses the mode where each triangle is found once. That made its single-thread time much smaller than in the plain mode (2.69 s instead of about 4.0 s), and its 12-thread time dropped from 0.77 s to 0.64 s. Its *speedup ratio* (4.2×) is lower than it was in the plain mode (5.2×), but only because the phase that scaled best got much cheaper: the absolute time is what improved.

**The coloring phase scales less (2.8–3.5×)** for three reasons:
1. **Memory latency.** H has 3–9 million nodes and 10–24 million edges, i.e. hundreds of MB, far more than the 16 MB L3. Every neighbor visit reads the color, saturation, and bitset of a random node, which is almost always a cache miss. Such a workload is limited by the number of outstanding memory requests the chip can sustain, not by the number of cores.
2. **Synchronization per round.** DSATUR is inherently sequential. The batched version needs 300–40,000 rounds, and every large round has about 9 barriers.
3. **Amdahl's law.** Rounds with fewer than 128 candidates, and the bucket reservation, run on one thread. G3 has by far the most rounds (about 40,000, because its many dense communities keep the saturation levels small), so its single-thread part is the largest.

Because coloring takes 66–80% of the serial time on the sparse graphs, it sets the overall speedup. G3 (79% coloring, most rounds) and G5 (the largest, most memory-bound graph) scale least, at 3.6–3.7×.

**Improvement step.** The elimination step takes under 0.05 s but is useful: it removed one color on G1 and G3 and matched DSATUR elsewhere.

**Coloring quality.** Batched DSATUR plus elimination is never worse than largest-degree-first greedy plus elimination, and it is better on G3 (10 vs 11). On G2, plain greedy needs 21 colors versus 19. On the tiny test graphs the program found the exact optimum every time. For the random graphs the clique lower bound (3) is weak, since there are no large cliques, so the gap to the bound does not mean the colorings are poor. In our experiments on medium graphs, sequential DSATUR was at most one color better.

**12 vs. 16 threads.** Beyond 12 threads there is no further gain: 16 threads are 4–15% slower than 12 on every graph. The building phase still improves slightly, but the coloring phase gets slower. With all 16 hardware threads busy, any other activity on the machine (OS, desktop) preempts one of our threads, and then all threads wait for it at the next barrier. The coloring phase has thousands of barriers, so it is hit hardest, and its run-to-run variation is largest (G5 at 16 threads: 2.32–3.04 s). For this machine, 8–12 threads is the best choice.


---

## Question 2: Solving Ax = y for the "hourglass" matrix

### 2.1 Structure and sequential algorithm

Row *i* of A is non-zero only between columns min(i, 2n−i) and max(i, 2n−i). So row *n* contains a single entry, and **rows n−k and n+k both span exactly columns n−k … n+k**. Solving from the middle outwards:

* Step 0: x_n = y_n / A[n][n].
* Step k (k = 1..n): after the contributions of the already known inner unknowns x_{n−k+1} … x_{n+k−1} have been subtracted from y_{n−k} and y_{n+k}, only two unknowns remain:

  A[n−k][n−k]·x_{n−k} + A[n−k][n+k]·x_{n+k} = y'_{n−k}
  A[n+k][n−k]·x_{n−k} + A[n+k][n+k]·x_{n+k} = y'_{n+k}

  This 2×2 system is solved directly (Cramer's rule).

This is a block back-substitution. It needs O(n²) operations, which is optimal since A has about 2n² non-zeros that must each be read once. No inverse and no general elimination are needed, and there is no fill-in.

We organize the subtraction in "right-looking" form: as soon as x_{n−k} and x_{n+k} are known, their contribution is subtracted from all outer rows n−p, n+p (p > k).

### 2.2 Data layout ("rings")

Row-wise storage would make this update access two elements per row, scattered over memory. Instead, while reading the input we store A by **rings**: ring *k* holds, for every row pair *p ≥ k*, the four entries A[n−p][n−k], A[n−p][n+k], A[n+p][n−k], A[n+p][n+k]. The update of step *k* is then **one contiguous, unit-stride, vectorizable pass over ring k**, and every non-zero of A is read exactly once during the whole solve. Total storage is 2(n+1)(n+2) doubles, the same as the number of non-zeros.

### 2.3 Parallelization

The steps k = 0..n depend on each other (step k needs the updates of all earlier steps), so the parallelism must come from within each step and from overlapping steps.

**Version 1 (barrier).** One parallel region. In each step, every thread solves the 2×2 system itself (a few flops, which avoids an extra synchronization), then updates its static share of the outer row pairs, followed by a barrier. That is *n* barriers in total.

**Version 2 (pipelined, submitted default).** The row pairs are split into one contiguous block per thread, fixed for the whole solve, so a thread always works on the same part of y and of every ring. Row pair *p* receives *p* updates, so the block boundaries are placed at n·√(t/T) to balance the total work. A thread applies step *k* to its block as soon as x of step *k* is available. There are no barriers. The solved pivots are published in order through a single atomic counter (release/acquire), and threads wait with a short spin (`pause`). For pivots inside its own block, a thread first updates only the *next* pivot row, solves and publishes it immediately (**look-ahead**), and only then updates the rest of its block. Threads that depend on it therefore never wait for its whole block update. Faster threads run ahead in a pipeline instead of meeting at a barrier every step.

Because the pipelined version spin-waits, it falls back to the barrier version when more threads than cores are requested. It also falls back to the serial loop for n < 64 or a single thread. The environment variable `Q2_ALGO=barrier|serial` selects a version for experiments.

#### 2.3.1 Further optimizations (measured)

The solve reads every non-zero of A exactly once, so it is bound by memory bandwidth, and no reorganization can reduce that traffic. The optimizations therefore target what surrounds the data stream. We implemented four, each behind a switch, verified that each gives **bit-identical** x, and timed them against the plain version (median of 5 interleaved runs, n = 1000–8000, 1–16 threads):

| Optimization | 1 thread | 4–16 threads | Decision |
|---|---|---|---|
| **Huge pages for A.** A is aligned to 2 MB and the kernel is asked for transparent huge pages (`madvise`, Linux only; ignored elsewhere). With 4 KB pages, the 1 GB stream of A spans about 250,000 pages, far more than the TLB (address-translation cache) holds; 2 MB pages need 512× fewer translations | −1% … −8% | **−5% … −15%** (n ≥ 2000) | **not used:** it needs `madvise` from `<sys/mman.h>`, a system header outside standard C/C++, so we stayed strictly within the assignment's library rule |
| **Multi-step passes** (single thread). Blocks of 8 steps: the pivots of a block are solved first, then the outer rows receive all 8 steps in one pass, so each y value is loaded and stored once per 8 steps. The operations per row are the same and in the same order, so the result is bit-identical | **−12% … −18%** | +6% … +24% (slower) | **kept for 1 thread only** |
| **Adaptive thread count.** At most one thread per 500 row pairs: with less work per step, synchronization costs more than an extra thread gains, and a few threads already saturate the memory bandwidth | — | n = 1000: **−25% … −35%** at 12–16 threads; n = 2000–4000: −6% … −12% | **kept** |
| **Separate arrays per ring** (4 unit-stride arrays instead of 4 interleaved values, for cleaner SIMD) | −5% … −20% | +9% … +18% (slower) | rejected |

**Why two of them hurt in parallel.** Multi-step passes and separate arrays both make each thread read from several memory streams at once: 4 streams per ring, or 8 rings together. With 8–16 threads that is 32–128 concurrent streams, more than the hardware prefetchers and DRAM can serve efficiently. With one thread there is bandwidth to spare, and the lower overhead wins.

**Note on the thread count.** Because of the adaptive thread count, a request for T threads may use fewer. The program prints "threads = T (used U)", and the tables below are by *requested* thread count (e.g. n = 2000 uses at most 4 threads, n = 4000 at most 8).

**Numerical note.** Tests are generated as the assignment suggests: A and x are generated first, and y = Ax. Off-pivot entries are uniform in [−1, 1]. The diagonal of each 2×2 pivot block has magnitude in [β, 2β] with β = 2√(2n+1) + 4, so the pivot blocks are well conditioned and no element is very small or very large. The maximum error ‖x − x_true‖∞ was below 10⁻¹² for every test (n up to 8000, all thread counts). `InitializeInput(n)` (clearly marked in `q2.cpp`, with its call commented out as required) generates such an instance in memory for fast testing of large n.

### 2.4 Results

Times are in milliseconds (median of 5 runs), measured on instances made by `InitializeInput`. *GB/s* = bytes of A (non-zeros) divided by the solve time of the pipelined version.
#### n = 1,000 (matrix 2,001 x 2,001, 16 MB of non-zeros)

| Threads | Pipelined (ms) | Speedup | GB/s | Barrier version (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.90 | 1.00 | 17.9 | 0.53 | 1.00 |
| 2 | 0.70 | 1.28 | 22.9 | 1.11 | 0.47 |
| 4 | 0.68 | 1.31 | 23.5 | 1.06 | 0.50 |
| 6 | 0.76 | 1.18 | 21.1 | 1.07 | 0.49 |
| 8 | 0.73 | 1.23 | 22.0 | 1.08 | 0.49 |
| 12 | 0.77 | 1.17 | 20.8 | 1.12 | 0.47 |
| 16 | 0.74 | 1.21 | 21.7 | 1.04 | 0.51 |

#### n = 2,000 (matrix 4,001 x 4,001, 64 MB of non-zeros)

| Threads | Pipelined (ms) | Speedup | GB/s | Barrier version (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.44 | 1.00 | 18.6 | 3.17 | 1.00 |
| 2 | 3.40 | 1.01 | 18.8 | 3.95 | 0.80 |
| 4 | 3.26 | 1.06 | 19.7 | 3.59 | 0.88 |
| 6 | 3.17 | 1.09 | 20.2 | 3.64 | 0.87 |
| 8 | 3.45 | 1.00 | 18.6 | 3.60 | 0.88 |
| 12 | 3.17 | 1.08 | 20.2 | 3.55 | 0.89 |
| 16 | 3.50 | 0.98 | 18.3 | 3.47 | 0.91 |

#### n = 4,000 (matrix 8,001 x 8,001, 256 MB of non-zeros)

| Threads | Pipelined (ms) | Speedup | GB/s | Barrier version (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.63 | 1.00 | 26.6 | 9.77 | 1.00 |
| 2 | 10.72 | 0.90 | 23.9 | 10.35 | 0.94 |
| 4 | 9.23 | 1.04 | 27.7 | 9.38 | 1.04 |
| 6 | 8.88 | 1.09 | 28.9 | 9.29 | 1.05 |
| 8 | 9.21 | 1.05 | 27.8 | 9.74 | 1.00 |
| 12 | 9.03 | 1.07 | 28.4 | 9.85 | 0.99 |
| 16 | 9.08 | 1.06 | 28.2 | 9.79 | 1.00 |

#### n = 6,000 (matrix 12,001 x 12,001, 576 MB of non-zeros)

| Threads | Pipelined (ms) | Speedup | GB/s | Barrier version (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 21.07 | 1.00 | 27.4 | 20.70 | 1.00 |
| 2 | 20.59 | 1.02 | 28.0 | 20.25 | 1.02 |
| 4 | 18.35 | 1.15 | 31.4 | 17.89 | 1.16 |
| 6 | 18.27 | 1.15 | 31.5 | 18.10 | 1.14 |
| 8 | 18.49 | 1.14 | 31.2 | 19.56 | 1.06 |
| 12 | 19.87 | 1.06 | 29.0 | 21.56 | 0.96 |
| 16 | 19.94 | 1.06 | 28.9 | 21.55 | 0.96 |

#### n = 8,000 (matrix 16,001 x 16,001, 1,024 MB of non-zeros)

| Threads | Pipelined (ms) | Speedup | GB/s | Barrier version (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.93 | 1.00 | 28.5 | 37.34 | 1.00 |
| 2 | 35.12 | 1.02 | 29.2 | 33.79 | 1.11 |
| 4 | 30.96 | 1.16 | 33.1 | 30.13 | 1.24 |
| 6 | 30.78 | 1.17 | 33.3 | 30.53 | 1.22 |
| 8 | 31.60 | 1.14 | 32.4 | 33.45 | 1.12 |
| 12 | 34.35 | 1.05 | 29.8 | 36.74 | 1.02 |
| 16 | 36.69 | 0.98 | 27.9 | 42.37 | 0.88 |


### 2.5 Discussion

**The solve is memory-bandwidth bound.** Every non-zero of A is read exactly once and used in one multiply-add, which is about 0.25 flop per byte. Thanks to the multi-step passes (2.3.1), **a single thread now streams A at 27–29 GB/s** for n ≥ 4000. Before this optimization it was 19 GB/s: at n = 8000 the single-thread solve went from 52.7 ms to 35.9 ms. More threads raise the rate to at most about **33 GB/s**, reached with 4–6 threads. That is the practical DRAM bandwidth of this laptop for this access pattern (dual-channel DDR4; theoretical peak 51 GB/s).

Once one thread uses almost all of the available bandwidth, extra threads have little left to add. So the speedup is at most **1.15–1.17×** (n ≥ 6000, 4–6 threads), although the absolute times are the best we measured: 30.8 ms at n = 8000, against 35.7 ms before the optimizations. Beyond 6–8 threads, more threads only compete for the same bandwidth and add synchronization, and the time rises again (n = 8000: 36.7 ms at 16 threads). The "GB/s" column shows this directly: it stops growing at the same thread count where the time stops improving. On a server with more memory channels the same code would scale further. Further gains in time would require reading fewer bytes, which is impossible since every non-zero is needed.

**The serial time grows as n²,** as expected: from n = 4000 to 8000 (twice the size) the single-thread time grows 3.7×, from 9.6 ms to 35.9 ms.

**Small n and the adaptive thread count.** For n = 1000, A (16 MB) about fits in the L3 cache and the whole solve takes under 1 ms. There are only about 2000 multiply-adds per step, too little to pay for per-step synchronization among many threads. Before the adaptive thread count, 16 threads made the solve *slower* than 1 thread (1.75 ms vs 1.10 ms). Now the program uses at most one thread per 500 row pairs: 2 threads for n = 1000, 4 for n = 2000, 8 for n = 4000. The time therefore stays flat when more threads are requested (n = 1000: about 0.7 ms from 2 to 16 threads; n = 2000: 3.2–3.5 ms), instead of getting worse. Sub-millisecond timings vary noticeably from run to run.

**Barrier vs. pipelined version.** The two versions are within about 3% of each other up to 6 threads: the barrier version is slightly faster at 4 threads for n ≥ 6000. With more threads, the pipelined version is clearly better (n = 8000, 16 threads: 36.7 ms vs 42.4 ms). The barrier version executes *n* barriers, and each one waits for the slowest thread. In an earlier measurement with other programs running on the laptop, it even collapsed at 16 threads (190 ms vs 46 ms at n = 8000), while the pipelined version degraded gracefully: a delayed thread only delays the threads that need its pivots, and with look-ahead those pivots are published early. The pipelined version is therefore the default: it is as fast or faster in most configurations and robust when the machine is busy.

**Numerical accuracy** is identical for all versions, thread counts and optimizations: the outputs are bit-identical, because every row receives exactly the same floating-point operations in the same order. The maximum error against the known solution was about 10⁻¹⁴ for n = 3000–8000.


---

## Correctness testing (both questions)

* **Q1:** 200 tiny random graphs compared against an independent brute-force minimum (Python), with 0 mismatches. Over 400 random, clustered, and power-law graphs checked with the independent checker at thread counts 1–16, all outputs were valid and byte-identical across thread counts. The sample K₄ gives 5 colors, proven optimal.
* **Q2:** the sample input gives x = (−4.43856, 2.04545, 1.74669), checked by hand. Generated files with n ∈ {2, 5, 63, 64, 100, 1000, 2000} were solved by all three versions (serial, barrier, pipelined) at 1, 2, 3, 7, 16 threads: 105 runs, worst error 5·10⁻¹³.

## References

1. D. Brélaz, "New methods to color the vertices of a graph," *Communications of the ACM* 22(4), 1979. (DSATUR)
2. M. T. Jones and P. E. Plassmann, "A parallel graph coloring heuristic," *SIAM J. Scientific Computing* 14(3), 1993.
3. M. Luby, "A simple parallel algorithm for the maximal independent set problem," *SIAM J. Computing* 15(4), 1986.
4. P. Holme and B. J. Kim, "Growing scale-free networks with tunable clustering," *Physical Review E* 65, 026107, 2002. (test graph generator)
