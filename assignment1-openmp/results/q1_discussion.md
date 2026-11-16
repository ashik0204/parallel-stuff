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
