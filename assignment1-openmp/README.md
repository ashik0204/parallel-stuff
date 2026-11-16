# Assignment 1: OpenMP (C++)

The full write-up is in [report/report.pdf](report/report.pdf) (Markdown version: [report/report.md](report/report.md)).

## Q1: colouring vertices and edges with triangle constraints (`q1.cpp`)
Two objects must get different colours when they are the two endpoints of an edge, an edge and one of its endpoints, or two edges of the same triangle. The problem is turned into vertex colouring of a "conflict graph", which is then:
- built in parallel (merge-based triangle listing, with an adaptive "each triangle once" mode for dense graphs);
- coloured with a batched, deterministic parallel DSATUR;
- improved by a parallel local search that removes whole colour classes;
- checked against a clique lower bound, with an exact branch and bound for small inputs.

```
g++ -O3 -fopenmp q1.cpp -o q1
./q1 <input> <output> <threads>
```

## Q2: solving Ax = y for the "hourglass" matrix (`q2.cpp`)
An O(n²) block back-substitution from the middle of the matrix outwards. A ring-based data layout gives unit-stride streaming, and there are two parallel versions (barrier-based and pipelined).

```
g++ -O3 -fopenmp q2.cpp -o q2
./q2 <input> <output> <threads>
```

## Tools and results
- `tools/`: test generators (`gen_graph.cpp`, `gen_matrix.cpp`), an independent colouring checker (`check_coloring.cpp`), a brute-force reference (`brute_q1.py`), the benchmark scripts (`bench_*.sh`) and the table and report builders.
- `results/`: timing CSVs, colouring-quality data and the generated tables.

The benchmark scripts expect test graphs in `graphs/`, created with `tools/gen_graph` (see the source for its parameters). They aren't included here because of their size.
