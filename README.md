# parallel-stuff

Parallel programming projects from CS433 (IIT Kanpur), done in a team of three: Ashik Stenny, Spandan Pati and Vedhanth Balasubramanian. Each assignment follows the same loop: design a parallel algorithm, implement it, verify it carefully, benchmark it over a wide range of parameters, optimise it, and explain the results in a report.

| Folder | Topic | Highlights |
|---|---|---|
| [assignment1-openmp](assignment1-openmp/) | OpenMP on a multicore CPU | Parallel DSATUR graph colouring with triangle constraints (3.6–4.2× at 12 threads, output identical for every thread count); a memory-bound solver for a structured linear system |
| [assignment2-pthreads](assignment2-pthreads/) | Locks and lock-free synchronisation | Lock-free FIFO semaphore built only on `cmpxchg` (5.5–14.5× faster than the alternatives at 15 producers); concurrent graph with 6 locking strategies, benchmarked over 1,296 configurations |
| [assignment3-cuda](assignment3-cuda/) | GPU programming in CUDA | Lock-free GPU hash table with warp-cooperative probing (5.3–8.6× faster than one thread per operation); xᵀAy at 171 GB/s, 97% of the GPU's peak bandwidth |

Every folder has its own README with build and run commands, and a `report/` folder with the full PDF report, which describes the algorithms and explains every result.

## Test machine
All numbers come from one laptop:
- **CPU:** AMD Ryzen 7 5800HS (8 cores, 16 threads, 16 MB L3).
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (16 SMs, 4 GB GDDR6, 176 GB/s peak).
- **Software:** Ubuntu, g++, CUDA 12.4.

## Not in the repository
Input data (graphs, operation lists, matrices) and compiled binaries are left out, because they are large and can be regenerated with the generators in each folder.
