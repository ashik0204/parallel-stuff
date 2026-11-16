# Assignment 3: GPU programming (CUDA)

The full write-up is in [report/report.pdf](report/report.pdf). The LaTeX sources and the generated tables are in `report/`, and the raw measurements are in `results/`.

## Q1: lock-free concurrent hash table on the GPU (`q1-gpu-hash-table/`)
Open addressing with linear probing and `h(k) = k mod n`, using only `atomicCAS`. Deleted slots become tombstones, which later inserts can reuse.
- **P1 (`p1.cu`):** one thread per operation, with static, cyclic or dynamic work distribution.
- **P2 (`p2.cu`):** a group of up to 32 threads per operation. The group probes 32 slots in one coalesced memory access and coordinates with `__ballot_sync`, `__shfl_sync` and `__ffs`. Operations are assigned to groups statically or dynamically.
- **Verification:** every run checks every return value, the full table contents, and that every key is reachable.

```
make
./gen_ops ops.bin 4194304 80 0 20                 # file, ops, insert/delete/search %
./p1 ops.bin 4194304 4194304 1024 dynamic 1
./p2 ops.bin 4194304 67108864 128 32 dynamic 1
```
`run_q1.sh` runs the full sweep. The shared header `q1_common.h` is used by both programs.

## Q2: computing xᵀAy, optimised in steps (`q2-xTAy/`)
| version | adds |
|---|---|
| `q2_v1_basic` | contiguous chunk per thread |
| `q2_v2_coalesced` | memory access coalescing |
| `q2_v3_tree` | tree reduction |
| `q2_v4_shmem` | shared memory |
| `q2_v5_opt` | float4 loads, instruction-level parallelism, warp shuffle |
| `q2_omp.c` | OpenMP CPU version used for the comparison |

```
make
./q2_v5_opt 16384 16384 65536                     # m, n, threads [, block size]
```
`run_q2.sh` runs the full sweep.

## Tools
- `tools/check_gpu.sh`: a bandwidth check that catches a throttled laptop GPU before benchmarking.
- `tools/devinfo.cu`: prints the GPU properties used in the report.
