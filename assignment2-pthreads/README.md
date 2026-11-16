# Assignment 2: POSIX threads, locks and lock-free synchronisation (C/C++)

The full write-up is in [report/report.pdf](report/report.pdf).

## Q1: lock-free FIFO semaphore and a multi-producer single-consumer queue (`q1_mpsc.cpp`)
A ticket semaphore built only on `cmpxchg` that guarantees FIFO order, used in a lock-free multi-producer single-consumer ring buffer. It is compared with three other semaphores:

| version | semaphore |
|---|---|
| P1 | the lock-free semaphore |
| P2 | a busy-waiting spin-lock semaphore |
| P3 | the Linux `sem_t` |
| P4 | a condition-variable semaphore |

`tests/q1_fifo_test.cpp` checks the FIFO order and fairness directly.

## Q2: a concurrent directed graph as an array of linked lists (`q2_graph.cpp`)
Six versions:

| version | synchronisation |
|---|---|
| P1 | one global mutex |
| P2 | one mutex per list |
| P3 | one mutex per node (hand-over-hand) |
| P4 | one read/write lock per node |
| P5 | one `cmpxchg` spin lock per node |
| P6 | optimistic locking with lock-free search |

`gen_graph_ops.cpp` generates the operation lists.

## Build and run
```
make                         # builds q1_p1..q1_p4, q2_p1..q2_p6 and gen_graph_ops
./q1_p1 <N> <P> <n>          # buffer size, producers, values per producer
./gen_graph_ops 4000000 30 30 10000 ops.txt
./q2_p6 ops.txt 4000000 16   # op file, number of operations, threads
```
`bench.sh` runs the sweeps, and `plots.py` makes the figures from `results/`.
