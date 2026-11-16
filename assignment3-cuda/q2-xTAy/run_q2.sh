#!/bin/bash
# run_q2.sh -- every Problem 2 experiment.  Run with the charger plugged in.
# Output: ../results/q2_*.csv
set -e
cd "$(dirname "$0")"
make -s
R=../results
mkdir -p $R
M=16384; N=16384          # largest square size that fits (A = 1 GB of 4 GB)
HDR="version,m,n,t,blocks,blocksize,gpu,cpu,absdiff,avg_ms,min_ms,gbps"
run() { local out=$1; shift; "$@" | tee /dev/stderr | grep '^CSV' | sed 's/^CSV,//' >> $out; }

# ---- 1. basic program: thread count x block size ----
F=$R/q2_basic.csv; echo $HDR > $F
for lt in $(seq 5 22); do
	t=$((1 << lt))
	reps=3; [ $t -lt 2048 ] && reps=1
	for bs in 32 64 128 256 512 1024; do
		[ $bs -gt $t ] && continue
		REPS=$reps run $F ./q2_v1_basic $M $N $t $bs
	done
done

# ---- 2. every version: thread count x block size ----
F=$R/q2_versions.csv; echo $HDR > $F
for v in q2_v1_basic q2_v2_coalesced q2_v3_tree q2_v4_shmem q2_v5_opt; do
	for lt in $(seq 10 22); do
		for bs in 32 64 128 256 512 1024; do
			REPS=5 run $F ./$v $M $N $((1 << lt)) $bs
		done
	done
done

# ---- 3. best CUDA vs OpenMP for several (m, n) ----
F=$R/q2_compare.csv; echo $HDR > $F
for mn in "1024 1024" "4096 4096" "8192 8192" "16384 16384" "4096 65536" "65536 4096"; do
	set -- $mn
	for lt in 12 14 16 18 20; do
		for bs in 256 512 1024; do
			REPS=5 run $F ./q2_v5_opt $1 $2 $((1 << lt)) $bs
		done
	done
	for t in 1 2 4 8 16; do
		REPS=5 run $F ./q2_omp $1 $2 $t
	done
done
echo "Problem 2 experiments done."
