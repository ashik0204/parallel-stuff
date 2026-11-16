#!/bin/bash
# run_q1.sh -- every Problem 1 experiment.  Run with the charger plugged in.
# Output: ../results/q1_*.csv   (op files are cached in ../results/q1_ops)
set -e
cd "$(dirname "$0")"
make -s
R=../results
OPS=$R/q1_ops
mkdir -p $OPS

SIZES="1048576 4194304 16777216"                    # 2^20, 2^22, 2^24 operations
RATIOS="100,0,0 50,0,50 50,50,0 80,20,0 80,0,20"
TUNE=4194304                                          # tuning workload: 2^22 ops, 100% insert
HDR="prog,sched,param,threads,blocks,blocksize,nops,ins,del,srch,n,avg_ms,min_ms,verify"

gen() {   # gen <nops> <ins> <del> <srch>  -> prints file name
	local f=$OPS/ops_$1_$2_$3_$4.bin
	[ -f $f ] || ./gen_ops $f $1 $2 $3 $4 >&2
	echo $f
}
run() {   # run <csv> <cmd...>
	local out=$1; shift
	"$@" | tee /dev/stderr | grep '^CSV' | sed 's/^CSV,//' >> $out
}
best() {  # best <csv> [grep-pattern]  -> data line with the lowest avg time (header skipped)
	tail -n +2 $1 | grep -E "${2:-.}" | sort -t, -k12 -g | head -1
}
field() { echo "$1" | cut -d, -f$2; }

TF=$(gen $TUNE 100 0 0)

# ---- P1 static and cyclic assignment: thread count x block size ----
F=$R/q1_p1_tune.csv; echo $HDR > $F
for sched in static cyclic; do
	for t in 65536 262144 1048576 4194304; do
		for bs in 64 128 256 512 1024; do
			REPS=1 run $F ./p1 $TF $TUNE $t $bs $sched
		done
	done
done

# ---- P1 dynamic assignment: chunk size x thread count (block size 256) ----
F=$R/q1_p1_chunk.csv; echo $HDR > $F
for c in 1 2 4 8 16 32 64 128 256; do
	for t in 65536 262144 1048576 4194304; do
		REPS=1 run $F ./p1 $TF $TUNE $t 256 dynamic $c
	done
done
B=$(best $F); BC=$(field "$B" 3); BT=$(field "$B" 4)

# ---- P1 dynamic at the best chunk: thread count x block size ----
F=$R/q1_p1_dyn.csv; echo $HDR > $F
for t in 65536 262144 1048576 4194304; do
	for bs in 64 128 256 512 1024; do
		REPS=3 run $F ./p1 $TF $TUNE $t $bs dynamic $BC
	done
done
P1BS=$(field "$(best $F)" 6)

# ---- P2 group size x thread count (block size 256) ----
F=$R/q1_p2_group.csv; echo $HDR > $F
for g in 1 2 4 8 16 32; do
	for t in 1048576 4194304 16777216 67108864; do
		REPS=1 run $F ./p2 $TF $TUNE $t 256 $g
	done
done
BG=$(field "$(best $F)" 3)

# ---- P2 at the best group size: thread count x block size ----
F=$R/q1_p2_tune.csv; echo $HDR > $F
for t in 4194304 16777216 67108864; do
	for bs in 64 128 256 512 1024; do
		REPS=3 run $F ./p2 $TF $TUNE $t $bs $BG
	done
done
P2BS=$(field "$(best $F)" 6)

echo "best: P1 dynamic chunk=$BC blocksize=$P1BS ; P2 group=$BG blocksize=$P2BS" | tee $R/q1_best.txt

# ---- Final comparison: 5 ratios x 3 sizes, best thread count for each program ----
F=$R/q1_final.csv; echo $HDR > $F
for nops in $SIZES; do
	for r in $RATIOS; do
		IFS=, read pi pd ps <<< "$r"
		OF=$(gen $nops $pi $pd $ps)
		for t in 262144 1048576 4194304 16777216; do
			[ $t -gt $nops ] && continue
			REPS=3 run $F ./p1 $OF $nops $t $P1BS dynamic $BC
		done
		for t in 4194304 16777216 67108864; do
			REPS=3 run $F ./p2 $OF $nops $t $P2BS $BG
		done
	done
done
echo "Problem 1 experiments done."
