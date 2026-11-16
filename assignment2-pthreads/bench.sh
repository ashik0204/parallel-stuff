#!/bin/bash
# Runs the performance sweeps for both questions and writes CSV files to results/.
# Usage: ./bench.sh [q1|q2|all]   (default: all)
# Each configuration is run REPS times; the plots use the median.
set -e
cd "$(dirname "$0")"
make -s all
mkdir -p results ops

REPS=${REPS:-3}
Q1_LIMIT=${Q1_LIMIT:-120}
WHAT=${1:-all}

run_q1() {
	out=results/q1.csv
	echo "version,N,P,n,rep,time_us" > $out
	# Sweep 1: vary N and P with the total number of values nP fixed at ~4M.
	TOTAL=4000000
	for N in 8 64 1024; do
		for P in 1 2 4 8 12 15 23 31; do
			n=$((TOTAL / P))
			# More threads than hardware threads (P+1 > 16): busy-waiting can get
			# extremely slow, so run once with a time limit; -1 means it timed out.
			reps=$REPS; [ $P -ge 16 ] && reps=1
			for v in 1 2 3 4; do
				for r in $(seq $reps); do
					t=$(timeout $Q1_LIMIT ./q1_p$v $N $P $n 2>&1 >/dev/null | sed -n 's/.*check=OK Time: \([0-9]*\).*/\1/p' || true)
					[ -z "$t" ] && t=-1
					echo "$v,$N,$P,$n,$r,$t" >> $out
				done
			done
			echo "q1 N=$N P=$P done" >&2
		done
	done
	# Sweep 2: vary n at fixed N=64, P=8.
	for n in 125000 250000 500000 1000000; do
		for v in 1 2 3 4; do
			for r in $(seq $REPS); do
				t=$(./q1_p$v 64 8 $n 2>&1 >/dev/null | sed -n 's/.*check=OK Time: \([0-9]*\).*/\1/p')
				[ -z "$t" ] && { echo "Q1 P$v n=$n FAILED" >&2; exit 1; }
				echo "$v,64,8,$n,$r,$t" >> $out
			done
		done
	done
}

run_q2() {
	out=results/q2.csv
	echo "version,ratio,ops,threads,rep,time_us" > $out
	for ratio in "80 10 10" "30 30 40" "10 10 80"; do
		set -- $ratio
		tag="$1_$2_$3"
		for ops in 1000000 2000000 4000000; do
			f=ops/ops_${tag}_${ops}.txt
			[ -f $f ] || ./gen_graph_ops $ops $1 $2 10000 $f
			for T in 1 2 4 8 12 16 24 32; do
				for v in 1 2 3 4 5 6; do
					for r in $(seq $REPS); do
						res=$(./q2_p$v $f $ops $T || true)
						t=$(echo "$res" | grep -q "check=OK" && echo "$res" | sed -n 's/^Time: \([0-9]*\).*/\1/p' || true)
						[ -z "$t" ] && { echo "Q2 P$v $tag ops=$ops T=$T FAILED" >&2; exit 1; }
						echo "$v,$tag,$ops,$T,$r,$t" >> $out
					done
				done
				echo "q2 $tag ops=$ops T=$T done" >&2
			done
		done
	done
}

case $WHAT in
	q1) run_q1 ;;
	q2) run_q2 ;;
	all) run_q1; run_q2 ;;
esac
