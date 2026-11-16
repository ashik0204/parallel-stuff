#!/bin/bash
# Q2 optimization experiment: variants interleaved run by run, 5 runs each.
cd "$(dirname "$0")/.."
out=results/q2opt_times.csv; echo "n,threads,variant,run,solve_s" > $out
run() { # variant n t
  case $1 in
    base) e="";; h) e="Q2_OPTS=h";; s) e="Q2_OPTS=s";; m4) e="Q2_OPTS=m Q2_B=4";; m8) e="Q2_OPTS=m Q2_B=8";;
    t250) e="Q2_MINROWS=250";; t500) e="Q2_MINROWS=500";; ht500) e="Q2_OPTS=h Q2_MINROWS=500";;
  esac
  env $e ./results/q2_opt_bench $2 /dev/null $3 2>&1 | sed -n 's/.*solve time = \([0-9.]*\) s.*/\1/p'
}
for n in 1000 2000 4000 8000; do for t in 1 4 8 12 16; do for r in 1 2 3 4 5; do
  for v in base h s m4 m8 t250 t500 ht500; do echo "$n,$t,$v,$r,$(run $v $n $t)" >> $out; done
done; done; done
echo q2opt done
