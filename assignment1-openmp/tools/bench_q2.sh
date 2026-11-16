#!/bin/bash
# Q2 timing sweep: median of 5 runs for each (n, threads, version).
cd "$(dirname "$0")/.."
out=results/q2_times.csv
echo "algo,n,threads,median_s,all_s" > $out
for n in 1000 2000 4000 6000 8000; do
  for algo in pipe barrier; do
    for t in 1 2 4 6 8 12 16; do
      ts=()
      for r in 1 2 3 4 5; do
        ts+=($(Q2_ALGO=$algo ./tools/q2_bench $n /dev/null $t 2>&1 | sed -n 's/.*solve time = \([0-9.]*\) s.*/\1/p'))
      done
      med=$(printf "%s\n" "${ts[@]}" | sort -g | sed -n 3p)
      echo "$algo,$n,$t,$med,${ts[*]}" >> $out
    done
  done
done
echo done
