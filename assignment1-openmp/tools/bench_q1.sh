#!/bin/bash
# Q1 timing sweep: median of 3 runs of the measured time (build H .. end).
cd "$(dirname "$0")/.."
out=results/q1_times.csv
echo "graph,threads,median_total_s,build_s,color_s,improve_s,colors,all_totals" > $out
for g in g1_random_sparse g2_random_dense g3_clustered g4_powerlaw g5_random_large; do
  for t in 1 2 4 6 8 12 16; do
    tot=(); line=""
    for r in 1 2 3; do
      log=$(./q1 graphs/$g.txt /dev/null $t 2>&1)
      tot+=($(echo "$log" | sed -n 's/.*total \([0-9.]*\) s.*/\1/p'))
      line+="$log"$'\n'
    done
    med=$(printf "%s\n" "${tot[@]}" | sort -g | sed -n 2p)
    # phase times of the median run
    ml=$(echo "$line" | grep "total $med s" | head -1)
    b=$(echo "$ml" | sed -n 's/.*build H \([0-9.]*\) s.*/\1/p'); c=$(echo "$ml" | sed -n 's/.*coloring \([0-9.]*\) s.*/\1/p'); i=$(echo "$ml" | sed -n 's/.*improve \([0-9.]*\) s.*/\1/p')
    k=$(echo "$line" | sed -n 's/.*final = \([0-9]*\).*/\1/p' | head -1)
    echo "$g,$t,$med,$b,$c,$i,$k,${tot[*]}" >> $out
  done
  ./q1 graphs/$g.txt /dev/null 1 2>&1 | head -3 > results/q1_info_$g.txt
  Q1_INIT=jp ./q1 graphs/$g.txt /dev/null 8 2>&1 | head -3 > results/q1_jp_$g.txt
done
echo q1 done
