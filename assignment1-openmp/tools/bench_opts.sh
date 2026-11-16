#!/bin/bash
# Times each build optimization separately (experiment for the report).
cd "$(dirname "$0")/.."
out=results/opt_times.csv
echo "variant,graph,threads,median_total,build_of_median,relabel_of_median,colors" > $out
for v in base 3 2 1 4b 4d 123 1234b 1234d; do
  for g in g1_random_sparse g2_random_dense g3_clustered g4_powerlaw g5_random_large; do
    for t in 1 12; do
      runs=""
      for r in 1 2 3; do
        if [ $v == base ]; then log=$(./q1 graphs/$g.txt /dev/null $t 2>&1)
        else log=$(Q1_OPTS=$v ./results/q1_opt graphs/$g.txt /dev/null $t 2>&1); fi
        tot=$(echo "$log" | sed -n 's/.*total \([0-9.]*\) s.*/\1/p')
        b=$(echo "$log" | sed -n 's/.*build H \([0-9.]*\) s.*/\1/p')
        rl=$(echo "$log" | sed -n 's/.*relabel \([0-9.]*\) s.*/\1/p'); rl=${rl:-0}
        k=$(echo "$log" | sed -n 's/.*final = \([0-9]*\).*/\1/p')
        runs+="$tot $b $rl $k"$'\n'
      done
      med=$(echo "$runs" | grep . | sort -g | sed -n 2p)
      set -- $med
      echo "$v,$g,$t,$1,$2,$3,$4" >> $out
    done
  done
done
echo opts done
