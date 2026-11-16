#!/bin/bash
# Coloring-phase time of DSATUR variants, interleaved run by run.
cd "$(dirname "$0")/.."
out=results/ds_times.csv; echo "graph,threads,variant,run,color_s,total_s" > $out
for g in g1_random_sparse g2_random_dense g3_clustered g4_powerlaw g5_random_large; do
  for t in 1 8 12; do
    for r in 1 2 3; do
      for v in base A B AB; do
        if [ $v == base ]; then log=$(./q1 graphs/$g.txt /dev/null $t 2>&1); else log=$(Q1_DS=$v ./results/q1_ds graphs/$g.txt /dev/null $t 2>&1); fi
        c=$(echo "$log" | sed -n 's/.*coloring \([0-9.]*\) s.*/\1/p'); tt=$(echo "$log" | sed -n 's/.*total \([0-9.]*\) s.*/\1/p')
        echo "$g,$t,$v,$r,$c,$tt" >> $out
      done
    done
  done
done
echo ds done
