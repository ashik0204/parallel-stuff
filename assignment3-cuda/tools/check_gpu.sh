#!/bin/bash
# Sanity check before benchmarking: the bandwidth-bound x^T A y kernel must
# reach > 100 GB/s, otherwise the GPU is throttled (e.g. stuck in a low-power
# state after a suspend/resume, or running on battery) and results are invalid.
cd "$(dirname "$0")/../q2-xTAy" && make -s q2_v5_opt
bw=$(REPS=5 ./q2_v5_opt 16384 16384 65536 | grep -o 'bandwidth=[0-9.]*' | cut -d= -f2)
echo "GPU check: $bw GB/s"
awk -v b="$bw" -v t="${1:-100}" 'BEGIN { exit !(b > t) }' || { echo "GPU looks throttled -- not benchmarking"; exit 1; }
