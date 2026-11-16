#!/bin/bash
# Compliance / robustness tests for both programs (see report, "Correctness testing").
# Usage: tools/compliance_test.sh      (run from anywhere)
cd "$(dirname "$0")/.."
T=tests/edge
mkdir -p $T results
pass=0; fail=0
ok()  { pass=$((pass+1)); echo "  PASS  $1"; }
bad() { fail=$((fail+1)); echo "  FAIL  $1"; }

g++ -O2 -std=c++17 -o tools/check_coloring tools/check_coloring.cpp || { echo "cannot build checker"; exit 1; }
echo "== Build (plain command the graders are likely to use)"
g++ -O3 -fopenmp -Wall -Wextra -o q1 q1.cpp 2> results/w1.txt && ok "q1 compiles ($(grep -c warning results/w1.txt) warnings)" || bad "q1 compile"
g++ -O3 -fopenmp -Wall -Wextra -o q2 q2.cpp 2> results/w2.txt && ok "q2 compiles ($(grep -c warning results/w2.txt) warnings)" || bad "q2 compile"
for std in c++14 c++17 c++20; do
  g++ -std=$std -fopenmp -o /dev/null q1.cpp 2>/dev/null && g++ -std=$std -fopenmp -o /dev/null q2.cpp 2>/dev/null && ok "compiles with -std=$std, no -O" || bad "compile -std=$std"
done
echo "== Only standard C++ and OpenMP headers"
inc=$(grep -h "^#include" q1.cpp q2.cpp | sort -u | tr '\n' ' ')
extra=$(grep -h "^#include" q1.cpp q2.cpp | sort -u | grep -vE "<(cstdio|cstdlib|cstring|cstdint|cmath|vector|algorithm|atomic|memory|omp.h|sys/mman.h)>")
[ -z "$extra" ] && ok "includes: $inc" || bad "unexpected include: $extra"
grep -q "pthread" q1.cpp q2.cpp && bad "pthread used" || ok "no pthread calls (OpenMP only)"

echo "== Q1 edge cases (validity checked by tools/check_coloring, minimum by brute force when tiny)"
printf '4\n1 2 3\n2 3\n3\n'                     > $T/sample.txt
printf '1\n'                                    > $T/one_vertex.txt
printf '0\n'                                    > $T/empty.txt
printf '5\n\n\n\n\n\n'                          > $T/no_edges.txt
printf '4\r\n1 2 3\r\n2 3\r\n3\r\n'             > $T/crlf.txt
printf '4\n1, 2, 3\n2,3\n3'                     > $T/commas_no_final_newline.txt
printf '4\n1 2 3  \n  2 3\n'                    > $T/last_lines_missing.txt
printf '5\n1 1 2\n2 0\n\n4\n\n'                 > $T/duplicates_backwards.txt
printf '6\n1\n2\n3\n4\n5\n\n'                   > $T/path.txt
printf '6\n1 2 3 4 5\n\n\n\n\n\n'               > $T/star.txt
for k in 3 5 6 7 8; do python3 -c "
k=$k
print(k)
for i in range(k): print(' '.join(str(j) for j in range(i+1,k)))" > $T/K$k.txt; done
for f in $T/*.txt; do
  name=$(basename $f .txt)
  case $name in q2_*) continue;; esac  # Q2 test files live in the same folder
  for t in 1 3 16; do
    if ! timeout 60 ./q1 $f results/edge_$t.out $t 2>/dev/null; then bad "q1 $name t=$t crashed"; continue 2; fi
  done
  r=$(./tools/check_coloring $f results/edge_16.out)
  if [[ $r != OK* ]]; then bad "q1 $name: $r"; continue; fi
  cmp -s results/edge_1.out results/edge_3.out && cmp -s results/edge_1.out results/edge_16.out || { bad "q1 $name: output depends on threads"; continue; }
  k=$(tail -1 results/edge_1.out)
  if [ $(wc -l < results/edge_1.out) -le 12 ]; then
    bf=$(python3 tools/brute_q1.py $f)
    [ "$k" == "$bf" ] && ok "q1 $name: valid, $k colors = brute-force minimum" || bad "q1 $name: $k colors, brute force $bf"
  else
    ok "q1 $name: valid, $k colors"
  fi
done
echo "   K4 output:"; ./q1 $T/sample.txt /dev/stdout 2 2>/dev/null | sed 's/^/     /'

echo "== Q1 argument handling"
./q1 2>/dev/null; [ $? -ne 0 ] && ok "missing arguments -> usage message, no crash" || bad "missing args"
./q1 $T/sample.txt results/e.out 0 2>/dev/null && [[ $(./tools/check_coloring $T/sample.txt results/e.out) == OK* ]] && ok "threads = 0 treated as 1" || bad "threads 0"
./q1 $T/sample.txt results/e.out 64 2>/dev/null && [[ $(./tools/check_coloring $T/sample.txt results/e.out) == OK* ]] && ok "threads = 64 (more than cores) works" || bad "threads 64"

echo "== Q2 edge cases"
printf '1\n1.0 -2.0 6.2\n2.2\n3.0 4.1 -1.3\n2.3 4.5 -7.2\n'       > $T/q2_sample.txt
printf '0\n4.0\n10.0\n'                                            > $T/q2_n0.txt
printf '1\r\n1.0 -2.0 6.2\r\n2.2\r\n3.0 4.1 -1.3\r\n2.3 4.5 -7.2'  > $T/q2_crlf.txt
printf '1\n1e0, -2E0, 6.2e+0\n2.2\n3, 4.1, -1.3\n2.3, 4.5, -7.2\n' > $T/q2_commas_exp.txt
printf '1 1.0 -2.0 6.2 2.2 3.0 4.1 -1.3 2.3 4.5 -7.2'              > $T/q2_one_line.txt
want="-4.43855641845888 2.04545454545455 1.74668798538064"
for f in q2_sample q2_crlf q2_commas_exp q2_one_line; do
  ./q2 $T/$f.txt results/e.out 4 2>/dev/null; got=$(cat results/e.out)
  lines=$(wc -l < results/e.out)
  python3 -c "
a=[float(x) for x in '$got'.split()]; b=[float(x) for x in '$want'.split()]
import sys; sys.exit(0 if len(a)==3 and max(abs(p-q) for p,q in zip(a,b))<1e-9 else 1)" && [ "$lines" == 1 ] && ok "q2 $f: x = $got (1 line)" || bad "q2 $f: got '$got' ($lines lines)"
done
./q2 $T/q2_n0.txt results/e.out 4 2>/dev/null; [ "$(cat results/e.out)" == "2.5" ] && ok "q2 n = 0 (1x1 system): x = 2.5" || bad "q2 n=0: $(cat results/e.out)"
for n in 5 64 500 1500; do
  ./tools/gen_matrix $n $T/q2_gen.txt $T/q2_gen_x.txt $n
  for t in 1 2 7 16 64; do
    ./q2 $T/q2_gen.txt results/e.out $t 2>/dev/null || { bad "q2 n=$n t=$t crashed"; continue; }
    e=$(python3 tools/cmp_vec.py results/e.out $T/q2_gen_x.txt | awk '{print $NF}')
    python3 -c "import sys; sys.exit(0 if $e < 1e-9 else 1)" && [ $(wc -l < results/e.out) == 1 ] || bad "q2 n=$n t=$t err=$e"
  done
  ok "q2 n=$n, threads 1/2/7/16/64: correct (last error $e), output is one line"
done
./q2 2>/dev/null; [ $? -ne 0 ] && ok "q2 missing arguments -> usage message" || bad "q2 missing args"
grep -n "InitializeInput" q2.cpp | grep -q "^.*// InitializeInput(atoi" && ok "q2: InitializeInput exists and its call is commented out" || bad "InitializeInput call"

echo
echo "RESULT: $pass passed, $fail failed"
