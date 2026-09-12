#!/usr/bin/env bash
# Judge entry point: no arguments or external workflow tools required.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
unset OMP_PLACES GOMP_CPU_AFFINITY OMP_DYNAMIC OMP_THREAD_LIMIT
export OMP_NUM_THREADS=38 OMP_PROC_BIND=true
gcc -O3 -fno-tree-vectorize bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp
output=$(mktemp)
trap 'rm -f -- "$output"' EXIT
for case_args in '4096 6144 39 39 1' '6144 4096 41 41 1' '4256 6390 55 55 1' '6390 4256 81 81 1'; do
    read -r h w kh kw repeats <<< "$case_args"
    # Original benchmark checks correctness before warming up and timing.
    numactl -N 1 ./conv2d_test "$h" "$w" "$kh" "$kw" "$repeats" | tee "$output"
    # Its exit code alone does not distinguish PASS from FAIL.
    if grep -q 'FAIL' "$output" || ! grep -Eq '[[:space:]]PASS[[:space:]]*$' "$output"; then
        echo 'Correctness check failed or no PASS result; stopping.' >&2
        exit 1
    fi
done
