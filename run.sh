#!/usr/bin/env bash
# Run on an allocated compute node; use submit.sh from a login node.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
version=${1:?Usage: bash run.sh VERSION [RUN_ID]}
[[ $version =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid version name'; exit 2; }
case $(hostname -s) in login*) echo 'Use bash submit.sh VERSION on a login node.'; exit 2;; esac
run_id=${2:-$(date -u +%Y%m%dT%H%M%S)-$version-$$}
[[ $run_id =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
run_dir="$PWD/runs/$run_id"
mkdir -p "$run_dir"
[[ ! -e "$run_dir/started" ]] || { echo 'Run already started; use a new RUN_ID'; exit 2; }
touch "$run_dir/started"
unset OMP_PLACES GOMP_CPU_AFFINITY OMP_DYNAMIC OMP_THREAD_LIMIT
export OMP_NUM_THREADS=38 OMP_PROC_BIND=true
echo "VERSION=$version RUN_ID=$run_id"
hostname > "$run_dir/environment.log"
date -Is >> "$run_dir/environment.log"
gcc --version >> "$run_dir/environment.log"
grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status >> "$run_dir/environment.log"
# nproc can honour OMP_NUM_THREADS; remove it for the CPU availability check.
available=$(env -u OMP_NUM_THREADS numactl -N 1 nproc)
[[ $available == 38 ]] || { echo "Expected 38 available CPUs on NUMA 1; got $available"; exit 2; }
numactl -N 1 sh -c 'grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/self/status' >> "$run_dir/environment.log"
expected=2548861ae7e29826e454b4c0b098d682f7f04996222e664cd1ea9bc92dd2e927
actual=$(sha256sum bench_conv.c | cut -d ' ' -f 1)
[[ $actual == "$expected" ]] || { echo 'bench_conv.c differs from the original; stopping.'; exit 2; }
cp conv2d.c bench_conv.c "$run_dir/"
sha256sum "$run_dir/conv2d.c" "$run_dir/bench_conv.c" > "$run_dir/source.sha256"
echo 'gcc -O3 bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp' > "$run_dir/build.log"
gcc -O3 "$run_dir/bench_conv.c" "$run_dir/conv2d.c" -o "$run_dir/conv2d_test" -lm -fopenmp >> "$run_dir/build.log" 2>&1 || { cat "$run_dir/build.log"; exit 3; }
case_no=0
for spec in '4096 6144 39 39 1' '6144 4096 41 41 1' '4256 6390 55 55 1' '6390 4256 81 81 1'; do
    case_no=$((case_no + 1))
    read -r h w kh kw repeats <<< "$spec"
    echo "START case$case_no: $spec"
    # The original benchmark validates BEFORE warming up and timing.
    numactl -N 1 "$run_dir/conv2d_test" "$h" "$w" "$kh" "$kw" "$repeats" 2>&1 | tee "$run_dir/case$case_no.log"
    python3 workflow.py check "$case_no" "$run_dir/case$case_no.log"
done
python3 workflow.py record "$version" "$run_dir"
echo "ALL FOUR CASES PASS. Logs: $run_dir"
