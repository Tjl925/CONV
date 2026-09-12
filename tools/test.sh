#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
if [[ ${1:-} == --worker ]]; then
    run_id=${2:?}
    [[ $run_id =~ ^v[0-9]+-[0-9]{8}T[0-9]{6}-[0-9]+$ ]] || exit 2
    run="$root/logs/$run_id"
    echo RUNNING > "$run/status"
    trap 'rc=$?; echo "$rc" > "$run/exitcode"; if ((rc == 0)); then echo DONE > "$run/status"; else echo FAILED > "$run/status"; fi' EXIT
    env -u OMP_NUM_THREADS numactl -N 1 nproc > "$run/cpu-count.txt"
    [[ $(cat "$run/cpu-count.txt") == 38 ]] || { echo 'NUMA 1 must expose 38 CPUs' >&2; exit 2; }
    cd "$run"
    bash CONV/run.sh 2>&1 | tee run.log
    sha256sum CONV/conv2d.c CONV/bench_conv.c CONV/run.sh CONV/conv2d_test > files.sha256
    # Copy back only the executable, and only when sources have not changed.
    for f in conv2d.c bench_conv.c run.sh; do cmp -s "CONV/$f" "$root/CONV/$f" || { echo 'Source changed during test; not updating executable' >&2; exit 3; }; done
    cp CONV/conv2d_test "$root/CONV/conv2d_test"
    exit 0
fi
version=${1:?Usage: bash tools/test.sh v00}
[[ $version =~ ^v[0-9]+$ ]] || { echo 'Version must look like v00 or v01' >&2; exit 2; }
expected=2548861ae7e29826e454b4c0b098d682f7f04996222e664cd1ea9bc92dd2e927
[[ $(sha256sum CONV/bench_conv.c | cut -d ' ' -f 1) == "$expected" ]] || { echo 'Original bench_conv.c was changed' >&2; exit 2; }
run_id="$version-$(date +%Y%m%dT%H%M%S)-$RANDOM"
run="$root/logs/$run_id"
mkdir -p "$run/CONV"
cp CONV/conv2d.c CONV/bench_conv.c CONV/run.sh "$run/CONV/"
echo "$version" > "$run/version"
echo PENDING > "$run/status"
printf '%s\n' "$run_id" > "logs/$version.latest"
echo "Run: $run_id"
echo 'Ctrl+C only stops watching. Use djob or dkill with the returned job ID.'
dsub -q q_kunpeng -n "conv_$version" --cwd "$root" \
    -R 'cpu=608,mem=2GB' -a 'numa[count=16,distribution=pack]' \
    -x job -T 900 -wo -o "$run/scheduler.log" \
    "bash tools/test.sh --worker '$run_id'"
echo "Next: bash tools/scores.sh $version"
