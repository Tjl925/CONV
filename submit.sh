#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
version=${1:?Usage: bash submit.sh VERSION}
[[ $version =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Use letters, digits, dots, underscores or hyphens for VERSION'; exit 2; }
run_id=$(date -u +%Y%m%dT%H%M%S)-$version-$$
mkdir -p "runs/$run_id"
echo "Logs: runs/$run_id/job.log"
echo 'Ctrl+C stops watching; the scheduler job continues. Do not submit a duplicate.'
dsub -q q_kunpeng -n "conv_$version" --cwd "$PWD" \
    -R 'cpu=608,mem=2GB' -a 'numa[count=16,distribution=pack]' \
    -x job -T 900 -wo -o "$PWD/runs/$run_id/job.log" \
    "bash ./run.sh '$version' '$run_id'"
