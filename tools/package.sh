#!/usr/bin/env bash
# Uses Linux zip, but checks that the four packaged files were actually tested.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
version=${1:?Usage: bash tools/package.sh v00}
[[ $version =~ ^v[0-9]+$ ]] || exit 2
run_id=$(cat "logs/$version.latest")
[[ $run_id =~ ^v[0-9]+-[0-9TZ]+-[0-9]+$ ]] || exit 2
bash tools/scores.sh "$version"
sha256sum -c "logs/$run_id/files.sha256"
[[ $(find CONV -mindepth 1 -maxdepth 1 | wc -l) == 4 ]] || { echo 'CONV must contain only the four required files'; exit 2; }
for f in conv2d.c bench_conv.c run.sh conv2d_test; do [[ -f CONV/$f && ! -L CONV/$f ]] || exit 2; done
mkdir -p packages
archive="CONV-$run_id.zip"
[[ ! -e packages/$archive ]] || { echo 'Package already exists; keep it or run a new test.'; exit 2; }
zip -r "packages/$archive" CONV
unzip -t "packages/$archive"
(cd packages && sha256sum "$archive" > "$archive.sha256")
printf '%s\n' "$archive" > "packages/$version.latest"
echo "Package: packages/$archive"
