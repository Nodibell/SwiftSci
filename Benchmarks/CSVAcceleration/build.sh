#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
build_dir=${1:-${TMPDIR:-/tmp}/swiftsci-csv-acceleration}
mkdir -p "$build_dir"
# Freeze the parser used for the recorded measurements without maintaining a copy.
baseline=dc31b5f1afc7f92810cfd472f7269f0d05533438
git -C "$repo" show "$baseline:Sources/SwiftDataFrame/IO/SystemsCSVParser.swift" > "$build_dir/SystemsCSVParser.swift"
xcrun clang -O3 -c "$here/classify.c" -o "$build_dir/classify.o"
xcrun swiftc -O -whole-module-optimization "$build_dir/SystemsCSVParser.swift" "$here/main.swift" "$build_dir/classify.o" -o "$build_dir/csv-gpu"
printf '%s\n' "$build_dir/csv-gpu"
