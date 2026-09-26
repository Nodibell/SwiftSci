#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [[ $# -eq 0 ]]; then
    printf 'Usage: %s input.csv [another.csv ...]\n' "$0" >&2
    exit 2
fi
binary=$("$here/build.sh")
"$binary" --check
for csv in "$@"; do
    "$binary" "$csv"
done
