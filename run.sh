#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p build
go build -o build/welvet_sweep_2_snat .
exec ./build/welvet_sweep_2_snat "$@"
