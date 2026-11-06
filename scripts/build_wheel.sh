#!/usr/bin/env bash
# SPDX-License-Identifier: MIT-0
# Build the perception-free openmorph wheel into dist/ (manylinux2014: needs `pip install maturin ziglang`).
set -euo pipefail
cd "$(dirname "$0")/../crates/openmorph-py"
maturin build --release --zig --compatibility manylinux2014 -o ../../dist "$@"
