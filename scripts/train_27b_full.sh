#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
torchrun --standalone --nnodes 1 --nproc_per_node 8 -m sii_decision.train \
  --config "$root/configs/27B/full.json" "$@"
