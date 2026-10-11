#!/bin/zsh
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:$PATH"
uv run python train.py --steps 150000 --out "runs/run-$(date +%Y%m%d-%H%M%S)"
