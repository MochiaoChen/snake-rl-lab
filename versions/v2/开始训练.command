#!/bin/zsh
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:$PATH"
uv run python train_v2.py --steps 500000 --out "runs/v2-$(date +%Y%m%d-%H%M%S)"
