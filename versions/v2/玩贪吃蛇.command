#!/bin/zsh
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:$PATH"
if [[ -x .play-venv/bin/python && ! -f runs/first/best.zip ]]; then
    .play-venv/bin/python play.py
else
    uv run python play.py
fi
