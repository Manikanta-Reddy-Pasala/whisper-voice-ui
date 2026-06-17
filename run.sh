#!/usr/bin/env bash
# One-shot: install everything and launch the Whisper voice-to-text UI.
# Usage (WSL Ubuntu):  ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

# 1. System deps (ffmpeg for audio decode). Skip if already present.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo ">> Installing ffmpeg + python venv (needs sudo)..."
  sudo apt-get update -y
  sudo apt-get install -y ffmpeg python3-venv
fi

# 2. Virtual env + Python deps.
if [ ! -d .venv ]; then
  echo ">> Creating virtual env..."
  python3 -m venv .venv
fi
source .venv/bin/activate

echo ">> Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. Launch. First run downloads large-v3 (~3 GB, cached after).
echo ">> Starting UI at http://localhost:7860 (open in your Windows browser)"
python app.py
