#!/usr/bin/env bash
# One-shot: install everything and launch the Whisper voice-to-text UI.
# Works on macOS (Homebrew) and WSL/Linux (apt) — detects the host.
# Usage:  ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

OS="$(uname -s)"

install_ffmpeg() {
  command -v ffmpeg >/dev/null 2>&1 && return 0

  case "$OS" in
    Darwin)
      if ! command -v brew >/dev/null 2>&1; then
        echo "!! Homebrew not found. Install it: https://brew.sh" >&2
        exit 1
      fi
      echo ">> Installing ffmpeg via Homebrew..."
      brew install ffmpeg
      ;;
    Linux)
      echo ">> Installing ffmpeg + python venv + CA certs via apt (needs sudo)..."
      sudo apt-get update -y
      sudo apt-get install -y ffmpeg python3-venv ca-certificates
      sudo update-ca-certificates || true
      ;;
    *)
      echo "!! Unsupported OS: $OS. Install ffmpeg + python3 manually." >&2
      exit 1
      ;;
  esac
}

install_ffmpeg

# Virtual env + Python deps.
if [ ! -d .venv ]; then
  echo ">> Creating virtual env..."
  python3 -m venv .venv
fi
source .venv/bin/activate

echo ">> Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Launch. First run downloads the model (base ~140 MB, cached after).
echo ">> Starting UI at http://localhost:7860"
python app.py
