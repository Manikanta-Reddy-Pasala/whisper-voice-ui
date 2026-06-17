# Whisper Voice-to-Text

Simple web UI for speech-to-text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Record from your mic or upload an audio file, get a transcript. Runs great on **WSL (Windows)**.

## Why it works on WSL

The UI is a web page. Your **Windows browser** captures the microphone and sends the audio to the server — so you don't need any Linux/WSL audio drivers. WSL just runs Python.

## Setup (WSL Ubuntu)

```bash
# 1. system deps (ffmpeg for audio decoding)
sudo apt update && sudo apt install -y ffmpeg python3-venv

# 2. clone / enter the folder
cd whisper-voice-ui

# 3. virtual env + install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python app.py
```

Open **http://localhost:7860** in your Windows browser. Allow microphone access when prompted.

## Usage

1. Click the mic and record (or upload a `.wav`/`.mp3`/`.m4a`).
2. Pick a model size — `base` is a good default. `small`/`medium` = more accurate, slower.
3. Click **Transcribe**.

First run downloads the model (cached in `~/.cache/huggingface`). No API key, no internet needed after that — fully local.

## Notes

- CPU works fine; `tiny`/`base` are near real-time. Big models want a GPU.
- GPU: if you have CUDA in WSL, it's used automatically (`device="auto"`).
- Mic not showing in browser? `localhost` is treated as secure, so it should work. If using a remote IP, browsers block mic over plain HTTP.
