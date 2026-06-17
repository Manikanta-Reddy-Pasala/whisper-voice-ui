# Whisper Voice-to-Text

Simple web UI for speech-to-text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Record from your mic or upload an audio file, get a transcript. Runs great on **WSL (Windows)**.

## Why it works on WSL

The UI is a web page. Your **Windows browser** captures the microphone and sends the audio to the server — so you don't need any Linux/WSL audio drivers. WSL just runs Python.

## Quick start (WSL / Linux / macOS)

```bash
cd whisper-voice-ui
./run.sh
```

`run.sh` detects the host (macOS → Homebrew, WSL/Linux → apt), installs ffmpeg, creates a venv, installs Python deps, and launches the UI. First run downloads the `large-v3` model (~3 GB, cached after).

> macOS needs [Homebrew](https://brew.sh) installed first.

Then open **http://localhost:7860** in your Windows browser. Allow microphone access when prompted.

### Manual setup (if you prefer)

```bash
sudo apt install -y ffmpeg python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Usage

1. Click the mic and record (or upload a `.wav`/`.mp3`/`.m4a`).
2. Pick a model size — `base` is a good default. `small`/`medium` = more accurate, slower.
3. Click **Transcribe**.

First run downloads the model (cached in `~/.cache/huggingface`). No API key, no internet needed after that — fully local.

## Notes

- CPU works fine; `tiny`/`base` are near real-time. Big models want a GPU.
- GPU: if you have CUDA in WSL, it's used automatically (`device="auto"`).
- Mic not showing in browser? `localhost` is treated as secure, so it should work. If using a remote IP, browsers block mic over plain HTTP.
