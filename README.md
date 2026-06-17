# Whisper Live Voice-to-Text

Simple web UI for **near-live** speech-to-text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Click the mic, start talking, and the transcript updates as you speak. Runs on **WSL (Windows)**, Linux, and macOS.

## Why it works on WSL

The UI is a web page. Your **Windows browser** captures the microphone and sends the audio to the server — so you don't need any Linux/WSL audio drivers. WSL just runs Python.

## Quick start (WSL / Linux / macOS)

```bash
cd whisper-voice-ui
./run.sh
```

`run.sh` detects the host (macOS → Homebrew, WSL/Linux → apt), installs ffmpeg, creates a venv, installs Python deps, and launches the UI. First run downloads the `base` model (~140 MB, cached after).

> macOS needs [Homebrew](https://brew.sh) installed first.

Then open **http://localhost:7860** in your browser. Allow microphone access when prompted.

### Manual setup (if you prefer)

```bash
sudo apt install -y ffmpeg python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Usage

1. Click the mic and start speaking.
2. The transcript updates roughly every second and self-corrects as more context arrives.
3. Pick a model: `tiny` = fastest, `base` = good default, `small` = more accurate but more lag.

First run downloads the model (cached in `~/.cache/huggingface`). No API key, no internet needed after that — fully local.

## Notes

- **Live = small models only.** Whisper re-transcribes the rolling buffer each tick, so `large` can't keep up on CPU. Use `tiny`/`base`/`small`.
- GPU: if you have CUDA, it's used automatically (`device="auto"`) and you can run bigger models live.
- Long sessions: the buffer grows, so each pass gets slower. Best for dictation-length clips; restart the recording to clear it.
- Mic not showing in browser? `localhost` is treated as secure, so it works. Over a remote IP, browsers block mic on plain HTTP.
