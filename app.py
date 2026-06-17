"""Near-live voice-to-text web UI using Whisper (faster-whisper).

Speak into the mic and the transcript updates every ~1s. Whisper
re-transcribes the rolling buffer each tick, so it self-corrects as
more context arrives.

Run:  python app.py   (or ./run.sh)
Then open http://localhost:7860 in your browser.

Mic is captured by the browser, so this works on WSL with no Linux
audio drivers — the OS only runs the web server.
"""

import os

# HuggingFace's Rust downloaders (hf_xet, hf_transfer) read the *system* CA
# store, which minimal WSL/containers often lack -> "No CA certificates were
# loaded from the system". Disable both so downloads use the Python client,
# and point any TLS client at certifi's bundled certs. Must run before
# huggingface_hub is imported (faster_whisper pulls it in).
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import numpy as np
import gradio as gr
from faster_whisper import WhisperModel

TARGET_SR = 16000  # Whisper expects 16 kHz mono float32

_MODELS = {}


def get_model(size: str) -> WhisperModel:
    if size not in _MODELS:
        _MODELS[size] = WhisperModel(size, device="auto", compute_type="int8")
    return _MODELS[size]


def _to_mono_16k(sr: int, y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    if y.ndim > 1:                       # stereo -> mono
        y = y.mean(axis=1)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1.0:                       # int16 range -> [-1, 1]
        y = y / 32768.0
    if sr != TARGET_SR and y.size:       # linear resample (no scipy dep)
        new_len = int(round(y.shape[0] * TARGET_SR / sr))
        y = np.interp(
            np.linspace(0, 1, new_len, endpoint=False),
            np.linspace(0, 1, y.shape[0], endpoint=False),
            y,
        ).astype(np.float32)
    return y


def stream(new_chunk, state, model_size, language):
    """Append the new mic chunk to the buffer and re-transcribe it."""
    buf = state if state is not None else np.zeros(0, dtype=np.float32)

    if new_chunk is not None:
        sr, y = new_chunk
        buf = np.concatenate([buf, _to_mono_16k(sr, y)])

    if buf.size < TARGET_SR // 2:        # <0.5s: wait for more audio
        return buf, ""

    model = get_model(model_size)
    lang = None if language == "auto" else language
    segments, _ = model.transcribe(
        buf,
        language=lang,
        task="transcribe",          # transcribe in spoken language, never translate
        beam_size=5,                # better accuracy than greedy
        vad_filter=True,            # drop silence -> far less hallucination
        condition_on_previous_text=False,  # re-transcribing a buffer: don't loop on prior text
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return buf, text


def reset():
    """Clear the buffer + transcript when a new recording starts."""
    return np.zeros(0, dtype=np.float32), ""


with gr.Blocks(title="Whisper Live Voice-to-Text") as demo:
    gr.Markdown("# 🎙️ Whisper Live Voice-to-Text\nClick the mic and start talking — the transcript updates as you speak.")

    state = gr.State(None)

    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
            model_size = gr.Dropdown(
                [
                    "tiny", "tiny.en",
                    "base", "base.en",
                    "small", "small.en",
                    "medium", "medium.en",
                    "large-v2", "large-v3",
                    "distil-small.en", "distil-medium.en", "distil-large-v3",
                ],
                value="base",
                label="Model (.en = English-only, faster; distil-* = fast + accurate; medium/large lag live on CPU, want a GPU)",
            )
            language = gr.Dropdown(
                ["auto", "en", "hi", "te", "es", "fr", "de", "zh", "ja"],
                value="auto",
                label="Language",
            )
        output = gr.Textbox(label="Live transcript", lines=12, show_copy_button=True)

    # Clear the buffer + transcript each time a new recording starts.
    audio.start_recording(reset, outputs=[state, output])
    # Re-transcribe the rolling buffer every 0.5s of new audio.
    audio.stream(
        stream,
        inputs=[audio, state, model_size, language],
        outputs=[state, output],
        stream_every=0.5,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
