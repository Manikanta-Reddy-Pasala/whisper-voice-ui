"""Near-live voice-to-text web UI using Whisper (faster-whisper).

Speak into the mic and the transcript updates every ~1s. Whisper
re-transcribes the rolling buffer each tick, so it self-corrects as
more context arrives.

Run:  python app.py   (or ./run.sh)
Then open http://localhost:7860 in your browser.

Mic is captured by the browser, so this works on WSL with no Linux
audio drivers — the OS only runs the web server.
"""

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
    segments, _ = model.transcribe(buf, beam_size=1, language=lang)
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
                ["tiny", "base", "small"],
                value="base",
                label="Model (live needs a fast one; large is too slow on CPU)",
            )
            language = gr.Dropdown(
                ["auto", "en", "hi", "te", "es", "fr", "de", "zh", "ja"],
                value="auto",
                label="Language",
            )
        output = gr.Textbox(label="Live transcript", lines=12, show_copy_button=True)

    # Reset buffer each time a new recording starts.
    audio.start_recording(reset, outputs=[state, output])
    # Fire on every streamed chunk (~1s).
    audio.stream(
        stream,
        inputs=[audio, state, model_size, language],
        outputs=[state, output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
