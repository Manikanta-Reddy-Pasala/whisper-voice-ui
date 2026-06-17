"""Simple voice-to-text web UI using Whisper (faster-whisper).

Run:  python app.py
Then open http://localhost:7860 in your Windows browser.

The browser captures the microphone, so this works on WSL without any
Linux audio drivers — WSL only runs the web server.
"""

import gradio as gr
from faster_whisper import WhisperModel

# Cache loaded models so switching size is fast on repeat use.
_MODELS = {}


def get_model(size: str) -> WhisperModel:
    if size not in _MODELS:
        # int8 on CPU = small + fast. Auto-uses GPU if CUDA is available.
        _MODELS[size] = WhisperModel(size, device="auto", compute_type="int8")
    return _MODELS[size]


def transcribe(audio_path: str, model_size: str, language: str):
    if not audio_path:
        return "No audio. Record with the mic or upload a file first."

    model = get_model(model_size)
    lang = None if language == "auto" else language
    segments, info = model.transcribe(audio_path, language=lang, beam_size=5)

    text = " ".join(seg.text.strip() for seg in segments).strip()
    header = f"[detected: {info.language} ({info.language_probability:.0%})]\n\n"
    return header + (text or "(no speech detected)")


with gr.Blocks(title="Whisper Voice-to-Text") as demo:
    gr.Markdown("# 🎙️ Whisper Voice-to-Text\nRecord or upload audio, then click **Transcribe**.")

    with gr.Row():
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
        with gr.Column():
            model_size = gr.Dropdown(
                ["tiny", "base", "small", "medium", "large-v3"],
                value="base",
                label="Model (bigger = more accurate, slower)",
            )
            language = gr.Dropdown(
                ["auto", "en", "hi", "te", "es", "fr", "de", "zh", "ja"],
                value="auto",
                label="Language",
            )
            btn = gr.Button("Transcribe", variant="primary")

    output = gr.Textbox(label="Transcript", lines=10, show_copy_button=True)

    btn.click(transcribe, inputs=[audio, model_size, language], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
