"""Gradio web demo for VoxSplit (Phase 6).

Upload a recording with several overlapping speakers and get, per detected
speaker, a playable track, an after-separation spectrogram, and a
Whisper transcript, plus the input spectrogram and a speaking timeline. The
speaker count is discovered automatically (no count supplied).

  python demo/app.py                 # loads models, serves on localhost:7860
  python demo/app.py --share         # public Gradio link

Models load once at startup (~30 s). Uses the robust OR-PIT checkpoint by
default; override with --orpit-ckpt / --clf-ckpt.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "tts"))
from pipeline import Pipeline, DEFAULT_ORPIT, DEFAULT_CLF  # noqa: E402
from voice_clone import _tts_env_available  # noqa: E402

MAX_SPK = 8  # UI rows and the recursion cap; >5 is out of the trained range.

_PIPELINE = {"obj": None, "kwargs": {}}


def get_pipeline() -> Pipeline:
    if _PIPELINE["obj"] is None:
        print("Loading models (first request) ...")
        _PIPELINE["obj"] = Pipeline(**_PIPELINE["kwargs"])
    return _PIPELINE["obj"]


LANG_CHOICES = [
    ("English", "en"), ("Spanish", "es"), ("French", "fr"), ("German", "de"),
    ("Italian", "it"), ("Portuguese", "pt"), ("Russian", "ru"),
    ("Chinese", "zh-cn"), ("Hindi", "hi"), ("Japanese", "ja"),
    ("Korean", "ko"), ("Arabic", "ar"),
]


def _fan_out(out, label="Speaker"):
    """Turn a pipeline result dict into the fixed UI output slots.

    The last slot repopulates the voice-clone speaker picker with this run's
    speakers, so cloning works off whichever tab was used last.
    """
    sr, n = out["sr"], out["num_speakers"]
    src = out.get("face_source")
    status = (f"Detected **{n}** {label.lower()}(s) in {out['duration']:.1f}s"
              + (f" (faces via {src})" if src else
                 f" (input {out['orig_sr']} Hz)") + ".")
    results = [status, out["input_fig"], out["timeline_fig"]]
    for i in range(MAX_SPK):
        if i < n:
            results += [
                gr.update(visible=True),
                gr.update(value=(sr, out["tracks"][i]), visible=True,
                          label=f"{label} {i + 1}"),
                gr.update(value=out["speaker_figs"][i], visible=True),
                gr.update(value=out["transcripts"][i] or "(no speech detected)",
                          visible=True)]
        else:
            results += [gr.update(visible=False)] * 4

    choices = [f"{label} {i + 1}" for i in range(n)]
    results.append(gr.update(choices=choices,
                             value=choices[0] if choices else None,
                             interactive=bool(choices)))
    return results


def clone_voice_fn(text, speaker_label, language_label):
    """Speak `text` in a separated speaker's voice (XTTS v2 zero-shot)."""
    if not text or not text.strip():
        raise gr.Error("Enter some text to synthesise.")
    if not speaker_label:
        raise gr.Error("No speakers yet — run a separation first.")
    try:
        speaker_idx = int(str(speaker_label).split()[-1]) - 1
    except (ValueError, IndexError):
        speaker_idx = 0
    language = dict(LANG_CHOICES).get(language_label, "en")
    try:
        wav, sr = get_pipeline().clone_voice(text.strip(), speaker_idx,
                                             language)
    except (RuntimeError, ValueError, ImportError) as exc:
        raise gr.Error(str(exc))
    except Exception as exc:
        raise gr.Error(f"TTS failed: {exc}")
    return gr.update(value=(sr, wav), visible=True)


def separate(audio_path, sensitivity, count_choice):
    """Audio tab: separate an uploaded recording (auto or forced count)."""
    if not audio_path:
        raise gr.Error("Please upload an audio file first.")
    recursion_threshold = float(np.clip(1.0 - sensitivity, 0.2, 0.7))
    force_count = None if count_choice == "Auto" else int(count_choice)
    out = get_pipeline().process(audio_path,
                                 recursion_threshold=recursion_threshold,
                                 force_count=force_count)
    return _fan_out(out)


def separate_video(video_path, num_faces):
    """Video tab: audio-visual separation — video sets count, lips assign."""
    if not video_path:
        raise gr.Error("Please upload a video file first.")
    try:
        out = get_pipeline().process_video(video_path, int(num_faces))
    except ValueError as exc:
        raise gr.Error(str(exc))
    return _fan_out(out, label="On-screen speaker")


def _output_block():
    """Build the shared status + spectrograms + per-speaker rows; return slots."""
    status = gr.Markdown()
    with gr.Row():
        input_spec = gr.Plot(label="Input spectrogram")
        timeline = gr.Plot(label="Speaking timeline")
    outputs = [status, input_spec, timeline]
    for i in range(MAX_SPK):
        with gr.Group(visible=False) as row:
            gr.Markdown(f"### Speaker {i + 1}")
            a = gr.Audio(label=f"Speaker {i + 1}")
            with gr.Row():
                s = gr.Plot(label="Spectrogram")
                t = gr.Textbox(label="Transcript", lines=3)
        outputs += [row, a, s, t]
    return outputs


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VoxSplit") as demo:
        gr.Markdown("# VoxSplit — multi-speaker speech separation\n"
                    "Separate a recording where several people talk at once "
                    "into one clean track per speaker, with spectrograms, "
                    "transcripts, and a speaking timeline.")
        with gr.Tabs():
            with gr.Tab("Audio"):
                with gr.Row():
                    inp = gr.Audio(type="filepath", label="Input recording")
                    with gr.Column():
                        count_choice = gr.Dropdown(
                            choices=["Auto"] + [str(i)
                                                for i in range(1, MAX_SPK + 1)],
                            value="Auto", label="Speaker count",
                            info="Auto detects the count; a number forces it.")
                        sensitivity = gr.Slider(
                            0.3, 0.8, value=0.5, step=0.05,
                            label="Split sensitivity (Auto mode only)",
                            info="Raise if voices merge; lower if one splits.")
                        btn = gr.Button("Separate", variant="primary")
                out_a = _output_block()

            with gr.Tab("Video (audio-visual)"):
                gr.Markdown("Upload a video of people talking. The on-screen "
                            "faces set the speaker count and each track is "
                            "matched to its speaker by lip motion. Speakers "
                            "should be in distinct left-to-right regions.")
                with gr.Row():
                    vinp = gr.Video(label="Input video")
                    with gr.Column():
                        nfaces = gr.Number(
                            value=2, precision=0, label="Number of faces",
                            info="How many people are on screen (0 = try to "
                                 "auto-detect with mediapipe).")
                        vbtn = gr.Button("Separate (AV)", variant="primary")
                out_v = _output_block()

        # Voice cloning works off the most recent separation from EITHER tab:
        # the separated track itself is the voice reference. The controls are
        # only interactive when coqui-tts is importable, so an env without it
        # shows an explanation instead of throwing on every click.
        # Ready if coqui-tts is importable here, OR the isolated voxsplit-tts
        # env exists (we shell out to it — it can't share this env's deps).
        tts_ready = (importlib.util.find_spec("TTS") is not None
                     or _tts_env_available())
        title = ("Voice Clone TTS (XTTS v2)" if tts_ready else
                 "Voice Clone TTS (XTTS v2) — not installed")
        with gr.Accordion(title, open=False):
            if tts_ready:
                gr.Markdown("Type any text and hear it in a separated "
                            "speaker's voice — the separated track is the "
                            "voice reference. Run a separation first. XTTS v2 "
                            "(~1.8 GB) downloads on first use, and the first "
                            "clone takes ~30-60 s on CPU.")
            else:
                gr.Markdown(
                    "Voice cloning is **opt-in and not set up here**. The code "
                    "path is complete (`src/tts/voice_clone.py`), but "
                    "`coqui-tts` needs `transformers < 5`, which conflicts "
                    "with what Gradio requires — so it runs in an isolated "
                    "env. Create it with:\n\n"
                    "```\nconda create -y -n voxsplit-tts python=3.10\n"
                    "conda run -n voxsplit-tts pip install coqui-tts\n```\n"
                    "then restart the demo. Separation, AV, transcripts and "
                    "timelines are unaffected.")
            with gr.Row():
                # allow_custom_value: a browser reloading against a fresh
                # server still holds the old "Speaker 1" value while choices
                # are empty, which Gradio otherwise rejects outright.
                tts_speaker = gr.Dropdown(choices=[], value=None,
                                          label="Voice", interactive=False,
                                          allow_custom_value=True)
                tts_lang = gr.Dropdown(
                    choices=[name for name, _ in LANG_CHOICES],
                    value="English", label="Language")
            tts_text = gr.Textbox(label="Text to speak", lines=2,
                                  placeholder="Type something...")
            tts_btn = gr.Button("Clone voice", variant="secondary",
                                interactive=tts_ready)
            tts_audio = gr.Audio(label="Synthesised speech")
            if tts_ready:
                tts_btn.click(clone_voice_fn,
                              [tts_text, tts_speaker, tts_lang], tts_audio)

        # The speaker picker is the last output slot of each separation.
        btn.click(separate, [inp, sensitivity, count_choice],
                  out_a + [tts_speaker])
        vbtn.click(separate_video, [vinp, nfaces], out_v + [tts_speaker])
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the VoxSplit demo.")
    parser.add_argument("--orpit-ckpt", default=DEFAULT_ORPIT)
    parser.add_argument("--clf-ckpt", default=DEFAULT_CLF)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    _PIPELINE["kwargs"] = {"orpit_ckpt": args.orpit_ckpt,
                           "clf_ckpt": args.clf_ckpt, "device": args.device}
    print("Loading models ...")
    get_pipeline()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port,
                share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
