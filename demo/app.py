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
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import Pipeline, DEFAULT_ORPIT, DEFAULT_CLF  # noqa: E402

MAX_SPK = 5  # UI rows; the model can detect up to this many speakers.

_PIPELINE = {"obj": None, "kwargs": {}}


def get_pipeline() -> Pipeline:
    if _PIPELINE["obj"] is None:
        print("Loading models (first request) ...")
        _PIPELINE["obj"] = Pipeline(**_PIPELINE["kwargs"])
    return _PIPELINE["obj"]


def separate(audio_path, sensitivity):
    """Run the pipeline and fan the results out into the fixed UI slots.

    `sensitivity` is 1 - recursion_threshold, so a higher slider value splits
    more aggressively (finds more speakers). Use it if the detected count looks
    too low (voices merged) or too high (a speaker split in two).
    """
    if not audio_path:
        raise gr.Error("Please upload an audio file first.")
    recursion_threshold = float(np.clip(1.0 - sensitivity, 0.2, 0.7))
    out = get_pipeline().process(audio_path,
                                 recursion_threshold=recursion_threshold)
    sr = out["sr"]
    n = out["num_speakers"]

    status = (f"Detected **{n}** speaker(s) in {out['duration']:.1f}s "
              f"(input {out['orig_sr']} Hz).")
    results = [status, out["input_fig"], out["timeline_fig"]]

    for i in range(MAX_SPK):
        if i < n:
            audio = (sr, out["tracks"][i])
            results += [
                gr.update(visible=True),
                gr.update(value=audio, visible=True, label=f"Speaker {i + 1}"),
                gr.update(value=out["speaker_figs"][i], visible=True),
                gr.update(value=out["transcripts"][i] or "(no speech detected)",
                          visible=True),
            ]
        else:
            results += [gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False)]
    return results


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VoxSplit") as demo:
        gr.Markdown("# VoxSplit — multi-speaker speech separation\n"
                    "Upload a recording where several people talk at once. "
                    "VoxSplit finds how many speakers there are and returns a "
                    "clean track, spectrogram, and transcript for each.")
        with gr.Row():
            inp = gr.Audio(type="filepath", label="Input recording")
            with gr.Column():
                status = gr.Markdown()
                sensitivity = gr.Slider(
                    0.3, 0.8, value=0.5, step=0.05,
                    label="Split sensitivity",
                    info="Raise if voices are merged into one track; lower if "
                         "one speaker is split into two.")
                btn = gr.Button("Separate", variant="primary")
        with gr.Row():
            input_spec = gr.Plot(label="Input spectrogram")
            timeline = gr.Plot(label="Speaking timeline")

        rows, audios, specs, texts = [], [], [], []
        for i in range(MAX_SPK):
            with gr.Group(visible=False) as row:
                gr.Markdown(f"### Speaker {i + 1}")
                a = gr.Audio(label=f"Speaker {i + 1}")
                with gr.Row():
                    s = gr.Plot(label="Spectrogram")
                    t = gr.Textbox(label="Transcript", lines=3)
            rows.append(row)
            audios.append(a)
            specs.append(s)
            texts.append(t)

        outputs = [status, input_spec, timeline]
        for r, a, s, t in zip(rows, audios, specs, texts):
            outputs += [r, a, s, t]
        btn.click(separate, inputs=[inp, sensitivity], outputs=outputs)
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
