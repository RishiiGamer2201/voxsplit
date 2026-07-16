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

MAX_SPK = 8  # UI rows and the recursion cap; >5 is out of the trained range.

_PIPELINE = {"obj": None, "kwargs": {}}


def get_pipeline() -> Pipeline:
    if _PIPELINE["obj"] is None:
        print("Loading models (first request) ...")
        _PIPELINE["obj"] = Pipeline(**_PIPELINE["kwargs"])
    return _PIPELINE["obj"]


def _fan_out(out, label="Speaker"):
    """Turn a pipeline result dict into the fixed UI output slots."""
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
    return results


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
                btn.click(separate, [inp, sensitivity, count_choice], out_a)

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
                vbtn.click(separate_video, [vinp, nfaces], out_v)
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
