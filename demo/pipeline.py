"""End-to-end VoxSplit pipeline for the Phase 6 demo.

Loads the robust OR-PIT separator, the count/stop classifier, ECAPA, and a
Whisper model once, then turns an uploaded recording into: per-speaker audio,
before/after spectrograms, a speaker-attributed transcript, and a speaking
timeline. Shared by demo/app.py (Gradio) and usable directly as a library.

Run this file directly to process one file from the command line:
  python demo/pipeline.py path/to/mix.wav
"""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src" / "inference"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "models"))
from audio_io import load_normalized, MODEL_SR  # noqa: E402
from separate_longform import separate_longform, load_longform_models  # noqa: E402
from timeline import speaking_segments, plot_timeline  # noqa: E402
import transcribe  # noqa: E402

DEFAULT_ORPIT = "checkpoints/orpit_robust/ckpt_step26000.pt"
DEFAULT_CLF = "checkpoints/count_clf_res/ckpt_step8000.pt"
DEFAULT_ECAPA = "pretrained_models/ecapa-dl"


def spectrogram_fig(wav: np.ndarray, sr: int, title: str):
    """A log-power spectrogram Figure for display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.specgram(np.asarray(wav, dtype=np.float32), NFFT=256, Fs=sr,
                noverlap=192, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    fig.tight_layout()
    return fig


class Pipeline:
    """Loads all models once; process() runs the full chain on one file."""

    def __init__(self, orpit_ckpt: str = DEFAULT_ORPIT,
                 clf_ckpt: str = DEFAULT_CLF, ecapa_dir: str = DEFAULT_ECAPA,
                 init_model: str = "speechbrain/sepformer-wsj02mix",
                 whisper_size: str = "base.en", device: str = "auto") -> None:
        self.fwd, self.pm, self.emb = load_longform_models(
            Path(orpit_ckpt), Path(clf_ckpt), Path(ecapa_dir), init_model,
            device)
        self.whisper = transcribe.load_model(whisper_size, device)

    def process(self, audio_path: str, chunk_seconds: float = 4.0,
                overlap_seconds: float = 1.0,
                transcribe_tracks: bool = True) -> Dict:
        signal, orig_sr = load_normalized(Path(audio_path), MODEL_SR)
        chunk_len = int(round(chunk_seconds * MODEL_SR))
        hop_len = max(int(round((chunk_seconds - overlap_seconds) * MODEL_SR)),
                      1)
        tracks = separate_longform(signal, self.fwd, self.pm, self.emb,
                                   chunk_len, hop_len)

        transcripts: List[str] = []
        segments: List[list] = []
        for tr in tracks:
            segments.append(speaking_segments(tr, MODEL_SR))
            if transcribe_tracks:
                transcripts.append(
                    transcribe.transcribe_track(self.whisper, tr, MODEL_SR))
            else:
                transcripts.append("")

        duration = len(signal) / MODEL_SR
        return {
            "sr": MODEL_SR,
            "orig_sr": orig_sr,
            "duration": duration,
            "num_speakers": len(tracks),
            "tracks": [np.asarray(t, dtype=np.float32) for t in tracks],
            "input_fig": spectrogram_fig(signal, MODEL_SR, "Input mixture"),
            "speaker_figs": [spectrogram_fig(t, MODEL_SR, f"Speaker {i + 1}")
                             for i, t in enumerate(tracks)],
            "transcripts": transcripts,
            "timeline_fig": plot_timeline(segments, duration),
        }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run the VoxSplit pipeline.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--orpit-ckpt", default=DEFAULT_ORPIT)
    parser.add_argument("--clf-ckpt", default=DEFAULT_CLF)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    pipe = Pipeline(orpit_ckpt=args.orpit_ckpt, clf_ckpt=args.clf_ckpt,
                    device=args.device)
    out = pipe.process(str(args.input))
    print(f"Detected {out['num_speakers']} speaker(s) in "
          f"{out['duration']:.1f}s.")
    for i, text in enumerate(out["transcripts"], start=1):
        print(f"\n[Speaker {i}]\n{text or '(no speech detected)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
