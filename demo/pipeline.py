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
import torch
import torchaudio.functional as AF

_SRC = Path(__file__).resolve().parent.parent / "src" / "inference"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "models"))
from audio_io import load_normalized, MODEL_SR  # noqa: E402
from separate_longform import separate_longform, load_longform_models  # noqa: E402
from timeline import speaking_segments, plot_timeline  # noqa: E402
import transcribe  # noqa: E402

# Default to the clean 20k OR-PIT: the count/stop classifier was trained on
# THIS model's residuals, so counts are most accurate here. The robust
# (noise/reverb) checkpoint separates degraded audio better but slightly
# mis-counts clean input; pass it explicitly for noisy recordings.
DEFAULT_ORPIT = "checkpoints/orpit/ckpt_step20000.pt"
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
                recursion_threshold: float = 0.5,
                cluster_threshold: float = 0.55,
                force_count: "int | None" = None,
                transcribe_tracks: bool = True) -> Dict:
        signal, orig_sr = load_normalized(Path(audio_path), MODEL_SR)
        chunk_len = int(round(chunk_seconds * MODEL_SR))
        hop_len = max(int(round((chunk_seconds - overlap_seconds) * MODEL_SR)),
                      1)
        tracks = separate_longform(signal, self.fwd, self.pm, self.emb,
                                   chunk_len, hop_len,
                                   cluster_threshold=cluster_threshold,
                                   recursion_threshold=recursion_threshold,
                                   force_count=force_count)
        return self._build_outputs(signal, tracks, orig_sr, transcribe_tracks)

    def _build_outputs(self, signal, tracks, orig_sr, transcribe_tracks,
                       label="Speaker") -> Dict:
        """Spectrograms, transcripts, and a timeline for a set of tracks."""
        transcripts: List[str] = []
        segments: List[list] = []
        for tr in tracks:
            segments.append(speaking_segments(tr, MODEL_SR))
            transcripts.append(
                transcribe.transcribe_track(self.whisper, tr, MODEL_SR)
                if transcribe_tracks else "")
        duration = len(signal) / MODEL_SR
        return {
            "sr": MODEL_SR,
            "orig_sr": orig_sr,
            "duration": duration,
            "num_speakers": len(tracks),
            "tracks": [np.asarray(t, dtype=np.float32) for t in tracks],
            "input_fig": spectrogram_fig(signal, MODEL_SR, "Input mixture"),
            "speaker_figs": [spectrogram_fig(t, MODEL_SR, f"{label} {i + 1}")
                             for i, t in enumerate(tracks)],
            "transcripts": transcripts,
            "timeline_fig": plot_timeline(segments, duration),
        }

    def process_video(self, video_path: str, num_faces: int = 0,
                      transcribe_tracks: bool = True) -> Dict:
        """Audio-visual: video sets the count, lip motion assigns tracks.

        num_faces = 0 tries mediapipe to count faces, else falls back to a
        2-speaker assumption; pass a number for the ROI-motion fallback.
        """
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                                / "src" / "av"))
        from separate_av import read_video  # noqa: E402
        from lipmotion import (extract_face_motions,  # noqa: E402
                               mediapipe_mouth_series)
        from av_assign import audio_envelope, assign_tracks_to_faces  # noqa

        frames, audio, sr, fps = read_video(Path(video_path))
        if frames is None or audio is None:
            raise ValueError("The video has no frames or no audio track.")

        mp_faces = mediapipe_mouth_series(frames, max(num_faces or 5, 1))
        n = len(mp_faces) if mp_faces else (num_faces or 2)

        audio8k = audio
        if sr != MODEL_SR:
            audio8k = AF.resample(torch.from_numpy(audio.astype(np.float32)),
                                  sr, MODEL_SR).numpy()

        chunk_len, hop_len = int(4.0 * MODEL_SR), int(3.0 * MODEL_SR)
        tracks = separate_longform(audio8k, self.fwd, self.pm, self.emb,
                                   chunk_len, hop_len, force_count=n)

        face_motions = mp_faces or extract_face_motions(frames, n)
        envs = [audio_envelope(t, MODEL_SR, len(frames), fps) for t in tracks]
        assignment, _ = assign_tracks_to_faces(
            envs, [m for _, m in face_motions])

        # Order tracks by face (left-to-right); unassigned tracks appended.
        ordered = [None] * len(face_motions)
        leftover = []
        for t_idx, f_idx in enumerate(assignment):
            if 0 <= f_idx < len(ordered) and ordered[f_idx] is None:
                ordered[f_idx] = tracks[t_idx]
            else:
                leftover.append(tracks[t_idx])
        ordered = [t for t in ordered if t is not None] + leftover

        out = self._build_outputs(audio8k, ordered, sr, transcribe_tracks,
                                  label="On-screen speaker")
        out["face_source"] = "mediapipe" if mp_faces else "ROI fallback"
        return out


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
