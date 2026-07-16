"""Audio-visual separation scaffold (VoxSplit Phase 6/7 bonus).

A runnable skeleton for the *Looking to Listen* homage. It reads a video's
frames (PyAV) and audio track, defines the two components a full AV system
needs (a face/mouth cropper and an audio-visual separator), and — because
neither a pretrained AV model nor a face detector is wired in this environment —
falls back to VoxSplit's audio-only blind pipeline on the video's audio. This
means the entry point WORKS today on any video; finishing AV mode is a matter of
implementing the two interfaces (see src/av/README.md).

Run this file directly for a self-test (synthetic frames, no real video/model):
  python src/av/separate_av.py --self-test
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf


class FaceCropper:
    """Interface: detect faces and return per-speaker mouth-region clips.

    A real implementation (mediapipe / MTCNN) returns, for each detected face,
    a [num_frames, H, W, 3] uint8 mouth crop aligned to the audio. The scaffold
    returns an empty list (no faces), which triggers the audio-only fallback.
    """

    def crop(self, frames: np.ndarray) -> List[np.ndarray]:  # [T, H, W, 3]
        return []  # TODO: implement with a face detector; see README.


class AVSeparator:
    """Interface: separate audio using per-speaker mouth crops.

    A real implementation loads a pretrained AV model (RTFS-Net / IIANet /
    CTCNet) and returns one audio track per face clip. The scaffold is not
    available, so callers must fall back to the audio-only pipeline.
    """

    available = False

    def separate(self, audio: np.ndarray, sr: int,
                 face_clips: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError(
            "Wire a pretrained AV model (RTFS-Net/IIANet/CTCNet); see README.")


def read_video(path: Path):
    """Read frames [T, H, W, 3] uint8 and the mono audio track (sr, samples).

    Uses PyAV (installed via faster-whisper). Returns (frames, audio, sr);
    frames or audio may be None if the container lacks that stream.
    """
    import av  # PyAV
    container = av.open(str(path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    frames = np.stack(frames) if frames else None

    audio_chunks = []
    sr = None
    try:
        container.seek(0)
        for aframe in container.decode(audio=0):
            sr = aframe.sample_rate
            audio_chunks.append(aframe.to_ndarray().reshape(-1))
    except Exception:
        pass
    audio = (np.concatenate(audio_chunks).astype(np.float32)
             if audio_chunks else None)
    return frames, audio, sr


def audio_only_fallback(audio: np.ndarray, sr: int, orpit_ckpt: str,
                        clf_ckpt: str, out_dir: Path) -> int:
    """Run VoxSplit's audio-only blind pipeline on the video's audio track."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
    import soundfile as _sf  # noqa: F401
    tmp = out_dir / "_audio_from_video.wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp), audio, sr, subtype="FLOAT")
    from separate_unknown import main as sep_main  # reuse the CLI
    argv = ["separate_unknown", str(tmp), "--orpit-ckpt", orpit_ckpt,
            "--clf-ckpt", clf_ckpt, "--out-dir", str(out_dir)]
    old = sys.argv
    sys.argv = argv
    try:
        return sep_main()
    finally:
        sys.argv = old


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audio-visual separation scaffold (audio-only fallback).")
    parser.add_argument("input", nargs="?", type=Path, help="Input video.")
    parser.add_argument("--orpit-ckpt",
                        default="checkpoints/orpit/ckpt_step20000.pt")
    parser.add_argument("--clf-ckpt",
                        default="checkpoints/count_clf_res/ckpt_step8000.pt")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if not (args.input and args.out_dir):
        print("input video and --out-dir are required (or --self-test).")
        return 1

    frames, audio, sr = read_video(args.input)
    print(f"video: {0 if frames is None else len(frames)} frames, "
          f"audio {'present' if audio is not None else 'missing'}.")

    cropper = FaceCropper()
    face_clips = cropper.crop(frames) if frames is not None else []
    if AVSeparator.available and face_clips:
        tracks = AVSeparator().separate(audio, sr, face_clips)  # not wired yet
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(tracks, 1):
            sf.write(str(args.out_dir / f"speaker{i}.wav"), t, sr,
                     subtype="FLOAT")
        print(f"AV separation wrote {len(tracks)} tracks.")
        return 0

    if audio is None:
        print("No audio track and no AV model; nothing to do.")
        return 1
    print("No AV model/faces available -> audio-only fallback.")
    return audio_only_fallback(audio, sr, args.orpit_ckpt, args.clf_ckpt,
                               args.out_dir)


def _self_test() -> int:
    """Interfaces behave as documented on synthetic frames."""
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    assert FaceCropper().crop(frames) == []          # no detector -> no faces
    assert AVSeparator.available is False
    try:
        AVSeparator().separate(np.zeros(8000, np.float32), 8000, [])
        raise AssertionError("should have raised NotImplementedError")
    except NotImplementedError:
        pass
    print("AV scaffold self-test passed (interfaces + fallback wiring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
