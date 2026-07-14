"""Per-speaker transcription for VoxSplit Phase 6.

Runs Whisper on each separated track to produce a speaker-attributed
transcript. Uses faster-whisper (CTranslate2), NOT openai-whisper, because the
latter pulls numba/llvmlite.dll, which is blocked by a Windows Application
Control policy on this machine (the same block that stops ClearerVoice).

The transcriber is loaded once (WhisperModel) and reused across tracks. Tracks
are 8 kHz mono from the separators; Whisper wants 16 kHz, so they are
upsampled before transcription.

Run this file directly for a self-test (no real speech; checks the resample and
empty-audio paths, skips model download):
  python src/inference/transcribe.py --self-test

Example:
  python src/inference/transcribe.py --in-dir out/meeting --pattern "speaker*.wav"
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

WHISPER_SR = 16000


def load_model(size: str = "base.en", device: str = "auto"):
    """Load a faster-whisper model. CPU int8 is plenty for short tracks."""
    from faster_whisper import WhisperModel
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(size, device=device, compute_type=compute_type)


def transcribe_track(model, wav: np.ndarray, sr: int) -> str:
    """Transcribe one mono track, returning the joined text.

    Resamples to 16 kHz, runs Whisper, and concatenates segment texts.
    """
    wav = np.asarray(wav, dtype=np.float32).flatten()
    if sr != WHISPER_SR:
        wav = AF.resample(torch.from_numpy(wav), sr, WHISPER_SR).numpy()
    if wav.size == 0 or float(np.max(np.abs(wav))) < 1e-4:
        return ""
    segments, _ = model.transcribe(wav, language="en", beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe_dir(in_dir: Path, pattern: str = "speaker*.wav",
                   size: str = "base.en", device: str = "auto") -> List[str]:
    """Transcribe every track matching pattern under in_dir, in sorted order."""
    tracks = sorted(in_dir.glob(pattern))
    if not tracks:
        print(f"No tracks matching {pattern} under {in_dir}.")
        return []
    model = load_model(size, device)
    texts = []
    for track in tracks:
        data, sr = sf.read(str(track), dtype="float32", always_2d=True)
        text = transcribe_track(model, data.mean(axis=1), sr)
        texts.append(text)
        print(f"\n[{track.stem}]\n{text or '(no speech detected)'}")
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe separated speaker tracks with Whisper.")
    parser.add_argument("--in-dir", type=Path,
                        help="Directory of separated track wavs.")
    parser.add_argument("--pattern", default="speaker*.wav")
    parser.add_argument("--model", default="base.en",
                        help="faster-whisper model size (e.g. tiny.en, "
                             "base.en, small.en).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if not args.in_dir or not args.in_dir.is_dir():
        print("--in-dir (an existing directory) is required, or --self-test.")
        return 1
    transcribe_dir(args.in_dir, args.pattern, args.model, args.device)
    return 0


def _self_test() -> int:
    """Check the empty/near-silent path returns '' without loading a model."""
    class _Dummy:
        def transcribe(self, *a, **k):
            raise AssertionError("silent audio must not reach the model")

    sr = 8000
    silent = np.zeros(sr, dtype=np.float32)
    assert transcribe_track(_Dummy(), silent, sr) == ""
    # Resample path on near-silent-but-nonzero returns '' too (below floor).
    tiny = 1e-6 * np.ones(sr, dtype=np.float32)
    assert transcribe_track(_Dummy(), tiny, sr) == ""
    print("transcribe self-test passed (silent-audio guard works).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
