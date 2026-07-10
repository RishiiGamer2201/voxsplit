"""Run ClearerVoice-Studio MossFormer2 separation on a single audio file.

MossFormer2_SS_16K is a 2-speaker, 16 kHz separation model. It ships in the
`clearvoice` package, whose dependencies differ from our main env, so this
script is meant to run in a SEPARATE conda env:

  conda create -y -n clearvoice python=3.10
  conda activate clearvoice
  pip install clearvoice
  python src/inference/separate_mossformer2.py <input.wav> --out-dir <dir>

It writes est1.wav ... estN.wav at 16 kHz so the main env's evaluate.py can
score them directly.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_SR = 16000


def patch_reader_to_soundfile() -> None:
    """Make clearvoice read audio via soundfile instead of pydub/ffprobe.

    clearvoice loads audio with pydub, which shells out to ffprobe. The
    conda-forge ffprobe on this machine fails to start (DLL entrypoint error),
    so we swap clearvoice's read_audio for a soundfile-backed shim that
    returns the same interface the loader consumes: frame_rate, channels,
    sample_width, and get_array_of_samples (interleaved int16, like pydub).
    """
    from clearvoice.dataloader import dataloader as dl

    class _Audio:
        def __init__(self, samples, sr: int, channels: int):
            self._samples = samples
            self.frame_rate = sr
            self.channels = channels
            self.sample_width = 2

        def get_array_of_samples(self):
            return self._samples

    def read_audio(path):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        interleaved = data.reshape(-1)  # [t0c0, t0c1, t1c0, ...] like pydub
        ints = np.clip(np.round(interleaved * 32767.0),
                       -32768, 32767).astype(np.int16)
        return _Audio(ints, sr, data.shape[1])

    dl.read_audio = read_audio


def to_source_list(output):
    """Normalize ClearVoice separation output into a list of 1-D arrays."""
    if isinstance(output, list):
        return [np.asarray(a, dtype=np.float32).reshape(-1) for a in output]
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 1:
        return [arr]
    # Two-dimensional: put the shorter axis as the speaker axis.
    if arr.shape[0] <= arr.shape[1]:
        return [arr[i].reshape(-1) for i in range(arr.shape[0])]
    return [arr[:, i].reshape(-1) for i in range(arr.shape[1])]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Separate 2 speakers with MossFormer2_SS_16K (clearvoice).")
    parser.add_argument("input", type=Path, help="Input mixture wav file.")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Directory to write estimated sources.")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}")
        return 1

    from clearvoice import ClearVoice

    patch_reader_to_soundfile()
    print("Loading MossFormer2_SS_16K ...")
    cv = ClearVoice(task="speech_separation",
                    model_names=["MossFormer2_SS_16K"])

    output = cv(input_path=str(args.input), online_write=False)
    sources = to_source_list(output)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, src in enumerate(sources, start=1):
        out_path = args.out_dir / f"est{i}.wav"
        sf.write(str(out_path), src.astype(np.float32), MODEL_SR, subtype="FLOAT")
        written.append(out_path)

    print(f"Wrote {len(written)} estimated source(s) at {MODEL_SR} Hz:")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
