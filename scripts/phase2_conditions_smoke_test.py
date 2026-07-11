"""Smoke test for the VoxSplit Phase 2 NOISE and REVERB conditions.

No real dataset or model is needed. The test:
  1. builds a tiny fake LibriSpeech tree and a clean manifest via
     build_eval_set.py, then realizes it (the clean path must still work);
  2. writes a couple of synthetic noise wavs into a temp dir;
  3. runs make_conditions for BOTH noise and reverb, then realizes each;
  4. checks every output folder has mixture.wav and equal-length source*.wav;
  5. checks reverb outputs are longer-tailed than the clean sources;
  6. asserts determinism: make_conditions twice with the same seed yields
     identical manifests (ignoring the created timestamp), and realize twice
     yields byte-identical audio.

Run:
  python scripts/phase2_conditions_smoke_test.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SR = 8000


def write_fake_librispeech(root: Path, num_speakers: int,
                           files_per_speaker: int, seconds: float) -> None:
    """Create test-clean/<spk>/<chapter>/<spk>-<chapter>-<idx>.flac tones."""
    subset = root / "test-clean"
    rng = np.random.default_rng(0)
    n = int(seconds * SR)
    t = np.arange(n) / float(SR)
    for s in range(num_speakers):
        speaker = str(100 + s)
        chapter = "0"
        chap_dir = subset / speaker / chapter
        chap_dir.mkdir(parents=True, exist_ok=True)
        for f in range(files_per_speaker):
            freq = 150.0 + 40.0 * s + 7.0 * f
            sig = 0.3 * np.sin(2.0 * np.pi * freq * t)
            sig = sig + 0.01 * rng.standard_normal(n)
            name = f"{speaker}-{chapter}-{f:04d}.flac"
            sf.write(str(chap_dir / name), sig.astype(np.float32), SR,
                     subtype="PCM_16")


def write_fake_noise(noise_dir: Path) -> None:
    """Write a couple of synthetic noise wavs of differing lengths."""
    noise_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for i, seconds in enumerate([3.0, 7.0]):
        n = int(seconds * SR)
        sig = 0.2 * rng.standard_normal(n)
        sub = noise_dir / f"cat{i}"
        sub.mkdir(parents=True, exist_ok=True)
        sf.write(str(sub / f"noise_{i}.wav"), sig.astype(np.float32), SR,
                 subtype="FLOAT")


def run(cmd: list) -> None:
    """Run a subprocess from the repo root and raise on failure."""
    print(">>>", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")


def load_wav(path: Path) -> np.ndarray:
    _, data = wavfile.read(str(path))
    return np.asarray(data)


def check_folder_lengths(eval_dir: Path) -> dict:
    """Assert each mixture folder has equal-length mixture and sources.

    Returns a map of mixture id -> source length.
    """
    lengths = {}
    subdirs = [d for d in sorted(eval_dir.iterdir()) if d.is_dir()]
    assert subdirs, f"No mixture folders in {eval_dir}"
    for d in subdirs:
        mix = d / "mixture.wav"
        assert mix.is_file(), f"Missing {mix}"
        mix_len = len(load_wav(mix))
        srcs = sorted(d.glob("source*.wav"))
        assert srcs, f"No source wavs in {d}"
        for s in srcs:
            slen = len(load_wav(s))
            assert slen == mix_len, (
                f"Length mismatch in {d.name}: {s.name}={slen} "
                f"mixture={mix_len}")
        lengths[d.name] = mix_len
    return lengths


def strip_created(manifest_path: Path) -> str:
    """Return manifest JSON text with the created field blanked out."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        man = json.load(fh)
    man["created"] = ""
    return json.dumps(man, indent=2, sort_keys=True)


def folder_bytes(eval_dir: Path) -> dict:
    """Map relative wav path -> raw bytes for every wav under eval_dir."""
    out = {}
    for wav in sorted(eval_dir.rglob("*.wav")):
        out[wav.relative_to(eval_dir).as_posix()] = wav.read_bytes()
    return out


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="voxsplit_cond_"))
    print(f"Working in {tmp}")
    try:
        libri_root = tmp / "LibriSpeech"
        source_dir = libri_root / "test-clean"
        write_fake_librispeech(libri_root, num_speakers=4,
                               files_per_speaker=2, seconds=5.0)

        noise_dir = tmp / "noise"
        write_fake_noise(noise_dir)

        clean_manifest = tmp / "eval_manifest.json"
        run([PY, "src/data/build_eval_set.py",
             "--source-dir", str(source_dir),
             "--librispeech-root", str(libri_root),
             "--out", str(clean_manifest),
             "--speaker-counts", "2,3",
             "--per-count", "2",
             "--sample-rate", str(SR),
             "--seed", "0",
             "--min-seconds", "4.0"])

        # Clean realize must still work unchanged.
        clean_set = tmp / "clean_set"
        run([PY, "src/data/realize_eval_set.py",
             "--manifest", str(clean_manifest),
             "--librispeech-root", str(libri_root),
             "--out-dir", str(clean_set)])
        clean_lengths = check_folder_lengths(clean_set)
        print("Clean realize OK.")

        # NOISE manifest and realize.
        noise_manifest = tmp / "eval_manifest_noise.json"
        run([PY, "src/data/make_conditions.py",
             "--manifest", str(clean_manifest),
             "--condition", "noise",
             "--noise-dir", str(noise_dir),
             "--out", str(noise_manifest),
             "--seed", "0"])
        with open(noise_manifest, "r", encoding="utf-8") as fh:
            nman = json.load(fh)
        assert nman["condition"] == "noise", nman.get("condition")
        assert nman["noise_dir"] == str(noise_dir), nman.get("noise_dir")
        for mix in nman["mixtures"]:
            assert "noise" in mix, mix["id"]
            assert "\\" not in mix["noise"]["relpath"], mix["noise"]["relpath"]
            assert isinstance(mix["noise"]["start"], int)
        print("Noise manifest schema OK.")

        noise_set = tmp / "noise_set"
        run([PY, "src/data/realize_eval_set.py",
             "--manifest", str(noise_manifest),
             "--librispeech-root", str(libri_root),
             "--out-dir", str(noise_set)])
        noise_lengths = check_folder_lengths(noise_set)
        # Noise references stay clean, so their lengths equal the clean run.
        assert noise_lengths == clean_lengths, (
            "Noise output lengths should match clean lengths")
        # The noisy mixture must actually differ from the clean mixture.
        any_diff = False
        for name in clean_lengths:
            c = load_wav(clean_set / name / "mixture.wav")
            nz = load_wav(noise_set / name / "mixture.wav")
            if not np.array_equal(c, nz):
                any_diff = True
        assert any_diff, "Noisy mixtures are identical to clean mixtures"
        print("Noise realize OK (mixtures differ, refs stay clean).")

        # REVERB manifest and realize.
        reverb_manifest = tmp / "eval_manifest_reverb.json"
        run([PY, "src/data/make_conditions.py",
             "--manifest", str(clean_manifest),
             "--condition", "reverb",
             "--out", str(reverb_manifest),
             "--seed", "0"])
        with open(reverb_manifest, "r", encoding="utf-8") as fh:
            rman = json.load(fh)
        assert rman["condition"] == "reverb", rman.get("condition")
        for mix in rman["mixtures"]:
            rv = mix["reverb"]
            assert len(rv["room_dim"]) == 3
            assert len(rv["mic_pos"]) == 3
            assert len(rv["source_pos"]) == mix["num_speakers"]
        print("Reverb manifest schema OK.")

        reverb_set = tmp / "reverb_set"
        run([PY, "src/data/realize_eval_set.py",
             "--manifest", str(reverb_manifest),
             "--librispeech-root", str(libri_root),
             "--out-dir", str(reverb_set)])
        reverb_lengths = check_folder_lengths(reverb_set)
        # Reverb keeps the convolution tail, so outputs are longer than clean.
        for name, rlen in reverb_lengths.items():
            assert rlen > clean_lengths[name], (
                f"Reverb {name} not longer-tailed: reverb={rlen} "
                f"clean={clean_lengths[name]}")
        print("Reverb realize OK (longer-tailed than clean).")

        # Determinism: make_conditions twice, same seed -> identical manifest
        # (ignoring the created timestamp).
        noise_manifest_b = tmp / "eval_manifest_noise_b.json"
        reverb_manifest_b = tmp / "eval_manifest_reverb_b.json"
        run([PY, "src/data/make_conditions.py",
             "--manifest", str(clean_manifest), "--condition", "noise",
             "--noise-dir", str(noise_dir), "--out", str(noise_manifest_b),
             "--seed", "0"])
        run([PY, "src/data/make_conditions.py",
             "--manifest", str(clean_manifest), "--condition", "reverb",
             "--out", str(reverb_manifest_b), "--seed", "0"])
        assert strip_created(noise_manifest) == strip_created(noise_manifest_b)
        assert strip_created(reverb_manifest) == strip_created(
            reverb_manifest_b)
        print("make_conditions determinism OK.")

        # Determinism: realize twice -> byte-identical audio.
        noise_set_b = tmp / "noise_set_b"
        reverb_set_b = tmp / "reverb_set_b"
        run([PY, "src/data/realize_eval_set.py",
             "--manifest", str(noise_manifest),
             "--librispeech-root", str(libri_root),
             "--out-dir", str(noise_set_b)])
        run([PY, "src/data/realize_eval_set.py",
             "--manifest", str(reverb_manifest),
             "--librispeech-root", str(libri_root),
             "--out-dir", str(reverb_set_b)])
        assert folder_bytes(noise_set) == folder_bytes(noise_set_b), (
            "Noise realize is not byte-reproducible")
        assert folder_bytes(reverb_set) == folder_bytes(reverb_set_b), (
            "Reverb realize is not byte-reproducible")
        print("realize determinism OK (byte-identical).")

        print("")
        print("PHASE 2 CONDITIONS SMOKE TEST PASSED.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
