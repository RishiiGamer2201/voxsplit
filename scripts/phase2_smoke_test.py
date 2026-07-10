"""End-to-end smoke test for VoxSplit Phase 2, no real data or model needed.

Creates a tiny fake LibriSpeech tree of synthetic audio, then exercises the
Phase 2 pipeline: build_eval_set -> realize_eval_set -> (fake estimates by
copying references) -> evaluate_set. Because the fake estimates are exact
copies of the references, SI-SDRi must come out very high, which confirms
the scoring path is wired up correctly. separate_set.py is not run end to
end here (it needs the pretrained model); its imports and --help are checked
separately in the report.

Run:
  python scripts/phase2_smoke_test.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

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
            # A distinct tone per file plus light noise so sources differ.
            freq = 150.0 + 40.0 * s + 7.0 * f
            sig = 0.3 * np.sin(2.0 * np.pi * freq * t)
            sig = sig + 0.01 * rng.standard_normal(n)
            sig = sig.astype(np.float32)
            name = f"{speaker}-{chapter}-{f:04d}.flac"
            sf.write(str(chap_dir / name), sig, SR, subtype="PCM_16")


def run(cmd: list) -> None:
    """Run a subprocess from the repo root and raise on failure."""
    print(">>>", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="voxsplit_phase2_"))
    print(f"Working in {tmp}")
    try:
        libri_root = tmp / "LibriSpeech"
        source_dir = libri_root / "test-clean"
        write_fake_librispeech(libri_root, num_speakers=4,
                               files_per_speaker=2, seconds=5.0)

        manifest = tmp / "eval_manifest.json"
        eval_set = tmp / "eval_set"
        ests_root = tmp / "ests"
        csv_path = tmp / "results.csv"

        run([PY, "src/data/build_eval_set.py",
             "--source-dir", str(source_dir),
             "--librispeech-root", str(libri_root),
             "--out", str(manifest),
             "--speaker-counts", "2,3",
             "--per-count", "2",
             "--sample-rate", str(SR),
             "--seed", "0",
             "--min-seconds", "4.0"])

        with open(manifest, "r", encoding="utf-8") as fh:
            man = json.load(fh)
        assert len(man["mixtures"]) == 4, man["mixtures"]
        assert man["mixtures"][0]["sources"][0]["gain"] == 1.0
        assert man["normalize_peak"] == 0.9
        assert man["speaker_counts"] == [2, 3]
        # Every relpath must point at a real file and be forward-slashed.
        for mix in man["mixtures"]:
            for src in mix["sources"]:
                assert "\\" not in src["relpath"], src["relpath"]
                assert (libri_root / src["relpath"]).is_file(), src["relpath"]
        print("Manifest schema checks passed.")

        run([PY, "src/data/realize_eval_set.py",
             "--manifest", str(manifest),
             "--librispeech-root", str(libri_root),
             "--out-dir", str(eval_set)])

        # Fake estimates: copy each mixture's source*.wav to est*.wav.
        for mix_dir in sorted(eval_set.iterdir()):
            if not mix_dir.is_dir():
                continue
            est_dir = ests_root / mix_dir.name
            est_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted(mix_dir.glob("source*.wav")):
                idx = src.stem.replace("source", "")
                shutil.copyfile(src, est_dir / f"est{idx}.wav")

        run([PY, "src/eval/evaluate_set.py",
             "--eval-dir", str(eval_set),
             "--ests-root", str(ests_root),
             "--sample-rate", str(SR),
             "--csv", str(csv_path),
             "--tag", "smoketest"])

        # CSV must exist with one row per K (2 and 3) plus a header.
        assert csv_path.is_file(), "CSV was not written"
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3, lines  # header + K=2 + K=3
        header = lines[0].split(",")
        assert header == ["timestamp", "tag", "num_speakers", "num_mixtures",
                          "mean_si_sdr", "mean_si_sdri", "mean_pesq",
                          "mean_stoi"], header
        # SI-SDRi is column index 5; ests == refs so it must be very high.
        for row in lines[1:]:
            cols = row.split(",")
            si_sdri = float(cols[5])
            print(f"K={cols[2]} num={cols[3]} SI-SDRi={si_sdri:.2f}")
            assert si_sdri > 50.0, f"SI-SDRi too low: {si_sdri}"
        print("")
        print("PHASE 2 SMOKE TEST PASSED.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
