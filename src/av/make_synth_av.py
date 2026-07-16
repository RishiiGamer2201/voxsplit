"""Render a synthetic audio-visual test clip for VoxSplit AV mode.

Builds a 2-speaker mixture from LibriSpeech and a video of two side-by-side
faces whose mouths open in time with each speaker's loudness. Muxes them into an
mp4 so `separate_av.py` can be validated end to end here (mediapipe won't detect
cartoon faces, so the ROI-motion fallback drives it — pass --num-faces 2).

  python src/av/make_synth_av.py --out scratch/av_test.mp4
  python src/av/separate_av.py scratch/av_test.mp4 --out-dir out/av --num-faces 2
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mixing"))

SR = 8000
FPS = 25


def envelope(wav: np.ndarray, num_frames: int) -> np.ndarray:
    hop = len(wav) / num_frames
    env = np.array([np.sqrt(np.mean(wav[int(i * hop):int((i + 1) * hop)] ** 2)
                            + 1e-9) for i in range(num_frames)])
    env = env - env.min()
    return env / (env.max() + 1e-9)


def draw_face(frame, cx, cy, mouth_open):
    import cv2
    cv2.circle(frame, (cx, cy), 80, (200, 190, 170), -1)          # head
    cv2.circle(frame, (cx - 30, cy - 25), 10, (30, 30, 30), -1)   # eyes
    cv2.circle(frame, (cx + 30, cy - 25), 10, (30, 30, 30), -1)
    h = int(4 + 34 * float(mouth_open))                           # mouth height
    cv2.ellipse(frame, (cx, cy + 40), (34, h), 0, 0, 360, (40, 20, 20), -1)


def main() -> int:
    import cv2
    import imageio.v2 as imageio
    from make_mixture import find_sources, load_mono_resampled, rms

    parser = argparse.ArgumentParser(description="Render a synthetic AV clip.")
    parser.add_argument("--out", type=Path, default=Path("scratch/av_test.mp4"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("data/LibriSpeech/test-clean"))
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    by = find_sources(args.source_dir, 4.0, SR)
    spk = sorted(by.keys())[:2]
    n = int(args.seconds * SR)
    sigs = []
    for s in spk:
        x = load_mono_resampled(by[s][0], SR)[:n]
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        sigs.append(x.astype(np.float32))
    sigs[1] = sigs[1] * (rms(sigs[0]) / rms(sigs[1]))
    mix = sigs[0] + sigs[1]
    scale = 0.9 / np.max(np.abs(mix))
    mix = (mix * scale).astype(np.float32)

    num_frames = int(args.seconds * FPS)
    env0 = envelope(sigs[0], num_frames)
    env1 = envelope(sigs[1], num_frames)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="voxsplit_av_"))
    tmp_video = tmp_dir / "video.mp4"
    tmp_audio = tmp_dir / "audio.wav"
    sf.write(str(tmp_audio), mix, SR, subtype="PCM_16")

    writer = imageio.get_writer(str(tmp_video), fps=FPS, macro_block_size=1)
    for i in range(num_frames):
        frame = np.full((240, 640, 3), 30, dtype=np.uint8)
        draw_face(frame, 160, 120, env0[i])   # left speaker (speaker A)
        draw_face(frame, 480, 120, env1[i])   # right speaker (speaker B)
        writer.append_data(frame)
    writer.close()

    # Mux video + audio with ffmpeg (installed via conda).
    ffmpeg = "ffmpeg"
    cmd = [ffmpeg, "-y", "-i", str(tmp_video), "-i", str(tmp_audio),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-shortest", str(args.out)]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Wrote synthetic AV clip: {args.out} "
          f"({num_frames} frames @ {FPS} fps, {args.seconds}s). "
          f"Left face = speaker A, right = speaker B.")
    # Also drop the clean per-speaker refs next to it for scoring.
    sf.write(str(args.out.with_name("av_ref_left.wav")),
             (sigs[0] * scale).astype(np.float32), SR, subtype="FLOAT")
    sf.write(str(args.out.with_name("av_ref_right.wav")),
             (sigs[1] * scale).astype(np.float32), SR, subtype="FLOAT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
