"""Audio-visual separation for VoxSplit (working feature).

Given a VIDEO with several people talking at once, VoxSplit:
  1. reads frames + the audio track (PyAV),
  2. counts on-screen faces (mediapipe) -> the speaker count comes from the
     VIDEO, so separation is forced to exactly that many speakers (no reliance
     on the shaky audio-only count classifier),
  3. separates the audio into that many tracks (our OR-PIT recursion),
  4. assigns each track to the face whose mouth motion best correlates with the
     track's loudness envelope (Hungarian), and
  5. writes one audio track per on-screen speaker, left-to-right.

This is a genuine audio-visual system: the video determines both the speaker
COUNT and each track's speaker IDENTITY. It runs fully in our env with no
external AV-separation model. Full AV masking (RTFS-Net/IIANet/CTCNet) remains
the heavier alternative documented in README.md.

If the input has no video frames, it falls back to the audio-only pipeline.

Run this file directly for a self-test (synthetic, no real video/model):
  python src/av/separate_av.py --self-test
Make a synthetic test clip and run it end to end with:
  python src/av/make_synth_av.py --out scratch/av_test.mp4
  python src/av/separate_av.py scratch/av_test.mp4 --out-dir out/av --num-faces 2
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

_AV = Path(__file__).resolve().parent
sys.path.insert(0, str(_AV))
sys.path.insert(0, str(_AV.parent / "inference"))
from av_assign import audio_envelope, assign_tracks_to_faces  # noqa: E402
from lipmotion import extract_face_motions, mediapipe_mouth_series  # noqa: E402

MODEL_SR = 8000


def read_video(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                    Optional[int], float]:
    """Return (frames [T,H,W,3] uint8, audio mono float32, sr, fps)."""
    import av  # PyAV
    container = av.open(str(path))
    fps = 25.0
    if container.streams.video:
        rate = container.streams.video[0].average_rate
        if rate:
            fps = float(rate)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    frames = np.stack(frames) if frames else None

    audio_chunks, sr = [], None
    try:
        container.seek(0)
        if container.streams.audio:
            astream = container.streams.audio[0]
            sr = astream.rate
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
            for aframe in container.decode(audio=0):
                for rframe in resampler.resample(aframe):
                    audio_chunks.append(rframe.to_ndarray().flatten())
            # Flush the resampler
            for rframe in resampler.resample(None) or []:
                audio_chunks.append(rframe.to_ndarray().flatten())
    except Exception as e:
        print(f"Warning: av.AudioResampler failed: {e}. Falling back to manual NumPy downmixing...")
        audio_chunks, sr = [], None
        try:
            container.seek(0)
            for aframe in container.decode(audio=0):
                sr = aframe.sample_rate
                data = aframe.to_ndarray()
                # planar format (channels, samples)
                if data.ndim == 2:
                    if data.shape[0] < data.shape[1]:
                        mono = data.mean(axis=0)
                    else:
                        mono = data.mean(axis=1)
                elif data.ndim == 1:
                    mono = data
                else:
                    mono = data.mean(axis=tuple(range(data.ndim - 1)))
                audio_chunks.append(mono.astype(np.float32))
        except Exception as e2:
            print(f"Warning: Manual downmixing failed: {e2}")

    audio = np.concatenate(audio_chunks) if audio_chunks else None
    return frames, audio, sr, fps


def separate_audio(audio8k: np.ndarray, num_speakers: int, orpit_ckpt: str,
                   clf_ckpt: str, device: str) -> List[np.ndarray]:
    """Separate the audio into exactly num_speakers tracks (forced count)."""
    from separate_longform import separate_longform, load_longform_models
    fwd, pm, emb = load_longform_models(
        Path(orpit_ckpt), Path(clf_ckpt), Path("pretrained_models/ecapa-dl"),
        "speechbrain/sepformer-wsj02mix", device)
    chunk_len = int(4.0 * MODEL_SR)
    hop_len = int(3.0 * MODEL_SR)
    return separate_longform(audio8k, fwd, pm, emb, chunk_len, hop_len,
                             force_count=num_speakers)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audio-visual separation: assign tracks to on-screen faces.")
    parser.add_argument("input", nargs="?", type=Path, help="Input video.")
    parser.add_argument("--orpit-ckpt",
                        default="checkpoints/orpit/ckpt_step20000.pt")
    parser.add_argument("--clf-ckpt",
                        default="checkpoints/count_clf_res/ckpt_step8000.pt")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--num-faces", type=int, default=0,
                        help="Speaker/face count when mediapipe finds none "
                             "(e.g. synthetic clips); 0 = auto from mediapipe.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()
    if not (args.input and args.out_dir):
        print("input video and --out-dir are required (or --self-test).")
        return 1

    device = "cuda:0" if (args.device in ("auto", "cuda")
                          and torch.cuda.is_available()) else "cpu"
    frames, audio, sr, fps = read_video(args.input)
    if frames is None or audio is None:
        print("No video frames or no audio; use the audio-only entry points.")
        return 1
    print(f"video: {len(frames)} frames @ {fps:.1f} fps, audio {sr} Hz.")

    # Speaker count from the video (mediapipe), else --num-faces.
    mp_faces = mediapipe_mouth_series(frames, max(args.num_faces or 5, 1))
    num_speakers = len(mp_faces) if mp_faces else args.num_faces
    if num_speakers < 1:
        print("Could not determine face count; pass --num-faces N.")
        return 1
    print(f"speakers from video: {num_speakers}"
          f" ({'mediapipe' if mp_faces else 'ROI fallback'}).")

    audio8k = audio
    if sr != MODEL_SR:
        audio8k = AF.resample(torch.from_numpy(audio), sr, MODEL_SR).numpy()
    tracks = separate_audio(audio8k, num_speakers, args.orpit_ckpt,
                            args.clf_ckpt, device)

    face_motions = (mp_faces if mp_faces
                    else extract_face_motions(frames, num_speakers))
    envs = [audio_envelope(t, MODEL_SR, len(frames), fps) for t in tracks]
    assignment, corr = assign_tracks_to_faces(
        envs, [m for _, m in face_motions])
    print(f"track->face assignment: {assignment}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for track_idx, face_idx in enumerate(assignment):
        label = face_idx + 1 if face_idx >= 0 else f"un{track_idx}"
        sf.write(str(args.out_dir / f"speaker{label}.wav"),
                 tracks[track_idx].astype(np.float32), MODEL_SR, subtype="FLOAT")
    print(f"Wrote {len(tracks)} per-speaker tracks (assigned to faces "
          f"left-to-right) in {args.out_dir}.")
    return 0


def _self_test() -> int:
    """Assignment wiring on synthetic tracks + face motions (no model/video)."""
    n = 150
    ma = np.zeros(n); ma[10:70] = 1
    mb = np.zeros(n); mb[80:140] = 1
    envs = [mb.copy(), ma.copy()]           # track0~faceB, track1~faceA
    assign, _ = assign_tracks_to_faces(envs, [ma, mb])
    assert assign == [1, 0], assign
    print("AV separate self-test passed (video-guided assignment wiring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
