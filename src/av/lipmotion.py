"""Per-face mouth-motion extraction for VoxSplit AV mode.

Two extractors:
  - mediapipe FaceMesh: real faces -> inner-lip aperture per detected face,
    slotted left-to-right by face centre so identities are stable across frames.
  - ROI-motion fallback: split the frame into vertical strips and measure
    lower-face frame-to-frame change per strip. Works on any video (including
    synthetic test clips mediapipe can't detect), and is the automatic fallback
    when no faces are found.

Both return a list of (center_x, motion_series) sorted left-to-right, one per
speaker slot, with motion standardized (zero-mean, unit-variance).

Run this file directly for a self-test (synthetic frames, ROI path):
  python src/av/lipmotion.py
"""
from typing import List, Optional, Tuple

import numpy as np

Face = Tuple[float, np.ndarray]  # (center_x in [0,1], motion series)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def mediapipe_mouth_series(frames: np.ndarray, num_faces: int
                           ) -> Optional[List[Face]]:
    """Inner-lip aperture per face via mediapipe FaceMesh, or None if no faces.

    frames: [T, H, W, 3] uint8 RGB. Faces are slotted by rounding their centre
    x to `num_faces` bins (assumes a roughly static multi-speaker layout).
    """
    try:
        import mediapipe as mp
        # Legacy Solutions API (absent in Tasks-only builds like 0.10.35).
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=num_faces,
            refine_landmarks=False, min_detection_confidence=0.5)
    except Exception:
        return None  # no legacy mediapipe -> caller uses the ROI fallback
    t = len(frames)
    # Per-slot series; NaN where a slot had no face that frame.
    series = [np.full(t, np.nan) for _ in range(num_faces)]
    centers = [[] for _ in range(num_faces)]
    try:
        for fi, frame in enumerate(frames):
            res = mesh.process(np.ascontiguousarray(frame))
            if not res.multi_face_landmarks:
                continue
            faces = []
            for lm in res.multi_face_landmarks:
                pts = lm.landmark
                cx = float(np.mean([p.x for p in pts]))
                # Inner lips 13 (upper) / 14 (lower); normalize by face height.
                top = min(p.y for p in pts)
                bot = max(p.y for p in pts)
                height = (bot - top) + 1e-6
                openness = abs(pts[13].y - pts[14].y) / height
                faces.append((cx, openness))
            faces.sort(key=lambda x: x[0])
            for slot, (cx, op) in enumerate(faces[:num_faces]):
                series[slot][fi] = op
                centers[slot].append(cx)
    finally:
        mesh.close()

    out: List[Face] = []
    for slot in range(num_faces):
        s = series[slot]
        if np.isnan(s).all():
            continue
        s = _fill_nan(s)
        cx = float(np.mean(centers[slot])) if centers[slot] else 0.5
        out.append((cx, _standardize(s)))
    return out or None


def roi_motion_series(frames: np.ndarray, num_regions: int) -> List[Face]:
    """Lower-face frame-difference motion per vertical strip (fallback)."""
    frames = np.asarray(frames)
    t, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    gray = frames.mean(axis=3) if frames.ndim == 4 else frames.astype(float)
    lower = gray[:, h // 2:, :]  # mouth is in the lower face
    strip = w / num_regions
    out: List[Face] = []
    for r in range(num_regions):
        x0, x1 = int(r * strip), int((r + 1) * strip)
        region = lower[:, :, x0:x1]
        diff = np.zeros(t)
        diff[1:] = np.mean(np.abs(np.diff(region, axis=0)), axis=(1, 2))
        cx = (r + 0.5) / num_regions
        out.append((cx, _standardize(diff)))
    return out


def _fill_nan(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    idx = np.where(~np.isnan(x))[0]
    if idx.size == 0:
        return np.zeros_like(x)
    x[:idx[0]] = x[idx[0]]
    x[idx[-1] + 1:] = x[idx[-1]]
    for i in range(len(x)):
        if np.isnan(x[i]):
            x[i] = x[i - 1]
    return x


def extract_face_motions(frames: np.ndarray, num_speakers: int) -> List[Face]:
    """Try mediapipe (real faces); fall back to ROI motion. Sorted by center_x."""
    faces = mediapipe_mouth_series(frames, num_speakers)
    if not faces:
        faces = roi_motion_series(frames, num_speakers)
    faces.sort(key=lambda f: f[0])
    return faces


def _self_test() -> int:
    # Two strips; a bright block flickers in the lower half of the left strip
    # early and the right strip late -> two distinct motion series.
    rng = np.random.default_rng(0)
    t, h, w = 120, 64, 64
    frames = (rng.integers(0, 20, size=(t, h, w, 3))).astype(np.uint8)
    for fi in range(t):
        if 20 <= fi < 60 and fi % 2 == 0:        # left mouth moving early
            frames[fi, h // 2:, :w // 2, :] = 220
        if 70 <= fi < 110 and fi % 2 == 0:       # right mouth moving late
            frames[fi, h // 2:, w // 2:, :] = 220

    faces = roi_motion_series(frames, 2)
    assert len(faces) == 2
    left, right = faces[0][1], faces[1][1]
    # Standardized motion is high (positive) during that strip's active window.
    assert left[20:60].mean() > left[70:110].mean()
    assert right[70:110].mean() > right[20:60].mean()
    print(f"left cx={faces[0][0]:.2f} right cx={faces[1][0]:.2f}")
    print("All lipmotion self-tests passed (ROI path).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
