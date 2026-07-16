# Audio-visual mode (scaffold) — the *Looking to Listen* homage

**Status: scaffold + plan, not a validated pipeline.** VoxSplit's evaluation is
audio-only, so AV mode is a genuine bonus. It is scoped here with a runnable
skeleton (`separate_av.py`) and the exact steps to finish it, rather than a
half-working end-to-end that can't be tested in this environment.

## Why it isn't fully built here
Three hard dependencies, none available in this automated setup:
1. **A pretrained AV separator** — RTFS-Net (ICLR 2024, https://github.com/spkgyk/RTFS-Net),
   IIANet, or CTCNet (https://github.com/JusperLee). These are separate repos
   with their own configs/weights that must be cloned and reconciled with our
   torch cu128 env.
2. **A face-detection + mouth-crop pipeline** (e.g. mediapipe or MTCNN) to feed
   one visual stream per speaker.
3. **Video test data** with overlapping speech (AVSpeech / LRS2/3), which is a
   large, flaky, multi-day download the plan explicitly flags as a time sink.

## Architecture (implemented as interfaces in `separate_av.py`)
```
video ──► frame extraction (PyAV, implemented)
            │
            ├─► face detection + mouth crops per speaker  (interface: FaceCropper)
            │
audio ──►  AV separator: audio + per-speaker mouth crops ─► per-speaker audio
            (interface: AVSeparator)                          │
                                                              ▼
                                       falls back to VoxSplit's audio-only
                                       blind pipeline if no faces / no model
```
The scaffold reads video, defines the `FaceCropper` and `AVSeparator`
interfaces, and wires the audio-only fallback so the entry point works today on
any input; plugging in a real face detector and AV model completes it.

## Steps to finish
1. `pip install mediapipe` (or MTCNN); implement `FaceCropper.crop` to return
   per-face mouth-region clips aligned to the audio.
2. Clone RTFS-Net, load its pretrained LRS checkpoint, and implement
   `AVSeparator.separate(audio, face_clips)`.
3. Get a few LRS2 test clips; score AV vs our audio-only separation on the same
   clips (SI-SDRi). Expect AV to win when faces are visible and reliable.

Until then, `separate_av.py` runs the audio-only pipeline on the video's audio
track, which is the honest, working behavior.
