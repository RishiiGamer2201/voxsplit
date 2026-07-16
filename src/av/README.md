# Audio-visual mode — the *Looking to Listen* homage (working feature)

VoxSplit's AV mode uses the **video to drive separation**: the on-screen faces
give the speaker COUNT, and each separated audio track is matched to its
on-screen speaker by correlating the track's loudness with each face's mouth
motion. It runs entirely in our env, no external AV-separation model.

## What it does (`separate_av.py`)
```
video ─► frames + audio (PyAV)
          │
          ├─► face count + per-face mouth motion
          │     mediapipe FaceMesh (real faces) → inner-lip aperture
          │     └─ fallback: ROI motion per vertical strip (any video)
          │
audio ─► forced-count OR-PIT separation to #faces  ─► K tracks
                                                        │
        assign track ↔ face by lip-motion vs audio-envelope correlation
                       (Hungarian, one-to-one)  ─► speaker{face}.wav
```
The video determines both the **count** (no reliance on the audio-only count
classifier) and each track's **identity** (which on-screen person it is).

## Run it
```bash
# make a synthetic 2-speaker talking-face clip and separate it
python src/av/make_synth_av.py --out scratch/av_test.mp4
python src/av/separate_av.py scratch/av_test.mp4 --out-dir out/av --num-faces 2
```
On the synthetic clip the two tracks are assigned to the correct faces at
~19.7 / 19.6 dB SI-SDR (wrong-face assignment is ~-36 dB).

## Real videos
- **mediapipe legacy FaceMesh** gives precise per-face lip aperture and an
  automatic face count. Some mediapipe builds (e.g. 0.10.35 here) ship only the
  Tasks API and lack `mp.solutions`; the code detects this and falls back.
- **ROI-motion fallback** works on any video with speakers in distinct
  horizontal regions (typical side-by-side / panel layouts); pass `--num-faces
  N` since it can't count faces itself.

## Components (each self-tested)
- `av_assign.py` — audio-envelope ↔ face-motion correlation + Hungarian.
- `lipmotion.py` — mediapipe FaceMesh + ROI-motion extractors.
- `separate_av.py` — the end-to-end entry point.
- `make_synth_av.py` — renders the synthetic test clip.

## Heavier alternative (not built)
True AV *masking* — a pretrained AV separator (RTFS-Net, ICLR 2024;
IIANet/CTCNet) that fuses mouth crops into the separation itself — would beat
audio-only when faces are reliable, but needs an external repo, its weights, and
video training data (AVSpeech/LRS). Our approach is AV *assignment*: it uses the
audio-only separator (which is strong) and lets the video resolve count and
identity, which is the part video helps most and runs with no external model.
