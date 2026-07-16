# VoxSplit, Multi-Speaker Speech Separation

Take an audio recording where **3 or more people speak at the same time** and return a
clean, separate audio track for **each speaker**. (The "cocktail party problem.")

Summer project inspired by Google's [*Looking to Listen*](https://looking-to-listen.github.io/),
but built **audio-only** with modern (2024 to 2026) separation models, since the
evaluation is on audio inputs. See **[PLAN.md](PLAN.md)** for the full research
summary and phase-by-phase roadmap.

## Status
- [x] Phase 0, environment and GPU verified
- [x] Phase 1, pretrained baselines compared. Best 3-spk: SepFormer libri3mix (mean SI-SDRi about 18.6 dB). Best 2-spk: MossFormer2 (about 20.2 dB). See PLAN.md for the table
- [x] Phase 2, frozen eval set (2 to 5 spk, 80 mixtures) plus WHAM-noise, reverb, and 16 kHz variants, all reproducible from committed manifests
- [x] Phase 3, OR-PIT fine-tuning: converged 20000-step run. One 2-head model separates 2 to 5 speakers, 2-spk direct at 19.25 dB SI-SDRi (beats the 17.35 dB baseline) and 3/4/5-spk via recursion at 15.62 / 8.50 / 5.96 dB (oracle head selection; blind selection is Phase 4). Benchmarked against fixed-N uPIT baselines (4-spk 15.16, 5-spk 11.01 dB, both trained via masknet head expansion) and a from-scratch TF-GridNet (3-spk, learning-scale). Optional W&B logging wired in (off by default). See PLAN.md
- [x] Phase 4, unknown speaker count: blind recursive OR-PIT driven by a count/stop classifier. One model, no count supplied: 0.71 count accuracy on the frozen 2-5 set, and separation matches the oracle recursion when the count is right. The stop classifier only worked once trained on the OR-PIT model's own residuals (naive clean-trained version scored 0.25). Entry point `src/inference/separate_unknown.py`. See PLAN.md
- [x] Phase 5, real-world robustness: input normalization, long-audio chunking with ECAPA identity stitching (Hungarian one-to-one matching across chunks), and a noise/reverb-augmented fine-tune. Robustness fine-tune lifts the degraded 2-spk score from 6.68 to 11.44 dB (noise) and 4.04 to 7.06 dB (reverb) with clean retained (19.02). End-to-end CLI `src/inference/separate_longform.py` handles arbitrary recordings (any rate/length, unknown count). ECAPA stitching is reliable for distinct voices, weaker for similar ones. See PLAN.md
- [x] Phase 6, addons: Gradio web demo (`demo/app.py`) with per-speaker players, before/after spectrograms, Whisper transcripts, and a speaking timeline; auto-detected count. Per-speaker transcription via faster-whisper (`src/inference/transcribe.py`); diarization timelines from our own separated tracks via energy VAD (`src/inference/timeline.py`, no gated pyannote). **Audio-visual mode** (`src/av/`): the video gives the speaker count and each track is matched to its on-screen speaker by lip-motion↔audio correlation; validated on a synthetic talking-face clip (correct face assignment at ~19.7 dB). See PLAN.md
- [x] Phase 7, evaluation and report: full metric sweep, ablations, and failure analysis consolidated in **[REPORT.md](REPORT.md)** and auto-generated **[experiments/RESULTS.md](experiments/RESULTS.md)** (regenerate with `python experiments/make_report.py`). Plots in `experiments/plots/`

## Results (headline)

Frozen eval set (LibriSpeech test-clean, 20 mixtures/level, 8 kHz, SI-SDRi dB):

| Level | Best fixed-N | OR-PIT blind (unknown count) | Blind count acc |
|---|---|---|---|
| 2 spk | 19.25 | 19.25 | 1.00 |
| 3 spk | 19.33 | 16.09 | 0.70 |
| 4 spk | 15.16 | 9.09 | 0.40 |
| 5 spk | 11.01 | 7.23 | 0.75 |

One model separates 2–5 speakers with **no count supplied** (0.71 overall count
accuracy). Robustness fine-tune: noise 6.68→11.44, reverb 4.04→7.06 dB. Full
tables, ablations, failure analysis, and plots in [REPORT.md](REPORT.md).

## Demo

```powershell
conda activate voxsplit
python demo/app.py            # http://localhost:7860
```
Two tabs: **Audio** (upload a recording) and **Video (audio-visual)** (upload a
video — on-screen faces set the count and lip motion assigns each track to its
speaker). Audio tab auto-detects the count, with a "Speaker count" force option
and a "Split sensitivity" nudge. Per speaker: player, spectrogram, transcript,
plus a speaking timeline.

## Machine
NVIDIA RTX 5070 Ti (16 GB, Blackwell/sm_120), Intel Core Ultra 7 265K, 32 GB RAM, Windows 11.

## Setup
```powershell
# 1. Create the environment
conda create -y -n voxsplit python=3.10
conda activate voxsplit

# 2. PyTorch: you MUST use the cu128 index for the Blackwell GPU.
#    Install the latest stable torch/torchaudio from that index, then verify in step 4.
#    Do not hard-pin a version blindly. Confirmed working on this machine: torch 2.11.0+cu128.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. ffmpeg and SoX (SoX is required by LibriMix generation in Phase 2), plus the rest
conda install -y -c conda-forge ffmpeg sox
pip install -r requirements.txt

# 4. Verify (should report CUDA True, sm_120 kernels, and a passing matmul)
python src/check_env.py
```

## Layout
```
src/
  mixing/      # mixture generation (2..N speakers, noise, reverb)
  models/      # training and fine-tuning recipes
  inference/   # separate.py, chunking, stitching, speaker-count estimation
  eval/        # SI-SDR, PESQ, STOI, permutation matching
  check_env.py # environment and GPU sanity check
data/          # dataset manifests and generation scripts (audio is gitignored)
demo/          # Gradio web demo
experiments/   # configs and result logs
```
