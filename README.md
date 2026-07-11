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
- [ ] Phase 3, OR-PIT training pipeline built and verified on GPU (warm-started SepFormer at ~12 to 14 dB SI-SNRi); full fine-tuning run in progress
- [ ] Phase 4, unknown speaker count
- [ ] Phase 5, real-world robustness
- [ ] Phase 6, addons (demo, transcription, audio-visual)
- [ ] Phase 7, evaluation and report

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
