# VoxSplit — Multi-Speaker Speech Separation

**Goal:** Given a single audio recording where ≥3 people speak simultaneously, output one clean audio track per speaker. Evaluated on hidden test inputs at multiple difficulty levels (by concurrent speaker count) and separation quality.

> Suggested project name: **VoxSplit** (voices, split apart). Rename the folder to `VoxSplit` before creating the GitHub repo, or pick your own — nothing in the code depends on the folder name. The conda env is already named `voxsplit`.

---

## Our machine (compute available)

| Component | Spec | Implication |
|---|---|---|
| GPU | **NVIDIA RTX 5070 Ti, 16 GB VRAM** (Blackwell / sm_120) | Strong for training. **Must use PyTorch cu128 wheels** — older cu124/cu126 lack Blackwell kernels |
| CPU | Intel Core Ultra 7 265K, 20 cores | Fast data generation / mixing |
| RAM | 32 GB | Fine |
| OS | Windows 11 Pro | We train on the **Windows side**, in this repo folder (easier version control). WSL not needed |
| CUDA (system) | Toolkit 12.6, driver CUDA 13.3 | PyTorch bundles its own runtime — system CUDA version doesn't matter for pip torch |
| Python | conda env `voxsplit` = Python 3.10 | |
| Tools present | git, conda, WSL2 (Ubuntu) | |
| Tools to add | `ffmpeg` (into env), `gh` CLI (optional, for GitHub) | |

**16 GB VRAM budget:** plenty for inference of any pretrained model; for training use 8 kHz + gradient accumulation / smaller batch for the big transformers (SepFormer, MossFormer2). Conv-TasNet and 8 kHz SepFormer fine-tune comfortably.

---

## 0. Research Summary (read this first)

### The reference paper: "Looking to Listen" (Google, 2018)
- Paper: https://arxiv.org/pdf/1804.03619 — audio-**visual** model: it uses the speaker's **face video** (face embeddings from 75 frames per 3s clip) + audio spectrogram to predict complex spectrogram masks per speaker.
- **Google never released official code or model weights.** Only the AVSpeech dataset metadata (YouTube URLs + timestamps + face crops) was released: https://looking-to-listen.github.io/avspeech/
- Open reimplementations exist but are **old (TF1/Keras, ~2019) and unmaintained**:
  - https://github.com/bill9800/speech_separation (audio-only + audio-visual variants)
  - https://github.com/JusperLee/Looking-to-Listen-at-the-Cocktail-Party (full pipeline: yt download, MTCNN face detection, preprocessing)
  - Useful as *reading material* for the pipeline design, not as a base to build on.

### Critical insight for OUR project
The seniors' evaluation is on **audio inputs** ("take an audio input which has the problem of multiple speakers"). The Looking-to-Listen approach *requires video of every speaker's face* — it cannot run on audio-only test inputs. Also: L2L separates one speaker per visible face; their test levels are by speaker count.

➜ **Our core system must be AUDIO-ONLY speech separation.** The field has moved far past 2018; modern audio-only models now beat the 2018 audio-visual results on standard benchmarks. Audio-visual can be an **addon** (bonus points) for inputs that do come with video.

### Modern landscape (2024–2026)
| Model | Type | Notes |
|---|---|---|
| Conv-TasNet (2019) | time-domain | Classic, easy to train, great learning baseline |
| DPRNN / DPTNet | time-domain | Dual-path processing of long sequences |
| **SepFormer** (SpeechBrain) | transformer | **Pretrained 3-speaker checkpoints exist** (`speechbrain/sepformer-wsj03mix`, ~19.8 dB SI-SNRi; `speechbrain/sepformer-libri3mix`) |
| **MossFormer2** (Alibaba) | hybrid | Current top-tier; in ClearerVoice-Studio toolkit: https://github.com/modelscope/ClearerVoice-Studio |
| TF-GridNet | time-frequency | SOTA in noisy/reverberant conditions |
| RTFS-Net / IIANet / CTCNet (JusperLee) | audio-visual | Modern AV separation, PyTorch, maintained — the modern successors of Looking-to-Listen |

### Key technical concepts to learn (Phase 0)
- STFT / spectrograms, time-domain vs T-F masking
- **Permutation Invariant Training (PIT)** — the core trick: model outputs N streams, loss takes the best output↔target permutation
- **SI-SDR / SI-SDRi** — the main metric (also PESQ, STOI, ESTOI)
- Why speaker count matters: a model with N output heads is trained for exactly N speakers → handling *unknown* count is its own problem (Phase 4) and likely the differentiator in evaluation.

### Datasets (audio-only — no need for the painful AVSpeech download)
- **LibriSpeech** (free) → generate **LibriMix**-style mixtures for 2/3/4/5 speakers: https://github.com/JorisCos/LibriMix (official scripts support Libri2Mix/Libri3Mix; extendable to more speakers with custom mixing)
- **WHAM! noise** corpus for realistic background noise; add reverb via pyroomacoustics/RIRs
- WSJ0-mix is the classic benchmark but **WSJ0 is paid (LDC license)** — skip it, use LibriMix
- AVSpeech only needed if we do the audio-visual addon (it's YouTube-scraping, ~1700h, flaky downloads — big time sink)

---

## Phase 0 — Foundations & environment (≈ week 1)
- [x] Create conda env `voxsplit` (Python 3.10)
- [x] Install PyTorch **cu128** + torchaudio (Blackwell GPU requirement) — torch 2.11.0+cu128
- [x] Install `ffmpeg` into the env
- [x] Install core stack: `speechbrain`, `librosa`, `soundfile`, `pesq`, `pystoi`, `mir_eval`, `fast_bss_eval`, `matplotlib`, `numpy`, `scipy`, `tqdm`
- [x] Verify `torch.cuda.is_available()` is True and reports the RTX 5070 Ti (sm_120 kernels present, matmul on GPU passes — run `python src/check_env.py`)
- [x] Decide training location: **local RTX 5070 Ti** (primary) — Kaggle/Colab as backup for big runs
- [x] Repo scaffolding: `src/` layout, `.gitignore`, `requirements.txt`, `README.md`; git initialized
- [ ] Team study session: STFT, masking, PIT, SI-SDR. Read L2L + SepFormer + Conv-TasNet papers (skim ok)  ← *your task while reading the Research Summary*
- [ ] Experiment logging choice (W&B free tier or CSV) — decide in Phase 1
- [ ] **Deliverable:** everyone can load a wav, compute a spectrogram, and compute SI-SDR between two signals

## Phase 1 — Working baseline (pretrained, zero training) — week 1–2
Get an end-to-end system running immediately — this de-risks the whole project.
- [ ] Run `speechbrain/sepformer-wsj03mix` (3-speaker, 8 kHz) on a synthetic 3-speaker mixture
- [ ] Also try `speechbrain/sepformer-libri3mix` (noisier training → more robust) and ClearerVoice-Studio MossFormer2 separators (8k/16k)
- [ ] Build `make_mixture.py`: pick K random LibriSpeech utterances, loudness-normalize (random SNR −5..+5 dB), sum, save mix + references. Doubles as test-set + training-data generator
- [ ] Build `evaluate.py`: (mix, references, estimates) → SI-SDRi, PESQ, STOI with best-permutation matching. Listen to outputs
- [ ] **Deliverable:** `separate.py input.wav --out_dir out/` producing per-speaker wavs + a metrics table for 2/3-speaker mixtures

## Phase 2 — Data pipeline — week 2–3 (overlaps Phase 1)
- [ ] Download LibriSpeech `train-clean-100` + `train-clean-360`, `dev-clean`, `test-clean`
- [ ] Generate Libri2Mix + Libri3Mix via official LibriMix scripts
- [ ] Write our own extension for **4- and 5-speaker mixtures** (sample K speakers, gain-normalize, sum, min/max-length variants)
- [ ] Variants per count: clean / +WHAM noise / +reverb (pyroomacoustics). 8 kHz first (fast), 16 kHz later
- [ ] Freeze held-out test sets per level (2,3,4,5 spk × clean/noisy) — never train on these
- [ ] **Deliverable:** `data/` with reproducible generation scripts + frozen eval sets

## Phase 3 — Train our own models — week 3–6 (the core)
- [ ] Warm-up: train Conv-TasNet on Libri2Mix (Asteroid recipe) — verifies pipeline in ~1 GPU-day, teaches PIT
- [ ] Main: fine-tune pretrained SepFormer (libri3mix) on our Libri3Mix
- [ ] Train **4-speaker and 5-speaker models** (fresh output heads, warm-start encoder/masknet from 3-spk checkpoint)
- [ ] If compute allows: MossFormer2 or TF-GridNet recipe for the 3-spk level (best quality/param 2025)
- [ ] Loss: SI-SDR with utterance-level PIT; track SI-SDRi per level on frozen sets
- [ ] **Deliverable:** model bank `{2spk, 3spk, 4spk, 5spk}`, each beating the pretrained baseline on its level

## Phase 4 — Unknown speaker count — week 5–7 ⭐ likely the evaluation differentiator
Test inputs won't announce how many speakers there are.
- [ ] **Speaker-count classifier (do first):** small CNN/CRNN on log-mel → predict K ∈ {1..5}; route to the K-speaker model. Training labels are free from our generator
- [ ] **Max-N + silence detection (fallback):** run 5-spk model, drop low-energy/no-speech channels; train 5-spk model with some lower-K mixtures (extra targets = silence)
- [ ] **Recursive one-vs-rest separation (stretch):** peel one speaker at a time until residual has no speech (Takahashi 2019; coarse-to-fine arXiv:2203.16054). Scales to arbitrary K — great "as many speakers as possible" story
- [ ] Cross-check counts from classifier vs silence-detection; reconcile (e.g. take max)
- [ ] **Deliverable:** single entry point `separate.py input.wav` working with no prior knowledge of speaker count

## Phase 5 — Robustness & real-world inputs — week 6–8
- [ ] Input normalization: resample to model rate, downmix to mono, loudness-normalize
- [ ] **Long-audio handling:** chunk into ~10 s windows with overlap; overlap-add with **permutation alignment** across chunks (correlate overlap regions and/or ECAPA-TDNN speaker embeddings + clustering — SpeechBrain has pretrained ECAPA). Prevents identity swaps mid-file
- [ ] Noise/reverb robustness: fine-tune on noisy/reverberant variants from Phase 2
- [ ] Optional polish: pass each separated stream through a speech-enhancement model (ClearerVoice FRCRN / MossFormer2-SE) to clean residual bleed
- [ ] Test on real-world audio: record ourselves talking simultaneously; YouTube podcast clips with crosstalk
- [ ] **Deliverable:** robust CLI handling arbitrary real recordings end-to-end

## Phase 6 — Addons (bonus, parallel from week 6 if time permits)
- [ ] **Web demo (highest impact, ~2 days):** Gradio app — upload audio → per-speaker players + before/after spectrograms
- [ ] **Per-speaker transcription:** run Whisper on each separated track → speaker-attributed transcript
- [ ] **Speaker diarization overlay:** pyannote to show speaking timelines
- [ ] **Audio-visual mode (the L2L homage):** for video inputs, run a modern pretrained AV model — RTFS-Net (https://github.com/spkgyk/RTFS-Net, ICLR 2024) or IIANet/CTCNet (https://github.com/JusperLee) — with face detection + mouth-crop pipeline. Only after Phases 1–5 are solid; the AVSpeech/LRS data pipeline is a major time sink

## Phase 7 — Evaluation, report, presentation (final 1–2 weeks)
- [ ] Full metric sweep: SI-SDRi / PESQ / STOI per level (2→5+ spk, clean/noisy), baseline vs our models — plot quality vs speaker count
- [ ] Ablations: pretrained vs fine-tuned; count-classifier accuracy; chunk-stitching on/off
- [ ] Failure analysis: same-gender similar voices, heavy noise, >5 speakers
- [ ] README + report + live demo; rehearse with a surprise mixture made by a teammate

---

## Suggested repo layout
```
VoxSplit/
├── data/               # generation scripts + manifests (not raw audio)
├── src/
│   ├── mixing/         # mixture generation (2..N speakers, noise, reverb)
│   ├── models/         # training / fine-tuning recipes
│   ├── inference/      # separate.py, chunking, stitching, count estimation
│   └── eval/           # metrics, permutation matching, reports
├── demo/               # Gradio app
├── experiments/        # configs + results logs
├── requirements.txt
└── PLAN.md
```

## Risks & mitigations
| Risk | Mitigation |
|---|---|
| Not enough GPU compute | 16 GB 5070 Ti is plenty; Phase 1 pretrained baseline already meets "≥3 speakers" with zero training; train at 8 kHz; Kaggle/Colab backup |
| 4–5+ speaker quality collapses (SOTA 3-spk ≈ 20 dB, 5-spk ≈ 10 dB — true for everyone) | Expected; show graceful degradation + recursive method story |
| Unknown test conditions (noise, length, format) | Phase 5 robustness; test on real recordings early |
| Deadline "coming soon" | Phases 1–2 give a submittable system within ~2 weeks; everything after improves it incrementally |

## Key references
- Looking to Listen: paper https://arxiv.org/pdf/1804.03619 · site https://looking-to-listen.github.io/ · dataset https://looking-to-listen.github.io/avspeech/
- Reimplementations: https://github.com/bill9800/speech_separation · https://github.com/JusperLee/Looking-to-Listen-at-the-Cocktail-Party
- SpeechBrain SepFormer 3-spk: https://huggingface.co/speechbrain/sepformer-wsj03mix · https://huggingface.co/speechbrain/sepformer-libri3mix
- ClearerVoice-Studio (MossFormer2): https://github.com/modelscope/ClearerVoice-Studio
- LibriMix generation: https://github.com/JorisCos/LibriMix
- Asteroid toolkit: https://github.com/asteroid-team/asteroid
- Unknown-count separation: Takahashi 2019 (recursive), Coarse-to-Fine recursive https://arxiv.org/abs/2203.16054
- Modern AV separation: https://github.com/spkgyk/RTFS-Net · https://github.com/JusperLee/IIANet · https://github.com/JusperLee/CTCNet
- Paper/code index for the whole field: https://github.com/gemengtju/Tutorial_Separation
