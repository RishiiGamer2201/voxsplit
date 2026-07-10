# VoxSplit, Multi-Speaker Speech Separation

**Goal:** Given a single audio recording where 3 or more people speak simultaneously, output one clean audio track per speaker. Evaluated on hidden test inputs at multiple difficulty levels (by concurrent speaker count) and separation quality.

> Suggested project name: **VoxSplit** (voices, split apart). Nothing in the code depends on the folder name. The conda env is named `voxsplit`.

---

## Our machine (compute available)

| Component | Spec | Implication |
|---|---|---|
| GPU | NVIDIA RTX 5070 Ti, 16 GB VRAM (Blackwell / sm_120) | Strong for training. Must use PyTorch cu128 wheels; older cu124/cu126 lack Blackwell kernels |
| CPU | Intel Core Ultra 7 265K, 20 cores | Fast data generation and mixing |
| RAM | 32 GB | Fine |
| OS | Windows 11 Pro | We train on the Windows side, in this repo folder (easier version control). WSL not needed |
| CUDA (system) | Toolkit 12.6, driver CUDA 13.3 | PyTorch bundles its own runtime, so the system CUDA version does not matter for pip torch |
| Python | conda env `voxsplit` = Python 3.10 | |
| Tools present | git, conda, WSL2 (Ubuntu) | |
| Tools to add | `ffmpeg` and `sox` (into env; SoX is required by LibriMix generation), `gh` CLI (optional, for GitHub) | |

**16 GB VRAM budget:** plenty for inference of any pretrained model. For training, use 8 kHz plus gradient accumulation or smaller batch for the big transformers (SepFormer, MossFormer2). Conv-TasNet and 8 kHz SepFormer fine-tune comfortably.

---

## 0. Research Summary (read this first)

### The reference paper: "Looking to Listen" (Google, 2018)
- Paper: https://arxiv.org/pdf/1804.03619 . This is an audio-visual model: it uses the speaker's face video (face embeddings from 75 frames per 3s clip) plus the audio spectrogram to predict complex spectrogram masks per speaker.
- Google never released official code or model weights. Only the AVSpeech dataset metadata (YouTube URLs, timestamps, face crops) was released: https://looking-to-listen.github.io/avspeech/
- Open reimplementations exist but are old (TF1/Keras, roughly 2019) and unmaintained:
  - https://github.com/bill9800/speech_separation (audio-only and audio-visual variants)
  - https://github.com/JusperLee/Looking-to-Listen-at-the-Cocktail-Party (full pipeline: yt download, MTCNN face detection, preprocessing)
  - Useful as reading material for the pipeline design, not as a base to build on.

### Critical insight for OUR project
The seniors' evaluation is on audio inputs ("take an audio input which has the problem of multiple speakers"). The Looking-to-Listen approach requires video of every speaker's face, so it cannot run on audio-only test inputs. Also, L2L separates one speaker per visible face, whereas their test levels are by speaker count.

So: our core system must be AUDIO-ONLY speech separation. The field has moved far past 2018; modern audio-only models now beat the 2018 audio-visual results on standard benchmarks. Audio-visual can be an addon (bonus points) for inputs that do come with video.

### Modern landscape (2024 to 2026)
| Model | Type | Notes |
|---|---|---|
| Conv-TasNet (2019) | time-domain | Classic, easy to train, great learning baseline |
| DPRNN / DPTNet | time-domain | Dual-path processing of long sequences |
| SepFormer (SpeechBrain) | transformer | Pretrained 3-speaker checkpoints exist (`speechbrain/sepformer-wsj03mix`, about 19.8 dB SI-SNRi; `speechbrain/sepformer-libri3mix`) |
| MossFormer2 (Alibaba) | hybrid | Current top-tier; in ClearerVoice-Studio toolkit: https://github.com/modelscope/ClearerVoice-Studio |
| TF-GridNet | time-frequency | SOTA in noisy/reverberant conditions |
| RTFS-Net / IIANet / CTCNet (JusperLee) | audio-visual | Modern AV separation, PyTorch, maintained. The modern successors of Looking-to-Listen |

### Key technical concepts to learn (Phase 0)
- STFT and spectrograms, time-domain versus time-frequency masking
- Permutation Invariant Training (PIT), the core trick: the model outputs N streams, and the loss takes the best output-to-target permutation
- SI-SDR and SI-SDRi, the main metric (also PESQ, STOI, ESTOI)
- Why speaker count matters: a model with N output heads is trained for exactly N speakers, so handling an unknown count is its own problem (Phase 4) and likely the differentiator in evaluation.

### Datasets (audio-only, no need for the painful AVSpeech download)
- LibriSpeech (free), used to generate LibriMix-style mixtures for 2, 3, 4, 5 speakers: https://github.com/JorisCos/LibriMix (official scripts support Libri2Mix and Libri3Mix; extendable to more speakers with custom mixing)
- WHAM! noise corpus for realistic background noise; add reverb via pyroomacoustics or RIRs
- WSJ0-mix is the classic benchmark, but WSJ0 is paid (LDC license), so skip it and use LibriMix
- AVSpeech is only needed if we do the audio-visual addon (it is YouTube-scraping, roughly 1700h, flaky downloads, a big time sink)

---

## Phase 0, Foundations and environment (about week 1)
- [x] Create conda env `voxsplit` (Python 3.10)
- [x] Install PyTorch cu128 and torchaudio (Blackwell GPU requirement), torch 2.11.0+cu128
- [x] Install `ffmpeg` and `sox` into the env (SoX is required by LibriMix generation in Phase 2)
- [x] Install core stack: `speechbrain`, `librosa`, `soundfile`, `pesq`, `pystoi`, `mir_eval`, `fast_bss_eval`, `matplotlib`, `numpy`, `scipy`, `tqdm`
- [x] Verify `torch.cuda.is_available()` is True and reports the RTX 5070 Ti (sm_120 kernels present, matmul on GPU passes; run `python src/check_env.py`)
- [x] Decide training location: local RTX 5070 Ti (primary), with Kaggle/Colab as backup for big runs
- [x] Repo scaffolding: `src/` layout, `.gitignore`, `requirements.txt`, `README.md`; git initialized
- [ ] Team study session: STFT, masking, PIT, SI-SDR. Read L2L, SepFormer, and Conv-TasNet papers (skim ok). This is your task while reading the Research Summary
- [x] Experiment logging decided: start with simple CSV logs in Phase 1 (no extra dependency), optionally add Weights & Biases (free tier) when training starts in Phase 3
- [ ] Deliverable: everyone can load a wav, compute a spectrogram, and compute SI-SDR between two signals

## Phase 1, Working baseline (pretrained, zero training), week 1 to 2
Get an end-to-end system running immediately. This de-risks the whole project.
- [x] Run `speechbrain/sepformer-wsj03mix` (3-speaker, 8 kHz) on a synthetic 3-speaker mixture
- [x] Also tried `speechbrain/sepformer-libri3mix` and ClearerVoice-Studio MossFormer2_SS_16K (see comparison table below)
- [x] Build `make_mixture.py`: pick K random LibriSpeech utterances, loudness-normalize (random SNR minus 5 to plus 5 dB), sum, save mix plus references. Doubles as test-set and training-data generator
- [x] Build `evaluate.py`: given (mix, references, estimates), compute SI-SDRi, PESQ, STOI with best-permutation matching. Listen to outputs
- [x] Deliverable: `separate.py input.wav --out-dir out/` producing per-speaker wavs plus a metrics table.

### Phase 1 model comparison (LibriSpeech dev-clean, 3 mixtures per model, scored at 8 kHz)

3-speaker mixtures:
| Model | mean SI-SDRi (dB) | PESQ | STOI |
|---|---|---|---|
| sepformer-wsj03mix | 15.7 | 2.58 | 0.91 |
| sepformer-libri3mix | **18.6** | **2.99** | **0.94** |

2-speaker mixtures:
| Model | mean SI-SDRi (dB) | PESQ | STOI |
|---|---|---|---|
| sepformer-wsj02mix | 17.2 | 3.44 | 0.97 |
| MossFormer2_SS_16K | **20.2** | **3.77** | **0.98** |

Takeaways:
- For our 3-speaker target, **sepformer-libri3mix is the best pretrained baseline** (domain match with LibriSpeech). Use it as the Phase 1 default.
- MossFormer2_SS_16K is the strongest model overall but is **2-speaker only**, so it cannot separate 3-or-more speaker inputs directly. Its 2-output "one and rest" shape makes it a natural backbone for the Phase 4 OR-PIT recursion.
- clearvoice runs in a separate conda env (its dependencies differ and it pulls its own CPU torch). See `src/inference/separate_mossformer2.py`. On this machine the conda-forge ffprobe is broken, so that script reads audio via soundfile instead of pydub.
- Raw per-mixture numbers are logged in `experiments/phase1_results.csv` (3-spk) and `experiments/phase1_2spk_results.csv` (2-spk).

## Phase 2, Data pipeline, week 2 to 3 (overlaps Phase 1)
- [x] Install SoX (done in Phase 0)
- [x] Download LibriSpeech `dev-clean` and `test-clean` (40 speakers each, disjoint sets). `train-clean-100/360` deferred until Phase 3 training needs them
- [x] Frozen clean eval set built with our own generator: 8 kHz, speaker counts 2, 3, 4, 5, 20 mixtures each (80 total), sampled from test-clean so eval speakers stay unseen by future training
- [x] Manifest-first, reproducible design: `src/data/build_eval_set.py` writes a small committed JSON manifest (relative source paths, speaker ids, exact linear gains); `src/data/realize_eval_set.py` regenerates byte-identical audio from it. Audio is gitignored, the manifest is committed, so the eval set is fully reproducible without storing audio in git
- [x] 4 and 5-speaker mixtures included. NOTE: these are our own custom dataset, NOT a standard published benchmark. The committed manifest keeps them reproducible
- [x] Batch tooling: `src/inference/separate_set.py` (loads a model once, separates a whole level) and `src/eval/evaluate_set.py` (scores and aggregates by speaker count into `experiments/eval_set_results.csv`)
- [ ] Still to do: add WHAM noise and reverb (pyroomacoustics) variants, then a 16 kHz set. Only after the clean path, which is done
- [x] Deliverable (clean): `data/eval_manifest.json` committed, reproducible generation and batch scoring scripts in place

### Frozen eval set baselines (test-clean, 20 mixtures per level, scored at 8 kHz)
| Level | Model | SI-SDRi (dB) | PESQ | STOI |
|---|---|---|---|---|
| 2 spk | sepformer-wsj02mix | 17.35 | 3.37 | 0.96 |
| 3 spk | sepformer-libri3mix | 19.33 | 3.20 | 0.93 |
| 4 spk | no pretrained model yet (Phase 3) | n/a | n/a | n/a |
| 5 spk | no pretrained model yet (Phase 3) | n/a | n/a | n/a |

These are the numbers every future model must beat on the same frozen mixtures. Per-level rows accumulate in `experiments/eval_set_results.csv`.

## Phase 3, Train our own models, week 3 to 6 (the core)
- [ ] Priority: fine-tune pretrained models (SepFormer, and ClearerVoice/MossFormer2) rather than training from scratch. For the time we have, fine-tuning gives the best results
- [ ] Optional learning warm-up: train Conv-TasNet on Libri2Mix (Asteroid recipe) to understand PIT end-to-end in about 1 GPU-day. Skip if short on time
- [ ] Baseline track: fine-tune pretrained SepFormer (libri3mix) on our Libri3Mix with standard utterance-level PIT and fixed 3 outputs. This is the comparison point, not the final system
- [ ] Headline track, SepFormer + OR-PIT (the team's proposal, and the right call). Retrain SepFormer with a 2-head "one and rest" output: head 1 is one speaker, head 2 is the sum of all remaining speakers.
  - IMPORTANT: OR-PIT is a training objective plus an output topology, NOT an inference-time switch. The pretrained uPIT checkpoints cannot do it as-is; this track requires real training
  - [ ] Warm-start from `speechbrain/sepformer-wsj02mix`, which already has exactly 2 output heads. Redefine target 2 as the residual mixture instead of "speaker 2"
  - [ ] OR-PIT loss: choose the permutation (which speaker is the "one") that minimizes SI-SDR loss on head 1, with head 2 scored against the sum of the rest
  - [ ] Train on a mix of 2 and 3-speaker data so recursion depth varies. Takahashi et al. trained on 2 and 3 speakers and generalized to 4 (arXiv:1904.03065)
  - [ ] Precedent: transformer/SepFormer-style extraction blocks trained with OR-PIT are published, so this is not unexplored territory (Deflationary Extraction Transformer, https://doi.org/10.3390/s25164905)
- [ ] Optional fixed-N comparison models: 4-speaker and 5-speaker uPIT models (fresh output heads, warm-start from the 3-spk checkpoint), purely to benchmark OR-PIT against
- [ ] If compute allows: MossFormer2 or TF-GridNet recipe for the 3-spk level (best quality per parameter as of 2025)
- [ ] Loss and tracking: SI-SDR; track SI-SDRi per level on the frozen sets
- [ ] VRAM note: SepFormer is heavy. On 16 GB use 8 kHz, short segments (about 3 s), small batch plus gradient accumulation
- [ ] Optionally add Weights & Biases (free tier) for run tracking now that training is underway; keep the CSV logs as the source of truth
- [ ] Deliverable: one OR-PIT SepFormer that separates 2 to 5 speakers, plus fixed-N baselines to compare it against

## Phase 4, Unknown speaker count, week 5 to 7 (the likely scoring edge, treat as a first-class goal)
Test inputs will not announce how many speakers there are. This is probably where the evaluation is won or lost. Both the external review and the team's own paper survey converge on recursive separation as the primary route, so that is what we build.
- [ ] PRIMARY, recursive OR-PIT separation using the Phase 3 model: extract the most dominant speaker, feed the residual back into the same network, repeat. Recursion depth adapts to the unknown speaker count, so one model covers every level
- [ ] Stopping criterion: a binary classifier on the residual deciding "speech remains" versus "noise only". This is the single highest-risk component. A false positive keeps recursing and injects noise; a false negative silently drops a real speaker. Budget real time for it
- [ ] Speaker count then comes for free from the recursion depth. Takahashi et al. report this is MORE accurate than estimating the count in advance with a separate classifier (arXiv:1904.03065)
- [ ] Secondary, speaker-count classifier: small CNN/CRNN on log-mel predicting K in {1..5}, routing to a fixed-N model. Build it as a cross-check and an ablation for the report, not as the main path
- [ ] Secondary, max-N plus silence detection: run the 5-spk model and drop low-energy or no-speech channels
- [ ] Known OR-PIT weaknesses we must measure and report honestly: error propagation (a bad early pass contaminates every later pass), no parallelism (passes are sequential, so inference is slow), and sensitivity to the stop classifier. Takahashi notes fine-tuning the recursion helps
- [ ] Compare all three on the frozen sets (count accuracy and separation quality), then pick or ensemble. Cross-check counts and reconcile disagreements
- [ ] Deliverable: single entry point `separate.py input.wav` working with no prior knowledge of speaker count

## Phase 5, Robustness and real-world inputs, week 6 to 8
- [ ] Input normalization: resample to model rate, downmix to mono, loudness-normalize
- [ ] Long-audio handling: chunk into about 10 s windows with overlap; overlap-add with permutation alignment across chunks (correlate overlap regions and/or ECAPA-TDNN speaker embeddings plus clustering; SpeechBrain has pretrained ECAPA). Prevents identity swaps mid-file
- [ ] OR-PIT interaction (important): recursion runs per chunk, so different chunks can extract speakers in a different ORDER and can even return a different NUMBER of speakers. ECAPA embeddings are therefore mandatory, not optional. Use them both to stitch identities across chunks and to reconcile per-chunk speaker counts (take the union across chunks)
- [ ] Note: PIT only resolves permutation inside a chunk. It has no memory across chunks, which is exactly why the embedding step exists
- [ ] Noise/reverb robustness: fine-tune on noisy and reverberant variants from Phase 2
- [ ] Optional polish: pass each separated stream through a speech-enhancement model (ClearerVoice FRCRN or MossFormer2-SE) to clean residual bleed
- [ ] Test on real-world audio: record ourselves talking simultaneously; YouTube podcast clips with crosstalk
- [ ] Deliverable: robust CLI handling arbitrary real recordings end-to-end

## Phase 6, Addons (bonus, parallel from week 6 if time permits)
- [ ] Web demo (highest impact, about 2 days): Gradio app, upload audio, get per-speaker players plus before/after spectrograms
- [ ] Per-speaker transcription: run Whisper on each separated track for a speaker-attributed transcript
- [ ] Speaker diarization overlay: pyannote to show speaking timelines
- [ ] Audio-visual mode (the L2L homage): for video inputs, run a modern pretrained AV model, RTFS-Net (https://github.com/spkgyk/RTFS-Net, ICLR 2024) or IIANet/CTCNet (https://github.com/JusperLee), with a face-detection and mouth-crop pipeline. Only after Phases 1 to 5 are solid; the AVSpeech/LRS data pipeline is a major time sink

## Phase 7, Evaluation, report, presentation (final 1 to 2 weeks)
- [ ] Full metric sweep: SI-SDRi, PESQ, STOI per level (2 to 5 or more spk, clean/noisy), baseline versus our models. Plot quality versus speaker count
- [ ] Ablations: pretrained versus fine-tuned; count-classifier accuracy; chunk-stitching on/off
- [ ] Failure analysis: same-gender similar voices, heavy noise, more than 5 speakers
- [ ] README, report, live demo; rehearse with a surprise mixture made by a teammate

---

## Suggested repo layout
```
VoxSplit/
  data/               # generation scripts and manifests (not raw audio)
  src/
    mixing/           # mixture generation (2..N speakers, noise, reverb)
    models/           # training and fine-tuning recipes
    inference/        # separate.py, chunking, stitching, count estimation
    eval/             # metrics, permutation matching, reports
  demo/               # Gradio app
  experiments/        # configs and results logs
  requirements.txt
  PLAN.md
```

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Not enough GPU compute | 16 GB 5070 Ti is plenty; the Phase 1 pretrained baseline already meets "3 or more speakers" with zero training; train at 8 kHz; Kaggle/Colab backup |
| 4 to 5 or more speaker quality collapses (SOTA 3-spk is about 20 dB, 5-spk about 10 dB, true for everyone) | Expected; show graceful degradation plus the recursive method story |
| Unknown test conditions (noise, length, format) | Phase 5 robustness; test on real recordings early |
| Deadline "coming soon" | Phases 1 to 2 give a submittable system within about 2 weeks; everything after improves it incrementally |

## Key references
- Looking to Listen: paper https://arxiv.org/pdf/1804.03619 , site https://looking-to-listen.github.io/ , dataset https://looking-to-listen.github.io/avspeech/
- Reimplementations: https://github.com/bill9800/speech_separation , https://github.com/JusperLee/Looking-to-Listen-at-the-Cocktail-Party
- SpeechBrain SepFormer 3-spk: https://huggingface.co/speechbrain/sepformer-wsj03mix , https://huggingface.co/speechbrain/sepformer-libri3mix
- ClearerVoice-Studio (MossFormer2): https://github.com/modelscope/ClearerVoice-Studio
- LibriMix generation: https://github.com/JorisCos/LibriMix
- Asteroid toolkit: https://github.com/asteroid-team/asteroid
- Unknown-count separation: Takahashi et al. 2019, recursive one-and-rest PIT (OR-PIT) https://arxiv.org/abs/1904.03065 , and Coarse-to-Fine recursive https://arxiv.org/abs/2203.16054
- SepFormer-style transformer extraction blocks trained with OR-PIT: Deflationary Extraction Transformer https://doi.org/10.3390/s25164905
- Deep Clustering (Hershey et al. 2016), the embedding/clustering alternative to PIT, and chronologically EARLIER than Conv-TasNet: https://arxiv.org/abs/1508.04306
- Modern AV separation: https://github.com/spkgyk/RTFS-Net , https://github.com/JusperLee/IIANet , https://github.com/JusperLee/CTCNet
- Paper and code index for the whole field: https://github.com/gemengtju/Tutorial_Separation
