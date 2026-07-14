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
- [x] Noise, reverb, and 16 kHz variants added (see robustness table below). WHAM noise (test split) assigned per mixture by `src/data/make_conditions.py`; reverb via pyroomacoustics with room geometry stored in the manifest; 16 kHz set via `build_eval_set --sample-rate 16000`. All variant manifests are committed and reproducible; the audio and the WHAM corpus stay gitignored
- [x] Deliverable (clean): `data/eval_manifest.json` committed, reproducible generation and batch scoring scripts in place

### Frozen eval set baselines (test-clean, 20 mixtures per level, scored at 8 kHz)
| Level | Model | SI-SDRi (dB) | PESQ | STOI |
|---|---|---|---|---|
| 2 spk | sepformer-wsj02mix | 17.35 | 3.37 | 0.96 |
| 3 spk | sepformer-libri3mix | 19.33 | 3.20 | 0.93 |
| 4 spk | no pretrained model yet (Phase 3) | n/a | n/a | n/a |
| 5 spk | no pretrained model yet (Phase 3) | n/a | n/a | n/a |

These are the numbers every future model must beat on the same frozen mixtures. Per-level rows accumulate in `experiments/eval_set_results.csv`.

### Robustness of the clean baseline under degradation (libri3mix, 3 speakers, 20 mixtures)
| Condition | SI-SDRi (dB) | PESQ | STOI |
|---|---|---|---|
| clean | 19.33 | 3.20 | 0.93 |
| WHAM noise (0 to 10 dB SNR) | 11.26 | 2.06 | 0.80 |
| reverb (RT60 0.2 to 0.6 s) | 6.05 | 1.79 | 0.65 |

The clean-trained SepFormer degrades gracefully under additive noise but collapses under reverberation. This quantifies the Phase 5 robustness gap and says reverb is the priority target. Condition rows are logged in `experiments/eval_set_conditions.csv`. Variant manifests: `data/eval_manifest_noise.json`, `data/eval_manifest_reverb.json`, and a 16 kHz clean set in `data/eval_manifest_16k.json`.

## Phase 3, Train our own models, week 3 to 6 (the core)
- [x] Priority: fine-tune pretrained models rather than training from scratch. Adopted: the OR-PIT headline track warm-starts from wsj02mix and the baseline track warm-starts from libri3mix; both fine-tune rather than train from zero
- [x] Optional learning warm-up: Conv-TasNet trained from scratch with standard PIT (torchaudio ConvTasNet, so no Asteroid install) via `src/train/train_convtasnet.py`. A short 6000-step run reaches about 5.0 dB SI-SDRi on the 2-speaker frozen level. The PIT pipeline works end to end; from scratch it is nowhere near converged (Conv-TasNet needs roughly a GPU-day). Purpose (understand PIT end to end) achieved
- [x] Baseline track: uPIT fine-tune of libri3mix (fixed 3 outputs) via `src/train/train_pit.py` and `src/train/pit_loss.py`. Result on the frozen 3-speaker level: 18.65 dB SI-SDRi, slightly BELOW the pretrained libri3mix (19.33 dB). Honest finding: libri3mix is already LibriSpeech-trained, so a short further fine-tune on our mixtures does not help (unlike OR-PIT from wsj02mix, which gained 1.5 dB because wsj02mix was WSJ-trained). The pretrained libri3mix stays our 3-speaker reference
- [x] Headline track, SepFormer + OR-PIT (the team's proposal, and the right call). Built and verified: 2-head "one and rest" model fine-tuned from wsj02mix, beating the baseline on the 2-speaker level. Details in the sub-bullets below.
  - IMPORTANT: OR-PIT is a training objective plus an output topology, NOT an inference-time switch. The pretrained uPIT checkpoints cannot do it as-is; this track requires real training
  - [x] Warm-start from `speechbrain/sepformer-wsj02mix` (already 2 output heads). Implemented in `src/train/train_orpit.py` (plain PyTorch, re-enables grad on the inference-loaded modules)
  - [x] OR-PIT loss implemented in `src/train/orpit_loss.py`: one-and-rest SI-SNR that tries every "one" speaker and both head assignments and keeps the best
  - [x] Dynamic 2 and 3-speaker training mixtures from train-clean-100 (`src/train/mix_dataset.py`). Pipeline VERIFIED on the RTX 5070 Ti: warm-started model already sits at about 12 to 14 dB SI-SNRi and the OR-PIT loss trains cleanly. Full fine-tuning run launched
  - [x] Precedent: transformer/SepFormer-style extraction blocks trained with OR-PIT are published, so this is not unexplored territory (Deflationary Extraction Transformer, https://doi.org/10.3390/s25164905)
  - [x] Initial 6000-step fine-tune done. On the frozen 2-speaker level the OR-PIT model scores 18.82 dB SI-SDRi (PESQ 3.51, STOI 0.97), beating its wsj02mix warm-start baseline (17.35 dB) by about 1.5 dB. Batch inference via `src/inference/separate_orpit_set.py` loading the trained checkpoint
  - [x] Longer training toward convergence: `train_orpit.py` gained a `--resume` flag (loads encoder/masknet/decoder + optimizer state, treats the step arg as an absolute target, and offsets the dataset seed by the resume step so continued training sees fresh mixtures). A run continuing from step 6000 to 20000 is what produced the converged checkpoint scored below
  - [x] Recursive separation for 3-or-more speakers built: `src/inference/separate_orpit_recursive_set.py` runs the 2-head OR-PIT model recursively (extract one speaker, feed the "rest" head back, repeat K-1 times for a known count K). Head selection (which head is the extracted speaker vs the residual to recurse on) uses an ORACLE against the references, so these numbers are the separation-quality CEILING of the recursion; the blind stop/selection classifier is the Phase 4 deliverable. Self-test recovers 2..5 synthetic sources with no model. This gives VoxSplit its first 4- and 5-speaker separation numbers (previously n/a)
- [x] Optional fixed-N comparison models: 4-speaker and 5-speaker uPIT models built and trained. `train_pit.expand_masknet_heads` reshapes the 3-head libri3mix masknet to N heads, warm-starting the new heads by copying existing ones (cycled) rather than random init, so they converge fast. 8000-step runs: 4-spk uPIT 15.16 dB, 5-spk uPIT 11.01 dB SI-SDRi. These beat OR-PIT recursion at the same levels (8.50 and 5.96 dB), quantifying the price recursion pays for count-agnostic coverage (error propagation across passes). See the comparison table below
- [x] MossFormer2 or TF-GridNet recipe for the 3-spk level: compact TF-GridNet built (`src/models/tfgridnet.py`) and trained from scratch (`train_tfgridnet.py`). No public 3-speaker TF-GridNet checkpoint exists to warm-start from (published ones are WSJ0-2mix, license-restricted), so this is from-scratch. An 8000-step learning-scale run reaches 3.76 dB SI-SDRi at 3 speakers, well short of the fine-tuned SepFormer baselines (a from-scratch TF-domain model needs far more steps, the full-band self-attention module, and wider dims to approach its ~23 dB potential). Honest learning-scale result, in the same spirit as the Conv-TasNet from-scratch warm-up. MossFormer2 was not used here because ClearerVoice ships only a 2-speaker MossFormer2
- [x] Loss and tracking: SI-SDR / SI-SNR (OR-PIT one-and-rest); SI-SDRi tracked per level on the frozen sets in `experiments/eval_set_results.csv`, and per-step training SI-SNRi in per-run `train_log.csv` files
- [x] VRAM note honored: trained at 8 kHz, 3 s segments, batch 1-2. Every run fit in the 16 GB RTX 5070 Ti; gradient accumulation was not needed at these batch sizes
- [x] Weights & Biases added as OPTIONAL: `src/train/wandb_logger.py` plus a `--wandb` flag on every trainer (offline mode default, no account needed). Off by default; the committed CSV logs remain the source of truth. `wandb` is an optional requirement, not needed to train
- [x] Deliverable: one OR-PIT SepFormer that separates 2 to 5 speakers (2-spk direct at 19.25 dB, 3/4/5-spk via recursion at 15.62 / 8.50 / 5.96 dB SI-SDRi), benchmarked against fixed-N uPIT baselines at every level and a from-scratch TF-GridNet at 3 speakers (comparison table below)

### Phase 3 training results so far (frozen eval set, 20 mixtures per level, 8 kHz)
| Model | Level | SI-SDRi (dB) | PESQ | STOI | vs its baseline |
|---|---|---|---|---|---|
| sepformer-wsj02mix (pretrained) | 2 spk | 17.35 | 3.37 | 0.96 | baseline |
| OR-PIT 6k (from wsj02mix) | 2 spk | 18.82 | 3.51 | 0.97 | +1.5 dB, helps |
| OR-PIT 20k (converged) | 2 spk | **19.25** | 3.53 | 0.97 | +1.9 dB, converged |
| sepformer-libri3mix (pretrained) | 3 spk | 19.33 | 3.20 | 0.93 | baseline |
| uPIT 6k (from libri3mix) | 3 spk | 18.65 | 3.11 | 0.93 | -0.7 dB, no gain |
| Conv-TasNet 6k (from scratch) | 2 spk | 5.04 | 1.76 | 0.78 | learning demo, unconverged |
| uPIT-4 8k (from libri3mix, expanded heads) | 4 spk | 15.16 | 2.23 | 0.85 | fixed-4 baseline |
| uPIT-5 8k (from libri3mix, expanded heads) | 5 spk | 11.01 | 1.74 | 0.72 | fixed-5 baseline |
| TF-GridNet 8k (from scratch) | 3 spk | 3.76 | 1.52 | 0.65 | learning demo, unconverged |

Reading: fine-tuning helps when the pretrained start is out-of-domain (OR-PIT from WSJ-trained wsj02mix gains on LibriSpeech), but not when it is already in-domain (libri3mix was LibriSpeech-trained). So the OR-PIT model is the one to carry into Phase 4; the pretrained libri3mix stays the fixed-3 reference. All rows logged in `experiments/eval_set_results.csv`.

### Recursive OR-PIT on 3-or-more speakers (oracle head selection, known count)
The single 2-head OR-PIT model, recursed K-1 times, separates every level. Numbers below use the converged 20000-step checkpoint. Oracle head selection means these are the recursion's quality ceiling, not a blind-inference result (blind stop/selection is Phase 4).

| Model | Level | SI-SDR (dB) | SI-SDRi (dB) | PESQ | STOI | vs 6k ckpt |
|---|---|---|---|---|---|---|
| OR-PIT 20k recursive | 3 spk | 12.36 | **15.62** | 2.75 | 0.90 | +2.0 dB |
| OR-PIT 20k recursive | 4 spk | 3.25 | **8.50** | 1.86 | 0.73 | +0.7 dB |
| OR-PIT 20k recursive | 5 spk | -0.68 | **5.96** | 1.60 | 0.62 | +0.1 dB |

Reading: one model covers 2 to 5 speakers with the expected graceful degradation (about 10 dB drop from 3 to 5 speakers, in line with the field). Longer training helped most at 2 and 3 speakers (+0.4 and +2.0 dB) and least at 5, where error propagation through four recursion passes dominates. At 3 speakers the recursion (15.62 dB) still trails the dedicated fixed-3 libri3mix head (19.33 dB): recursion trades some quality for count-agnostic coverage. Rows logged in `experiments/eval_set_results.csv` under tags `orpit_20k_2spk` and `orpit_20k_recursive_oracle` (the earlier `*_6k_*` rows are kept for the training-length ablation).

### Fixed-N baselines vs recursive OR-PIT (the benchmark)
The headline question: does one count-agnostic OR-PIT model give up much against dedicated fixed-N models trained per level? Fixed-N uPIT models (4- and 5-speaker) were warm-started from libri3mix via head expansion and trained 8000 steps; a from-scratch TF-GridNet gives a modern time-frequency point at 3 speakers.

| Level | Best fixed-N model (SI-SDRi) | OR-PIT recursive (SI-SDRi) | Gap |
|---|---|---|---|
| 3 spk | 19.33 (pretrained libri3mix); 18.65 uPIT-ft; 3.76 TF-GridNet from scratch | 15.62 | -3.7 dB vs best |
| 4 spk | 15.16 (uPIT-4, this work) | 8.50 | -6.7 dB |
| 5 spk | 11.01 (uPIT-5, this work) | 5.96 | -5.1 dB |

Reading: dedicated fixed-N heads clearly win on raw quality (no recursion error propagation, one-shot assignment), so if the speaker count is known and fixed, a per-N model is better. The OR-PIT recursion's value is different: a SINGLE model handles every count and, with the Phase 4 stop classifier, an UNKNOWN count, which is the actual evaluation setting. The gap grows with K (recursion compounds errors over more passes), so improving recursion at high K (fine-tuning the recursive loop, a better stop/selection rule) is the clear Phase 4 target. The from-scratch TF-GridNet (3.76 dB) confirms that TF-domain SOTA is not reachable at learning-scale step counts; it stays a learning baseline like Conv-TasNet, not a contender. All rows logged in `experiments/eval_set_results.csv` (tags `pit4_8k`, `pit5_8k`, `tfgridnet_8k`).

## Phase 4, Unknown speaker count, week 5 to 7 (the likely scoring edge, treat as a first-class goal)
Test inputs will not announce how many speakers there are. This is probably where the evaluation is won or lost. Both the external review and the team's own paper survey converge on recursive separation as the primary route, so that is what we build.
- [x] PRIMARY, blind recursive OR-PIT separation using the Phase 3 model: `src/inference/separate_recursive_blind.py`. Extract one speaker, feed the residual "rest" head back into the same network, repeat. Head selection and stopping are decided by the count/stop classifier below (no references), so recursion depth adapts to the unknown count and one model covers every level. Self-test recovers 1..5 speakers with an ideal model+classifier
- [x] Stopping criterion (the single highest-risk component, as predicted): a classifier on the residual deciding whether >=2 speakers remain. `src/models/count_classifier.py`, a log-mel CNN emitting K in {1..5} with `prob_multi()` = P(>=2). FIRST attempt trained on clean mixtures FAILED exactly as feared: OR-PIT separation artifacts make a separated single speaker read as multi (P(>=2)=0.96 vs 0.07 for clean single), so recursion never stopped and every mixture was called 5 speakers (count accuracy 0.25). FIX: train the classifier on the OR-PIT model's OWN outputs (residual domain) via `train_count_classifier.py --orpit-ckpt`, sampling raw mixtures, extracted single heads, and residual "rest" heads at a random recursion DEPTH (multi-level), oracle-labelled. This raised blind count accuracy from 0.25 to 0.49 (single-pass residuals) to 0.71 (multi-level residuals)
- [x] Speaker count comes for free from the recursion depth: the number of emitted estimates IS the predicted count. Blind count accuracy 0.71 on the frozen 2-5 set (per level: 2 spk 1.00, 3 spk 0.70, 4 spk 0.40, 5 spk 0.75); 4 speakers is the weak point where error propagation through three passes dominates. Predictions logged in `experiments/phase4_count_predictions.csv`
- [x] Secondary, speaker-count classifier: the SAME log-mel CNN doubles as the standalone K-in-{1..5} cross-check (its argmax), so no separate model was needed. Recursion-depth counting (0.71) is the primary count; the direct classifier is the ablation
- [x] Secondary, max-N plus silence detection: built as `src/inference/separate_maxn_set.py`. Always run the 5-head uPIT model, then drop channels whose RMS is more than --drop-db below the loudest (relative energy gating; self-tested). Kept-channel count is the prediction. Best at drop-db 10: count accuracy 0.53 (per level 2/3/4/5 = 0.55 / 0.65 / 0.10 / 0.80), below the recursion's 0.71. Instructive trade-off, though: on correctly-counted mixtures the fixed-5 model has NO recursion error propagation, so it separates high counts BETTER (5 spk 10.76 vs recursion 7.23 dB SI-SDRi) while doing WORSE on low counts (2 spk 13.25 vs 19.25, it over-splits a 2-speaker mixture across 5 heads). So the two routes are complementary: recursion for count accuracy and low K, max-N for high-K quality once the count is known, an ensemble hint for Phase 7. Predictions logged under tag `maxn5_drop10`
- [x] Known OR-PIT weaknesses measured and reported honestly: error propagation is visible (count accuracy and separation both dip at 4 speakers, deep residuals are the hardest); passes are sequential (no parallelism, slower inference); and the pipeline is sensitive to the stop classifier (the 0.25 -> 0.71 story quantifies exactly that). Fine-tuning the recursion loop is the clear next lever
- [x] Compare on the frozen sets (count accuracy and separation quality): blind separation quality on correctly-counted mixtures matches the oracle recursion (2/3/4/5 spk = 19.25 / 16.09 / 9.09 / 7.23 dB SI-SDRi vs oracle 19.25 / 15.62 / 8.50 / 5.96), i.e. once the count is right, blind head selection is as good as oracle. The count-error cost is the 23/80 mixtures scored as the wrong K. See the table below
- [x] Deliverable: single entry point `src/inference/separate_unknown.py input.wav` working with no prior knowledge of speaker count (writes one speaker*.wav per detected speaker). Verified end to end on a 3-speaker mixture (detected 3)

### Blind unknown-count results (frozen eval set, threshold 0.5)
The count/stop classifier drives blind recursion; nothing tells the model how many speakers there are.

| Level | Blind count accuracy | Blind SI-SDRi (correct-count subset) | Oracle SI-SDRi (known count) |
|---|---|---|---|
| 2 spk | 1.00 | 19.25 | 19.25 |
| 3 spk | 0.70 | 16.09 | 15.62 |
| 4 spk | 0.40 | 9.09 | 8.50 |
| 5 spk | 0.75 | 7.23 | 5.96 |

Reading: a SINGLE model now separates a recording with an unknown number of speakers, 0.71 count accuracy overall, and when it gets the count right it separates as well as the oracle-guided recursion. The stop classifier was indeed the make-or-break component: it only worked once trained on the OR-PIT model's own artifact-laden residuals at multiple recursion depths (the naive clean-trained version scored 0.25 and never stopped). Weakest at 4 speakers (0.40), the expected error-propagation regime. Rows: `experiments/eval_set_results.csv` tag `orpit_blind_ml`, counts in `experiments/phase4_count_predictions.csv`.

## Phase 5, Robustness and real-world inputs, week 6 to 8
- [x] Input normalization: `src/inference/audio_io.py` downmixes to mono, resamples to the 8 kHz model rate, and peak-normalizes, centralizing what was duplicated as load_mono across the inference scripts. Self-tested (stereo 16 kHz -> mono 8 kHz at peak 0.9, silent-input safe)
- [x] Long-audio handling: `src/inference/separate_longform.py` chunks the recording into ~4 s windows (kept near the 3 s training segment so the separator and count classifier stay in domain) with overlap, separates each chunk with blind recursion, and overlap-adds each speaker's chunks back with a tapered, weight-normalized window
- [x] OR-PIT interaction handled with ECAPA-TDNN: each chunk-speaker is embedded (SpeechBrain `spkrec-ecapa-voxceleb`), and identities are stitched across chunks by SEQUENTIAL ONE-TO-ONE (Hungarian) matching of each chunk's estimates to running global-speaker centroids, with unmatched estimates starting new speakers (so the global count is reconciled as the union across chunks). One-to-one matching per chunk was essential: free agglomerative clustering let two estimates from one chunk collapse into a single speaker and scrambled the tracks (negative SI-SDRi); Hungarian matching fixed it. IMPORTANT lesson learned: ECAPA discrimination is weak on SHORT, BLEEDY separated chunks (same-speaker-across-chunks and different-speaker cosine distances overlap around 0.3-0.5), so stitching is reliable for distinct voices and degrades for similar ones
- [x] Note confirmed empirically: PIT/recursion resolves permutation only inside a chunk; the embedding step is what carries identity across chunks. Validated on a synthetic 30 s two-speaker conversation: mixed-gender (distinct voices) recovers the correct count and +6.5 / +8.8 dB SI-SDRi; a same-gender similar-voice pair mis-counts (the documented ECAPA limitation)
- [x] Noise/reverb robustness: `src/train/augment.py` (AugMixDataset) adds WHAM noise and pooled-RIR reverb to training mixtures on the fly (clean refs for noise, reverberant refs for reverb, matching the Phase 2 eval convention), wired into `train_orpit.py --noise-dir/--reverb`. The clean 20k OR-PIT collapses on the degraded 2-speaker sets (noise 6.68, reverb 4.04 dB SI-SDRi vs 19.25 clean). A 6000-step augmented fine-tune from the 20k checkpoint recovers most of that gap: noise 6.68 -> 11.44 dB (+4.8), reverb 4.04 -> 7.06 dB (+3.0), while clean is essentially unchanged (19.25 -> 19.02). So one augmented fine-tune buys large robustness at almost no clean cost
- [x] Optional polish (speech-enhancement post-filter): built as `src/inference/enhance.py` using SpeechBrain MetricGAN+ (the ClearerVoice FRCRN / MossFormer2-SE models could not run here: their `clearvoice` env's numba/llvmlite.dll is blocked by a Windows Application Control policy, WinError 4551, so MetricGAN+ is the runnable pure-PyTorch stand-in). Ablation verdict: it HURTS. On the clean 2-speaker estimates SI-SDRi collapses 19.25 -> -0.49 (PESQ 3.53 -> 3.11); on the robust-model noisy estimates 11.44 -> -1.71 dB (PESQ 2.17 -> 2.34, marginal). MetricGAN+ is trained for single-speaker denoising and over-processes already-separated speech, destroying scale/waveform fidelity for at best a tiny perceptual gain. This confirms the plan's hypothesis: the residual-bleed problem is better fixed at the source (the augmented fine-tune) than with a post-filter. Honest negative result, kept as an ablation
- [x] Test on real-world-style audio: self-recording and YouTube scraping are not possible in this automated setting, so the hardest feasible test is a NOISY long-form conversation, a 30 s two-speaker mixed-gender conversation degraded with WHAM noise at 5 dB SNR, run through the FULL robust pipeline (robust OR-PIT + count classifier + ECAPA long-form stitching). Result: correct count (2) and +14.67 / +15.92 dB SI-SDRi, i.e. the end-to-end robust CLI holds up under realistic noise + length + unknown count. Real crosstalk-YouTube testing remains a manual follow-up
- [x] Deliverable: robust end-to-end CLI. `separate_longform.py input.wav --orpit-ckpt ... --clf-ckpt ...` takes an arbitrary recording (any rate/channels/length, unknown count), normalizes, chunks, separates blindly, stitches identities with ECAPA, and writes one speaker*.wav per detected speaker. `separate_unknown.py` remains the short-file entry point

### Robustness of OR-PIT under degradation (frozen 2-speaker sets, 20 mixtures)
| Condition | Clean 20k OR-PIT (SI-SDRi) | Noise/reverb-augmented fine-tune (SI-SDRi) | Gain |
|---|---|---|---|
| clean | 19.25 | 19.02 | -0.2 (retained) |
| WHAM noise | 6.68 | 11.44 | +4.8 |
| reverb | 4.04 | 7.06 | +3.0 |

Reading: the clean-trained separator collapses under noise and especially reverb (matching the Phase 2 finding). A single 6000-step fine-tune on WHAM-noise + pooled-RIR-reverb augmented mixtures recovers a large fraction of the gap (+4.8 dB noise, +3.0 dB reverb) with negligible clean regression, so it is the model to deploy for real recordings. Rows in `experiments/eval_set_results.csv` under tags `orpit20k_{noise,reverb}` (clean baseline) and `orpit_robust_{clean,noise,reverb}`.

## Phase 6, Addons (bonus, parallel from week 6 if time permits)
- [x] Web demo: `demo/app.py`, a Gradio app. Upload a recording, get for each auto-detected speaker a playable track, an after-separation spectrogram, and a transcript, plus the input spectrogram and a speaking timeline. The count is discovered (no count supplied); the robust OR-PIT checkpoint is the default. `demo/pipeline.py` holds the reusable end-to-end `Pipeline` (models loaded once) shared by the app and the CLI. Verified: the server builds and serves, and the pipeline runs end to end on a 30 s two-speaker file
- [x] Per-speaker transcription: `src/inference/transcribe.py` runs Whisper on each separated track for a speaker-attributed transcript. Uses faster-whisper (CTranslate2), NOT openai-whisper, because the latter pulls numba/llvmlite.dll, which is blocked by the Windows Application Control policy on this machine. Verified: on the separated 2-speaker demo file it returns clean, correctly-attributed transcripts (each speaker's LibriSpeech passage)
- [x] Speaker diarization overlay: `src/inference/timeline.py` derives a per-speaker "who spoke when" timeline with an energy VAD on each separated track and plots horizontal speaking bars. This deliberately avoids pyannote, whose diarization model is licence-gated (needs an accepted agreement + HF token); because each separated stream already IS one speaker, diarization reduces to VAD on that stream, so no gated model is needed. Self-tested
- [~] Audio-visual mode (the L2L homage): NOT built, deferred by scope. It requires cloning an external AV repo (RTFS-Net / IIANet / CTCNet), a full face-detection + mouth-crop video pipeline, and video test data (AVSpeech/LRS), which the plan itself flags as a major multi-day time sink and gates on "only after Phases 1-5 are solid". The project's core evaluation is audio-only (the test inputs are audio), so audio-visual stays a genuine bonus; the audio-only demo above is the shipped addon. Left as a clearly-scoped future extension

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
