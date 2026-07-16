# VoxSplit — Multi-Speaker Speech Separation
### Final report

## 1. Problem

Given a single-channel recording where **3 or more people speak at the same
time**, output one clean audio track per speaker — the "cocktail party problem".
The evaluation is on **audio inputs** with an **unknown number of speakers**, so
the system must be audio-only and count-agnostic. This report summarizes the
method, results, ablations, and honest limitations. Per-phase engineering
detail is in [PLAN.md](PLAN.md); auto-generated result tables and plots are in
[experiments/RESULTS.md](experiments/RESULTS.md).

## 2. Approach

Modern audio-only separators beat the 2018 audio-visual *Looking to Listen*
baseline, so the core system is audio-only. The design has four parts:

1. **Separator (OR-PIT SepFormer).** A 2-output SepFormer, warm-started from
   `speechbrain/sepformer-wsj02mix` and fine-tuned with **one-and-rest
   permutation invariant training** (OR-PIT, Takahashi et al. 2019): one head
   emits a single speaker, the other the sum of the rest. Trained 20k steps on
   on-the-fly 2- and 3-speaker LibriSpeech mixtures at 8 kHz.
2. **Recursion for unknown count.** To separate K speakers, extract one, feed
   the residual "rest" back into the same network, and repeat. Depth = speaker
   count, so a single model covers every level.
3. **Blind stop / count.** A small log-mel CNN predicts P(≥2 speakers). It
   decides, per recursion step, which head is the single speaker (head
   selection) and when to stop (the residual holds one speaker). The number of
   emitted tracks is the predicted count.
4. **Real-world front end.** Input normalization (mono, 8 kHz, peak), long-audio
   chunking, and cross-chunk identity stitching with ECAPA-TDNN embeddings
   (sequential one-to-one Hungarian matching), plus a noise/reverb-augmented
   fine-tune for robustness.

**Addons:** a Gradio web demo, per-speaker Whisper transcription, and speaking
timelines.

## 3. Headline results

Frozen eval set: LibriSpeech test-clean, 20 mixtures per level, 8 kHz, SI-SDRi.

| Level | Best fixed-N | OR-PIT recursion (oracle count) | OR-PIT blind (unknown count) |
|---|---|---|---|
| 2 spk | 19.25 | 19.25 | 19.25 |
| 3 spk | 19.33 | 15.62 | 16.09 |
| 4 spk | 15.16 | 8.50 | 9.09 |
| 5 spk | 11.01 | 5.96 | 7.23 |

- One model separates **2–5 speakers with graceful degradation**, the expected
  ~10 dB drop from 3 to 5 speakers.
- **Blind unknown-count accuracy 0.71** overall (2 spk 1.00, 3 spk 0.70, 4 spk
  0.40, 5 spk 0.75). When the count is right, blind separation matches the
  oracle-guided recursion.
- Dedicated fixed-N models win on raw quality (no recursion error propagation),
  but need the count known and a model per level; OR-PIT trades some quality for
  count-agnostic coverage — the actual evaluation setting.

## 4. Robustness

The clean-trained separator collapses under degradation; a 6k-step fine-tune on
WHAM-noise + pooled-RIR reverb recovers most of the gap with no clean cost:

| Condition | Clean-trained | Augmented-robust |
|---|---|---|
| clean | 19.25 | 19.02 |
| WHAM noise | 6.68 | 11.44 |
| reverb | 4.04 | 7.06 |

End-to-end on a 30 s conversation (unknown count, long-form): mixed-gender
recovers correct count and +6.5 / +8.8 dB; the same conversation at 5 dB SNR
noise through the robust pipeline gives +14.7 / +15.9 dB.

## 5. Ablations (see RESULTS.md for tables)

- **Fine-tuning helps out-of-domain, not in-domain.** OR-PIT from WSJ-trained
  wsj02mix gains +1.9 dB on LibriSpeech (17.35 → 19.25); a uPIT fine-tune of the
  already-LibriSpeech-trained libri3mix does not (19.33 → 18.65).
- **Training length:** 6k → 20k lifts recursion +2.0 dB at 3 speakers.
- **The count/stop classifier is make-or-break.** Trained on clean mixtures it
  fails (0.25 blind count accuracy — separation artifacts read as extra
  speakers, so recursion never stops). Training it on the separator's OWN
  residuals at multiple recursion depths lifts it to **0.71** (0.25 → 0.49 →
  0.71).
- **Cross-chunk stitching:** free clustering scrambles identities (negative
  SI-SDRi); sequential Hungarian matching fixes it (+6.5 / +8.8 dB).

## 6. Failure analysis

- **4 speakers** is the count weak point (0.40) — three recursion passes
  accumulate artifacts. Separation is fine once counted right (9.09 dB); the
  demo offers a "force count" override.
- **> 5 speakers** (untrained): a forced 6-speaker run still produces 6 tracks
  but quality collapses to ~3.0 dB mean (one source near-failed). Out of the
  trained 2–3 speaker range; the field degrades hard past 5 too.
- **Similar voices:** ECAPA discrimination is weak on short, bleed-heavy
  separated chunks, so same-gender similar-voice long-form conversations
  mis-stitch; distinct voices stitch cleanly.
- **Reverberation** is the harshest degradation; augmentation only partially
  recovers it (4.04 → 7.06 dB).
- **Speech-enhancement post-filter hurts** (MetricGAN+ → −0.49 dB): denoisers
  over-process already-separated speech. Fix bleed at the source, not after.

## 7. How to run

```bash
# environment (conda env voxsplit, torch cu128 for the Blackwell GPU)
python src/check_env.py

# separate one file, unknown count
python src/inference/separate_unknown.py mix.wav \
    --orpit-ckpt checkpoints/orpit/ckpt_step20000.pt \
    --clf-ckpt checkpoints/count_clf_res/ckpt_step8000.pt --out-dir out/

# long / real recordings (chunking + ECAPA stitching), robust model
python src/inference/separate_longform.py recording.wav \
    --orpit-ckpt checkpoints/orpit_robust/ckpt_step26000.pt \
    --clf-ckpt checkpoints/count_clf_res/ckpt_step8000.pt --out-dir out/

# web demo (per-speaker players, spectrograms, transcripts, timeline)
python demo/app.py            # http://localhost:7860

# regenerate result plots + tables
python experiments/make_report.py
```

## 8. Limitations and future work

- Blind count is unreliable at 4 and beyond 5; a stronger count model (or
  fine-tuning the recursion loop end-to-end, as Takahashi note) is the next
  lever.
- ECAPA stitching degrades for similar voices; embedding on enhanced audio or
  constrained clustering could help.
- **Audio-visual mode** (the *Looking to Listen* homage) is implemented and
  working (`src/av/`): the video drives the audio-only separator — on-screen
  faces set the speaker count, and each track is matched to its speaker by
  correlating loudness with per-face mouth motion (mediapipe FaceMesh, or a
  ROI-motion fallback). Validated on a synthetic talking-face clip (correct
  face assignment at ~19.7 dB). Full AV *masking* with a pretrained model
  (RTFS-Net/IIANet/CTCNet) — which fuses mouth crops into the separation itself
  and needs an external repo + video training data — is the documented heavier
  alternative. See [src/av/README.md](src/av/README.md).
