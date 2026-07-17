"""Long-audio separation with ECAPA identity stitching (VoxSplit Phase 5).

A separator has no memory across time: run it on 10-second chunks of a long
recording and each chunk returns its speakers in an arbitrary order, and (with
blind recursive OR-PIT) possibly a different COUNT. PIT only resolves the
permutation inside a chunk. To reassemble whole-file per-speaker tracks we must
match identities across chunks, which is what speaker embeddings are for.

Pipeline:
  1. Split the signal into chunk_len windows hopping by hop_len (overlap).
  2. Separate each chunk (blind recursive OR-PIT) -> variable speakers/chunk.
  3. Embed every chunk-speaker with ECAPA-TDNN.
  4. Cluster all embeddings (agglomerative, cosine): the number of clusters is
     the global speaker count (the union across chunks), and each cluster is
     one real speaker.
  5. Overlap-add each chunk-speaker into its cluster's full-length track,
     weight-normalized so overlaps blend and gaps stay silent.

The clustering IS the cross-chunk permutation alignment and count
reconciliation. Head selection/stopping inside a chunk stay the Phase 4 job.

ECAPA weights are loaded from a local directory (--ecapa-dir). Pre-fetch once
(the SpeechBrain HF fetch can hang on Windows):
  python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('speechbrain/spkrec-ecapa-voxceleb', \
    local_dir='pretrained_models/ecapa-dl')"

Run this file directly for a self-test that needs no real data or model:
  python src/inference/separate_longform.py --self-test

Example:
  python src/inference/separate_longform.py long_meeting.wav \
      --orpit-ckpt checkpoints/orpit/ckpt_step20000.pt \
      --clf-ckpt checkpoints/count_clf_res/ckpt_step8000.pt \
      --out-dir out/meeting --chunk-seconds 10 --overlap-seconds 2
"""
import argparse
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from audio_io import load_normalized, MODEL_SR  # noqa: E402
from separate_recursive_blind import blind_recursive_separate  # noqa: E402


def chunk_signal(signal: np.ndarray, chunk_len: int, hop_len: int
                 ) -> List[Tuple[int, np.ndarray]]:
    """Split into (start_sample, chunk) windows hopping by hop_len.

    The final window is whatever remains (shorter than chunk_len). Always
    returns at least one window.
    """
    n = len(signal)
    if n <= chunk_len:
        return [(0, signal)]
    starts = list(range(0, max(n - chunk_len, 0) + 1, hop_len))
    if starts[-1] + chunk_len < n:
        starts.append(n - chunk_len)
    return [(s, signal[s:s + chunk_len]) for s in starts]


def _taper_window(length: int, ramp: int) -> np.ndarray:
    """A flat window with linear ramps of `ramp` samples at each end."""
    w = np.ones(length, dtype=np.float64)
    r = min(ramp, length // 2)
    if r > 0:
        ramp_up = np.linspace(0.0, 1.0, r, endpoint=False)
        w[:r] = ramp_up
        w[length - r:] = ramp_up[::-1]
    return w


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector (so a dot product is a cosine similarity)."""
    return v / (np.linalg.norm(v) + 1e-9)


def _assign_chunk(embs, centroids, counts, threshold: float, force_count: "int | None" = None):
    """One-to-one match a chunk's embeddings to global speakers; grow as needed.

    Hungarian assignment minimizes total cosine distance between this chunk's
    estimates and existing global centroids. A match is accepted only if its
    cosine distance is below `threshold`; unmatched estimates start new global
    speakers. Matched centroids are updated with a running mean. Mutates
    `centroids`/`counts` and returns the global id per input embedding.
    """
    from scipy.optimize import linear_sum_assignment

    ids = [None] * len(embs)
    if centroids:
        cost = np.stack([[1.0 - float(np.dot(e, c)) for c in centroids]
                         for e in embs])
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if force_count is not None:
                is_at_limit = (len(centroids) >= force_count)
                if is_at_limit or cost[r, c] < threshold:
                    ids[r] = c
                    centroids[c] = _unit(centroids[c] * counts[c] + embs[r])
                    counts[c] += 1
            else:
                if cost[r, c] < threshold:
                    ids[r] = c
                    centroids[c] = _unit(centroids[c] * counts[c] + embs[r])
                    counts[c] += 1
    for r in range(len(embs)):
        if ids[r] is None:
            if force_count is not None and len(centroids) >= force_count:
                if centroids:
                    c = int(np.argmin([1.0 - float(np.dot(embs[r], cent)) for cent in centroids]))
                    ids[r] = c
                    centroids[c] = _unit(centroids[c] * counts[c] + embs[r])
                    counts[c] += 1
                else:
                    ids[r] = 0
                    centroids.append(embs[r].copy())
                    counts.append(1)
            else:
                ids[r] = len(centroids)
                centroids.append(embs[r].copy())
                counts.append(1)
    return ids


def separate_longform(
        signal: np.ndarray,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        prob_multi_fn: Callable[[np.ndarray], float],
        embed_fn: Callable[[np.ndarray], np.ndarray],
        chunk_len: int,
        hop_len: int,
        cluster_threshold: float = 0.55,
        recursion_threshold: float = 0.5,
        force_count: "int | None" = None,
        ramp: int = 400) -> List[np.ndarray]:
    """Separate a long signal into whole-file per-speaker tracks.

    Returns a list of G full-length tracks, one per clustered global speaker.
    recursion_threshold leans the per-chunk stop decision (a lower value
    over-extracts, which is safer than dropping a speaker in a chunk since the
    clustering merges duplicates anyway). cluster_threshold is a cosine distance
    on ECAPA embeddings of SEPARATED audio, whose speaker distances are
    compressed by residual bleed, so it is smaller than a clean-speech default.
    """
    n = len(signal)
    windows = chunk_signal(signal, chunk_len, hop_len)

    # Separate every chunk and embed every chunk-speaker, keeping the per-chunk
    # grouping (each chunk's estimates are distinct speakers).
    per_chunk = []          # list of (start, [wav...], [emb...])
    for start, chunk in windows:
        ests = blind_recursive_separate(chunk, forward_fn, prob_multi_fn,
                                        threshold=recursion_threshold,
                                        force_count=force_count)
        embs = [_unit(np.asarray(embed_fn(w), dtype=np.float64).flatten())
                for w in ests]
        per_chunk.append((start,
                          [np.asarray(w, dtype=np.float32) for w in ests],
                          embs))

    if not any(wavs for _, wavs, _ in per_chunk):
        return [np.zeros(n, dtype=np.float32)]

    # Sequential one-to-one identity stitching. Global speakers are seeded from
    # the first non-empty chunk; each later chunk Hungarian-matches its
    # estimates to existing global centroids (one-to-one, so two estimates from
    # the SAME chunk can never collapse into one speaker), and any estimate too
    # far from every centroid starts a new speaker (count reconciliation).
    items = []              # (start, wav, global_id)
    centroids: List[np.ndarray] = []
    counts: List[int] = []
    for start, wavs, embs in per_chunk:
        ids = _assign_chunk(embs, centroids, counts, cluster_threshold, force_count=force_count)
        for wav, g in zip(wavs, ids):
            items.append((start, wav, g))
    num_speakers = len(centroids)

    # Overlap-add each chunk-speaker into its global track, weight-normalized.
    outputs = [np.zeros(n, dtype=np.float64) for _ in range(num_speakers)]
    weights = [np.zeros(n, dtype=np.float64) for _ in range(num_speakers)]
    for start, wav, g in items:
        length = min(len(wav), n - start)
        if length <= 0:
            continue
        win = _taper_window(length, ramp)
        outputs[g][start:start + length] += wav[:length] * win
        weights[g][start:start + length] += win

    tracks = []
    for out, wt in zip(outputs, weights):
        safe = np.where(wt > 1e-8, wt, 1.0)
        tracks.append((out / safe).astype(np.float32))
    return tracks


def load_longform_models(orpit_ckpt: Path, clf_ckpt: Path, ecapa_dir: Path,
                         init_model: str, device: str):
    """Load OR-PIT + count classifier + ECAPA and return the three closures
    (forward_fn, prob_multi_fn, embed_fn) that separate_longform needs.

    Shared by the CLI and the Phase 6 demo pipeline so model wiring lives once.
    """
    from separate_recursive_blind import resolve_device
    from train_orpit import separate_forward, neutralize_lazy_modules
    from count_classifier import SpeakerCountCNN, MAX_SPEAKERS
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    device = resolve_device(device)
    savedir = Path("pretrained_models") / init_model.split("/")[-1]
    sep = SepformerSeparation.from_hparams(
        source=init_model, savedir=str(savedir),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    neutralize_lazy_modules()

    # Missing weights are a hard error, never a silent fallback: without the
    # OR-PIT checkpoint the heads are not "one-and-rest" (recursion is
    # meaningless), and a stubbed classifier would make blind separation return
    # the untouched mixture while still looking like it worked. Fail loudly and
    # say how to fix it (README "Setup on a new device", step 6).
    if not orpit_ckpt.is_file():
        raise FileNotFoundError(
            f"OR-PIT checkpoint not found: {orpit_ckpt}. Checkpoints are not "
            f"in git — copy them from the training machine (see README setup "
            f"step 6).")
    if not clf_ckpt.is_file():
        raise FileNotFoundError(
            f"Count/stop classifier not found: {clf_ckpt}. Checkpoints are not "
            f"in git — copy them from the training machine (see README setup "
            f"step 6).")

    oc = torch.load(str(orpit_ckpt), map_location=device)
    enc, mn, dec = sep.mods.encoder, sep.mods.masknet, sep.mods.decoder
    enc.load_state_dict(oc["encoder"])
    mn.load_state_dict(oc["masknet"])
    dec.load_state_dict(oc["decoder"])
    for m in (enc, mn, dec):
        m.eval()

    cc = torch.load(str(clf_ckpt), map_location=device)
    clf = SpeakerCountCNN(num_classes=cc.get("num_classes", MAX_SPEAKERS))
    clf.load_state_dict(cc["model"])
    clf.to(device).eval()

    def prob_multi_fn(sig: np.ndarray) -> float:
        x = torch.from_numpy(np.ascontiguousarray(sig)).unsqueeze(0).to(device)
        return float(clf.prob_multi(x)[0])

    # ECAPA, on the other hand, is safe to auto-download (kept from the fork):
    # it is a public pretrained model, not one of our trained checkpoints.
    if not (ecapa_dir / "embedding_model.ckpt").is_file():
        print(f"Downloading ECAPA weights from HuggingFace to {ecapa_dir}...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download('speechbrain/spkrec-ecapa-voxceleb', local_dir=str(ecapa_dir))
        except Exception as e:
            print(f"Failed to download ECAPA weights: {e}")
            raise FileNotFoundError(
                f"ECAPA weights not found in {ecapa_dir}. Pre-fetch once with "
                f"huggingface_hub.snapshot_download('speechbrain/"
                f"spkrec-ecapa-voxceleb', local_dir='{ecapa_dir}').")

    ecapa = EncoderClassifier.from_hparams(
        source=str(ecapa_dir), savedir=str(ecapa_dir),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)

    def forward_fn(sig: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(sig)).unsqueeze(0).to(device)
        with torch.no_grad():
            est = separate_forward(enc, mn, dec, x)
        return est.cpu().numpy()[0].T

    # prob_multi_fn is defined above, next to the classifier it closes over.

    def embed_fn(sig: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(sig)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = ecapa.encode_batch(x)
        return emb.squeeze().cpu().numpy()

    return forward_fn, prob_multi_fn, embed_fn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Long-audio separation with ECAPA identity stitching.")
    parser.add_argument("input", nargs="?", type=Path,
                        help="Input recording (any rate/channels).")
    parser.add_argument("--orpit-ckpt", type=Path)
    parser.add_argument("--clf-ckpt", type=Path)
    parser.add_argument("--init-model",
                        default="speechbrain/sepformer-wsj02mix")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--chunk-seconds", type=float, default=4.0,
                        help="Chunk length; kept near the 3 s training segment "
                             "so the separator and count classifier stay in "
                             "domain.")
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    parser.add_argument("--cluster-threshold", type=float, default=0.55,
                        help="Cosine-distance threshold to match a chunk "
                             "estimate to an existing global speaker.")
    parser.add_argument("--recursion-threshold", type=float, default=0.5,
                        help="Per-chunk stop threshold for blind recursion.")
    parser.add_argument("--ecapa-dir", type=Path,
                        default=Path("pretrained_models/ecapa-dl"),
                        help="Local ECAPA directory with hyperparams.yaml and "
                             "*.ckpt (pre-fetch via huggingface_hub).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if not (args.input and args.orpit_ckpt and args.clf_ckpt and args.out_dir):
        print("input, --orpit-ckpt, --clf-ckpt and --out-dir are required.")
        return 1
    if not args.input.is_file():
        print(f"Input file not found: {args.input}")
        return 1

    from separate_recursive_blind import resolve_device
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    try:
        forward_fn, prob_multi_fn, embed_fn = load_longform_models(
            args.orpit_ckpt, args.clf_ckpt, args.ecapa_dir, args.init_model,
            device)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    signal, orig_sr = load_normalized(args.input, MODEL_SR)
    chunk_len = int(round(args.chunk_seconds * MODEL_SR))
    hop_len = max(int(round((args.chunk_seconds - args.overlap_seconds)
                            * MODEL_SR)), 1)
    print(f"Loaded {args.input} (orig sr={orig_sr}, "
          f"{len(signal) / MODEL_SR:.1f}s at {MODEL_SR} Hz); "
          f"chunk={args.chunk_seconds}s hop={hop_len / MODEL_SR:.1f}s.")

    tracks = separate_longform(
        signal, forward_fn, prob_multi_fn, embed_fn, chunk_len, hop_len,
        cluster_threshold=args.cluster_threshold,
        recursion_threshold=args.recursion_threshold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("speaker*.wav"):
        old.unlink()  # avoid stale tracks from a previous, larger-count run
    for j, tr in enumerate(tracks, start=1):
        sf.write(str(args.out_dir / f"speaker{j}.wav"),
                 tr.astype(np.float32), MODEL_SR, subtype="FLOAT")
    print(f"Detected {len(tracks)} speaker(s) across the file; wrote "
          f"{len(tracks)} track(s) to {args.out_dir}.")
    return 0


def _self_test() -> int:
    """Chunking + clustering + overlap-add reassemble 2 speakers over a file.

    Two distinct-frequency "speakers" span a 30 s signal; ideal mocks stand in
    for the model and ECAPA so the stitching logic is what is under test.
    """
    sr = MODEL_SR
    dur = 30 * sr
    t = np.arange(dur) / float(sr)
    freqs = [200.0, 500.0]
    sources = [np.sin(2 * np.pi * f * t).astype(np.float32) for f in freqs]
    mixture = np.sum(sources, axis=0).astype(np.float32)

    def amp_at(sig, f):
        n = len(sig)
        tt = np.arange(n) / float(sr)
        return abs(float(2.0 * np.dot(sig, np.sin(2 * np.pi * f * tt)) / n))

    def present(sig):
        return [i for i in range(len(freqs)) if amp_at(sig, freqs[i]) > 0.3]

    def forward_fn(residual):
        p = present(residual)
        one_idx = min(p)
        n = len(residual)
        tt = np.arange(n) / float(sr)
        one = np.sin(2 * np.pi * freqs[one_idx] * tt).astype(np.float32)
        rest = (np.sum([np.sin(2 * np.pi * freqs[i] * tt) for i in p
                        if i != one_idx], axis=0).astype(np.float32)
                if len(p) > 1 else np.zeros(n, np.float32))
        return np.stack([one, rest])

    def prob_multi_fn(sig):
        return 1.0 if len(present(sig)) >= 2 else 0.0

    def embed_fn(sig):
        # Identity embedding = which tone dominates (well separated clusters).
        e = np.array([amp_at(sig, f) for f in freqs], dtype=np.float64)
        return e / (np.linalg.norm(e) + 1e-9)

    chunk_len = 10 * sr
    hop_len = 8 * sr
    tracks = separate_longform(mixture, forward_fn, prob_multi_fn, embed_fn,
                               chunk_len, hop_len, cluster_threshold=0.3)
    assert len(tracks) == 2, len(tracks)
    assert all(len(tr) == dur for tr in tracks), [len(x) for x in tracks]

    # Each recovered track should match one source over the full length.
    matched = 0
    for src in sources:
        best = max(_si(tr, src) for tr in tracks)
        if best > 10.0:
            matched += 1
    assert matched == 2, matched
    print(f"reassembled {len(tracks)} full-length tracks, both match a source")
    print("All separate_longform self-tests passed.")
    return 0


def _si(est, ref):
    """Tiny SI-SDR for the self-test (avoids importing the eval package)."""
    est = np.asarray(est, np.float64)
    ref = np.asarray(ref, np.float64)
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    a = float(np.dot(est, ref)) / (float(np.dot(ref, ref)) + 1e-12)
    proj = a * ref
    noise = est - proj
    return 10.0 * np.log10((np.dot(proj, proj) + 1e-12)
                           / (np.dot(noise, noise) + 1e-12))


if __name__ == "__main__":
    raise SystemExit(main())
