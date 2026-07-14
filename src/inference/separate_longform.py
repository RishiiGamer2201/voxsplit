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


def cluster_embeddings(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Agglomerative clustering on cosine distance; labels for each embedding.

    The cluster count is chosen automatically by the distance threshold, so it
    reflects the union of speakers seen across chunks. One embedding -> one
    cluster.
    """
    if len(embeddings) == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=threshold)
    return model.fit_predict(embeddings)


def separate_longform(
        signal: np.ndarray,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        prob_multi_fn: Callable[[np.ndarray], float],
        embed_fn: Callable[[np.ndarray], np.ndarray],
        chunk_len: int,
        hop_len: int,
        cluster_threshold: float = 0.6,
        ramp: int = 400) -> List[np.ndarray]:
    """Separate a long signal into whole-file per-speaker tracks.

    Returns a list of G full-length tracks, one per clustered global speaker.
    """
    n = len(signal)
    windows = chunk_signal(signal, chunk_len, hop_len)

    # Separate every chunk and embed every chunk-speaker.
    items = []              # (start, wav, embedding)
    for start, chunk in windows:
        ests = blind_recursive_separate(chunk, forward_fn, prob_multi_fn)
        for wav in ests:
            emb = np.asarray(embed_fn(wav), dtype=np.float64).flatten()
            items.append((start, np.asarray(wav, dtype=np.float32), emb))

    if not items:
        return [np.zeros(n, dtype=np.float32)]

    labels = cluster_embeddings(np.stack([e for _, _, e in items]),
                                cluster_threshold)
    num_speakers = int(labels.max()) + 1

    # Overlap-add each chunk-speaker into its cluster track, weight-normalized.
    outputs = [np.zeros(n, dtype=np.float64) for _ in range(num_speakers)]
    weights = [np.zeros(n, dtype=np.float64) for _ in range(num_speakers)]
    for (start, wav, _), g in zip(items, labels):
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
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--cluster-threshold", type=float, default=0.6,
                        help="Cosine-distance threshold for ECAPA clustering.")
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
    from train_orpit import separate_forward, neutralize_lazy_modules
    from count_classifier import SpeakerCountCNN, MAX_SPEAKERS

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    savedir = Path("pretrained_models") / args.init_model.split("/")[-1]
    sep = SepformerSeparation.from_hparams(
        source=args.init_model, savedir=str(savedir),
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    neutralize_lazy_modules()
    oc = torch.load(str(args.orpit_ckpt), map_location=device)
    enc, mn, dec = sep.mods.encoder, sep.mods.masknet, sep.mods.decoder
    enc.load_state_dict(oc["encoder"])
    mn.load_state_dict(oc["masknet"])
    dec.load_state_dict(oc["decoder"])
    for m in (enc, mn, dec):
        m.eval()

    cc = torch.load(str(args.clf_ckpt), map_location=device)
    clf = SpeakerCountCNN(num_classes=cc.get("num_classes", MAX_SPEAKERS))
    clf.load_state_dict(cc["model"])
    clf.to(device).eval()

    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa-voxceleb",
        run_opts={"device": device}, local_strategy=LocalStrategy.COPY)

    def forward_fn(sig: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(sig)).unsqueeze(0).to(device)
        with torch.no_grad():
            est = separate_forward(enc, mn, dec, x)
        return est.cpu().numpy()[0].T

    def prob_multi_fn(sig: np.ndarray) -> float:
        x = torch.from_numpy(np.ascontiguousarray(sig)).unsqueeze(0).to(device)
        return float(clf.prob_multi(x)[0])

    def embed_fn(sig: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(sig)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = ecapa.encode_batch(x)
        return emb.squeeze().cpu().numpy()

    signal, orig_sr = load_normalized(args.input, MODEL_SR)
    chunk_len = int(round(args.chunk_seconds * MODEL_SR))
    hop_len = max(int(round((args.chunk_seconds - args.overlap_seconds)
                            * MODEL_SR)), 1)
    print(f"Loaded {args.input} (orig sr={orig_sr}, "
          f"{len(signal) / MODEL_SR:.1f}s at {MODEL_SR} Hz); "
          f"chunk={args.chunk_seconds}s hop={hop_len / MODEL_SR:.1f}s.")

    tracks = separate_longform(
        signal, forward_fn, prob_multi_fn, embed_fn, chunk_len, hop_len,
        cluster_threshold=args.cluster_threshold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
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
