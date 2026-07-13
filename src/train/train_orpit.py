"""OR-PIT fine-tuning of SepFormer for VoxSplit Phase 3 (plain PyTorch).

Warm-starts from speechbrain/sepformer-wsj02mix (which already has exactly two
output heads) and fine-tunes its encoder, masknet, and decoder with the
one-and-rest permutation invariant loss (see orpit_loss.py) on on-the-fly 2-
and 3-speaker mixtures (see mix_dataset.py).

This is plain PyTorch: SpeechBrain is used only to load the pretrained
submodules, not its Brain/hparams training loop.

Example (real training on LibriSpeech train-clean-100):
  python src/train/train_orpit.py \
      --source-dir data/LibriSpeech/train-clean-100 \
      --out-dir checkpoints/orpit \
      --sample-rate 8000 --segment-seconds 3.0 --speaker-counts 2,3 \
      --batch-size 2 --lr 1e-4 --epochs 1 --steps-per-epoch 1000 \
      --ckpt-every 500 --log-every 20 --device auto --seed 0
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

# Cross-folder imports like src/eval/evaluate.py does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_dataset import MixDataset, collate  # noqa: E402
from orpit_loss import batch_orpit_loss, si_snr  # noqa: E402
from wandb_logger import WandbLogger  # noqa: E402


def resolve_device(device: str) -> str:
    """Turn the requested device string into a concrete torch device."""
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def neutralize_lazy_modules() -> None:
    """Drop SpeechBrain lazy-module proxies that raise on attribute access.

    SpeechBrain registers lazy import proxies (for optional extras like k2)
    in sys.modules. During the first backward, torch lazily imports pieces of
    torch.distributed and registers custom-op fakes; that registration calls
    inspect.getmodule, which does hasattr(mod, '__file__') on every entry in
    sys.modules. For an uninstalled extra, the proxy raises ImportError there
    (not AttributeError), which crashes the backward. Removing the proxies
    from sys.modules is safe (they were never really loaded) and lets
    inspect.getmodule skip them.
    """
    for name in list(sys.modules):
        if not name.startswith("speechbrain"):
            continue
        mod = sys.modules.get(name)
        try:
            hasattr(mod, "__file__")
        except Exception:
            sys.modules.pop(name, None)


def load_warm_start(init_model: str, device: str):
    """Load the pretrained SepFormer submodules for warm-start fine-tuning.

    The SpeechBrain inference wrapper freezes parameters (requires_grad=False),
    so grad is re-enabled here for fine-tuning. Returns (encoder, masknet,
    decoder) in train() mode. Imported inside the function so that --help stays
    fast and does not load heavy modules.
    """
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.utils.fetching import LocalStrategy

    savedir = Path("pretrained_models") / init_model.split("/")[-1]
    sep = SepformerSeparation.from_hparams(
        source=init_model,
        savedir=str(savedir),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    neutralize_lazy_modules()

    encoder = sep.mods.encoder
    masknet = sep.mods.masknet
    decoder = sep.mods.decoder
    for module in (encoder, masknet, decoder):
        module.train()
        for param in module.parameters():
            param.requires_grad_(True)
    return encoder, masknet, decoder


def separate_forward(encoder, masknet, decoder,
                     mix: torch.Tensor) -> torch.Tensor:
    """SepFormer forward matching the SpeechBrain sepformer recipe.

    mix: [B, T] -> est: [B, T2, 2]. T2 can differ slightly from T; the caller
    truncates est and targets to a common length.
    """
    mix_w = encoder(mix)                      # [B, F, L]
    est_mask = masknet(mix_w)                 # [2, B, F, L]
    sep_h = torch.stack([mix_w] * 2) * est_mask
    est = torch.cat(
        [decoder(sep_h[i]).unsqueeze(-1) for i in range(2)], dim=-1)
    return est                                # [B, T2, 2]


def si_snri_for_item(outputs: torch.Tensor, sources: torch.Tensor,
                     mixture: torch.Tensor, best_j: int) -> float:
    """SI-SNR improvement for one item under its chosen "one" speaker.

    Uses the best head assignment for (one, rest) and subtracts the
    mixture-vs-target baseline. outputs [2, T], sources [K, T], mixture [T].
    """
    n = min(outputs.shape[-1], sources.shape[-1], mixture.shape[-1])
    outputs = outputs[:, :n]
    sources = sources[:, :n]
    mixture = mixture[:n]

    one = sources[best_j]
    rest = sources.sum(dim=0) - one

    a = float(si_snr(outputs[0], one) + si_snr(outputs[1], rest))
    b = float(si_snr(outputs[1], one) + si_snr(outputs[0], rest))
    matched = max(a, b) / 2.0

    baseline = float(si_snr(mixture, one) + si_snr(mixture, rest)) / 2.0
    return matched - baseline


def parse_speaker_counts(text: str) -> Tuple[int, ...]:
    """Parse a comma-separated speaker-count string like '2,3'."""
    counts = tuple(int(x) for x in text.split(",") if x.strip())
    if not counts:
        raise ValueError("speaker-counts must list at least one integer.")
    return counts


def save_checkpoint(out_dir: Path, step: int, encoder, masknet, decoder,
                    optimizer) -> Path:
    """Save encoder/masknet/decoder and optimizer state to a step checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ckpt_step{step}.pt"
    torch.save({
        "step": step,
        "encoder": encoder.state_dict(),
        "masknet": masknet.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, str(path))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OR-PIT fine-tuning of SepFormer (plain PyTorch).")
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="Directory of single-speaker source speech "
                             "(searched recursively for .wav and .flac).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("checkpoints/orpit"),
                        help="Directory for checkpoints and train_log.csv.")
    parser.add_argument("--init-model",
                        default="speechbrain/sepformer-wsj02mix",
                        help="Pretrained SepFormer source to warm-start from.")
    parser.add_argument("--sample-rate", type=int, default=8000,
                        help="Target sample rate in Hz.")
    parser.add_argument("--segment-seconds", type=float, default=3.0,
                        help="Training segment length in seconds.")
    parser.add_argument("--speaker-counts", default="2,3",
                        help="Comma-separated speaker counts to sample from.")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Mixtures per batch.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam learning rate.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="If > 0, stop after this many steps (smoke run). "
                             "0 means run --epochs of --steps-per-epoch.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs when --max-steps is 0.")
    parser.add_argument("--steps-per-epoch", type=int, default=1000,
                        help="Optimizer steps per epoch.")
    parser.add_argument("--ckpt-every", type=int, default=500,
                        help="Save a checkpoint every N steps.")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Log running metrics every N steps.")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the dataset and torch.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint from a previous run to continue from. "
                             "Loads encoder/masknet/decoder and optimizer "
                             "state and treats --max-steps/--epochs*--steps "
                             "as the ABSOLUTE target step to train up to.")
    parser.add_argument("--wandb", action="store_true",
                        help="Also log to Weights & Biases (off by default; "
                             "CSV stays the source of truth).")
    parser.add_argument("--wandb-project", default="voxsplit",
                        help="W&B project name when --wandb is set.")
    parser.add_argument("--wandb-mode", default="offline",
                        choices=["offline", "online", "disabled"],
                        help="W&B mode; offline needs no account.")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Source directory not found: {args.source_dir}")
        return 1

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    speaker_counts = parse_speaker_counts(args.speaker_counts)

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * args.steps_per_epoch
    print(f"Planned total steps: {total_steps}")

    print(f"Warm-starting from {args.init_model} ...")
    encoder, masknet, decoder = load_warm_start(args.init_model, device)

    # Resume: load module weights from a previous checkpoint and continue.
    resume_ckpt = None
    start_step = 0
    if args.resume is not None:
        if not args.resume.is_file():
            print(f"Resume checkpoint not found: {args.resume}")
            return 1
        resume_ckpt = torch.load(str(args.resume), map_location=device)
        encoder.load_state_dict(resume_ckpt["encoder"])
        masknet.load_state_dict(resume_ckpt["masknet"])
        decoder.load_state_dict(resume_ckpt["decoder"])
        start_step = int(resume_ckpt.get("step", 0))
        print(f"Resumed weights from {args.resume} at step {start_step}; "
              f"training up to absolute step {total_steps}.")
        if start_step >= total_steps:
            print("Nothing to do: resume step already >= target steps.")
            return 0

    # A generous dataset length so the DataLoader never runs dry. On resume,
    # offset the seed by start_step so continued training sees fresh mixtures
    # rather than replaying the first run's samples.
    ds_length = max(total_steps * args.batch_size * 2, args.batch_size * 4)
    dataset = MixDataset(
        source_dir=args.source_dir,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        speaker_counts=speaker_counts,
        seed=args.seed + start_step,
        length=ds_length,
    )
    print(f"Dataset scanned {len(dataset.speakers)} speakers.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    params = (list(encoder.parameters())
              + list(masknet.parameters())
              + list(decoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)
    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "train_log.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fh)
    if write_header:
        csv_writer.writerow(
            ["step", "loss", "running_loss", "running_si_snri"])
        csv_fh.flush()

    wandb_log = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, mode=args.wandb_mode,
        config=vars(args), name=f"orpit_to{total_steps}")

    running_loss = 0.0
    running_si_snri = 0.0
    running_count = 0
    step = start_step
    stop = False

    while not stop:
        for batch in loader:
            step += 1
            mixture = batch["mixture"].to(device)            # [B, T]
            sources = [s.to(device) for s in batch["sources"]]

            est = separate_forward(encoder, masknet, decoder, mixture)
            # est is [B, T2, 2] -> [B, 2, T2] for the loss.
            outputs = est.permute(0, 2, 1).contiguous()

            loss, best_js = batch_orpit_loss(outputs, sources)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            # SI-SNRi over the batch (no grad needed).
            with torch.no_grad():
                batch_si_snri = 0.0
                for i in range(len(sources)):
                    batch_si_snri += si_snri_for_item(
                        outputs[i].detach(), sources[i], mixture[i],
                        best_js[i])
                batch_si_snri /= len(sources)

            running_loss += float(loss)
            running_si_snri += float(batch_si_snri)
            running_count += 1

            if step % args.log_every == 0:
                mean_loss = running_loss / running_count
                mean_si_snri = running_si_snri / running_count
                print(f"step {step}/{total_steps}  loss={float(loss):.4f}  "
                      f"running_loss={mean_loss:.4f}  "
                      f"running_si_snri={mean_si_snri:.4f} dB")
                csv_writer.writerow([step, f"{float(loss):.6f}",
                                     f"{mean_loss:.6f}",
                                     f"{mean_si_snri:.6f}"])
                csv_fh.flush()
                wandb_log.log({"loss": float(loss),
                               "running_loss": mean_loss,
                               "running_si_snri": mean_si_snri}, step=step)
                running_loss = 0.0
                running_si_snri = 0.0
                running_count = 0

            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                path = save_checkpoint(args.out_dir, step, encoder, masknet,
                                       decoder, optimizer)
                print(f"Saved checkpoint: {path}")

            if step >= total_steps:
                stop = True
                break

    # Final checkpoint.
    path = save_checkpoint(args.out_dir, step, encoder, masknet, decoder,
                           optimizer)
    print(f"Saved final checkpoint: {path}")
    csv_fh.close()
    wandb_log.finish()
    print(f"Training complete. Log at {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
