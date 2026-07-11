"""Train a torchaudio Conv-TasNet from scratch with PIT (Phase 3 warm-up).

This is the learning baseline: unlike train_pit.py (which fine-tunes a
pretrained SepFormer), this trains torchaudio's Conv-TasNet with randomly
initialized weights using utterance-level PIT (see pit_loss.py) on on-the-fly
N-speaker mixtures (see mix_dataset.py). It converges quickly and is a useful
sanity check that the data pipeline and loss learn something end to end.

The model forward takes [B, 1, T] and returns [B, num_sources, T].

Example (real training on LibriSpeech train-clean-100, 2 speakers):
  python src/train/train_convtasnet.py \
      --source-dir data/LibriSpeech/train-clean-100 \
      --out-dir checkpoints/convtasnet --num-speakers 2 \
      --sample-rate 8000 --segment-seconds 3.0 \
      --batch-size 4 --lr 1e-3 --epochs 1 --steps-per-epoch 1000 \
      --ckpt-every 500 --log-every 20 --device auto --seed 0
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchaudio.models import ConvTasNet

# Cross-folder imports like train_orpit.py does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_dataset import MixDataset, collate  # noqa: E402
from pit_loss import batch_pit_loss  # noqa: E402
from train_orpit import resolve_device  # noqa: E402
from train_pit import si_snri_for_item  # noqa: E402


def save_checkpoint(out_dir: Path, step: int, model, optimizer,
                    num_speakers: int) -> Path:
    """Save the Conv-TasNet model and optimizer state to a step checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ckpt_step{step}.pt"
    torch.save({
        "step": step,
        "num_speakers": num_speakers,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, str(path))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train torchaudio Conv-TasNet from scratch with PIT.")
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="Directory of single-speaker source speech "
                             "(searched recursively for .wav and .flac).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("checkpoints/convtasnet"),
                        help="Directory for checkpoints and train_log.csv.")
    parser.add_argument("--num-speakers", type=int, default=2,
                        help="Fixed number of sources per mixture and heads.")
    parser.add_argument("--sample-rate", type=int, default=8000,
                        help="Target sample rate in Hz.")
    parser.add_argument("--segment-seconds", type=float, default=3.0,
                        help="Training segment length in seconds.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Mixtures per batch.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (typical Conv-TasNet).")
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
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Source directory not found: {args.source_dir}")
        return 1

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    num_speakers = int(args.num_speakers)

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * args.steps_per_epoch
    print(f"Planned total steps: {total_steps}")

    print(f"Building Conv-TasNet from scratch (num_sources={num_speakers}) ...")
    model = ConvTasNet(num_sources=num_speakers).to(device)
    model.train()

    # A generous dataset length so the DataLoader never runs dry.
    ds_length = max(total_steps * args.batch_size * 2, args.batch_size * 4)
    dataset = MixDataset(
        source_dir=args.source_dir,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        speaker_counts=(num_speakers,),
        seed=args.seed,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "train_log.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fh)
    if write_header:
        csv_writer.writerow(
            ["step", "loss", "running_loss", "running_si_snri"])
        csv_fh.flush()

    running_loss = 0.0
    running_si_snri = 0.0
    running_count = 0
    step = 0
    stop = False

    while not stop:
        for batch in loader:
            step += 1
            mixture = batch["mixture"].to(device)            # [B, T]
            # Every item has exactly num_speakers sources, so stack to [B,N,T].
            targets = torch.stack(
                [s.to(device) for s in batch["sources"]], dim=0)

            est = model(mixture.unsqueeze(1))                # [B, N, T]

            # Truncate to a common length before scoring.
            length = min(est.shape[-1], targets.shape[-1])
            outputs_c = est[..., :length]
            targets_c = targets[..., :length]

            loss, best_perms = batch_pit_loss(outputs_c, targets_c)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # SI-SNRi over the batch (no grad needed).
            with torch.no_grad():
                batch_si_snri = 0.0
                b = outputs_c.shape[0]
                for i in range(b):
                    batch_si_snri += si_snri_for_item(
                        outputs_c[i].detach(), targets_c[i], mixture[i],
                        best_perms[i])
                batch_si_snri /= b

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
                running_loss = 0.0
                running_si_snri = 0.0
                running_count = 0

            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                path = save_checkpoint(args.out_dir, step, model, optimizer,
                                       num_speakers)
                print(f"Saved checkpoint: {path}")

            if step >= total_steps:
                stop = True
                break

    # Final checkpoint.
    path = save_checkpoint(args.out_dir, step, model, optimizer, num_speakers)
    print(f"Saved final checkpoint: {path}")
    csv_fh.close()
    print(f"Training complete. Log at {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
