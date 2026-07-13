"""Train the compact TF-GridNet from scratch with PIT (Phase 3, TF-domain).

Mirrors train_convtasnet.py but swaps in the time-frequency TF-GridNet
(src/models/tfgridnet.py). Like Conv-TasNet this is trained from random init
with utterance-level PIT (pit_loss.py) on on-the-fly N-speaker mixtures
(mix_dataset.py); there is no usable pretrained 3-speaker TF-GridNet to warm
from (public checkpoints are WSJ0-2mix, license-restricted). It is a
learning-scale TF-domain baseline for the 3-speaker level, reported honestly
alongside the Conv-TasNet warm-up rather than as a SOTA contender.

Example (3 speakers on LibriSpeech train-clean-100):
  python src/train/train_tfgridnet.py \
      --source-dir data/LibriSpeech/train-clean-100 \
      --out-dir checkpoints/tfgridnet --num-speakers 3 \
      --sample-rate 8000 --segment-seconds 3.0 \
      --batch-size 2 --lr 1e-3 --max-steps 8000 \
      --ckpt-every 2000 --log-every 50 --device auto --seed 0
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Cross-folder imports like train_convtasnet.py does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from mix_dataset import MixDataset, collate  # noqa: E402
from pit_loss import batch_pit_loss  # noqa: E402
from train_orpit import resolve_device  # noqa: E402
from train_pit import si_snri_for_item  # noqa: E402
from wandb_logger import WandbLogger  # noqa: E402
from tfgridnet import TFGridNet  # noqa: E402


def save_checkpoint(out_dir: Path, step: int, model, optimizer,
                    num_speakers: int) -> Path:
    """Save the TF-GridNet model and optimizer state to a step checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ckpt_step{step}.pt"
    torch.save({
        "step": step,
        "num_speakers": num_speakers,
        "config": {"num_spks": model.num_spks, "n_fft": model.n_fft,
                   "hop": model.hop, "dim": model.dim,
                   "hidden": model.hidden, "num_blocks": model.num_blocks},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, str(path))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train compact TF-GridNet from scratch with PIT.")
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="Directory of single-speaker source speech.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("checkpoints/tfgridnet"),
                        help="Directory for checkpoints and train_log.csv.")
    parser.add_argument("--num-speakers", type=int, default=3,
                        help="Fixed number of sources per mixture and heads.")
    parser.add_argument("--sample-rate", type=int, default=8000,
                        help="Target sample rate in Hz.")
    parser.add_argument("--segment-seconds", type=float, default=3.0,
                        help="Training segment length in seconds.")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Mixtures per batch.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--num-blocks", type=int, default=4,
                        help="Number of TF-GridNet blocks.")
    parser.add_argument("--dim", type=int, default=48,
                        help="Model channel dimension.")
    parser.add_argument("--hidden", type=int, default=64,
                        help="BLSTM hidden size.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="If > 0, stop after this many steps. 0 means "
                             "run --epochs of --steps-per-epoch.")
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
    parser.add_argument("--wandb", action="store_true",
                        help="Also log to Weights & Biases (off by default).")
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

    num_speakers = int(args.num_speakers)

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * args.steps_per_epoch
    print(f"Planned total steps: {total_steps}")

    print(f"Building TF-GridNet from scratch (num_spks={num_speakers}) ...")
    model = TFGridNet(num_spks=num_speakers, dim=args.dim, hidden=args.hidden,
                      num_blocks=args.num_blocks).to(device)
    model.train()

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

    wandb_log = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, mode=args.wandb_mode,
        config=vars(args), name=f"tfgridnet{num_speakers}spk_to{total_steps}")

    running_loss = 0.0
    running_si_snri = 0.0
    running_count = 0
    step = 0
    stop = False

    while not stop:
        for batch in loader:
            step += 1
            mixture = batch["mixture"].to(device)            # [B, T]
            targets = torch.stack(
                [s.to(device) for s in batch["sources"]], dim=0)

            est = model(mixture)                             # [B, N, T]

            length = min(est.shape[-1], targets.shape[-1])
            outputs_c = est[..., :length]
            targets_c = targets[..., :length]

            loss, best_perms = batch_pit_loss(outputs_c, targets_c)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

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
                wandb_log.log({"loss": float(loss),
                               "running_loss": mean_loss,
                               "running_si_snri": mean_si_snri}, step=step)
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

    path = save_checkpoint(args.out_dir, step, model, optimizer, num_speakers)
    print(f"Saved final checkpoint: {path}")
    csv_fh.close()
    wandb_log.finish()
    print(f"Training complete. Log at {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
