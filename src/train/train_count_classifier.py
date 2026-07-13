"""Train the Phase 4 speaker-count / stop classifier.

Trains SpeakerCountCNN (src/models/count_classifier.py) to predict the number
of active speakers (K in 1..5) from a waveform, using the same on-the-fly
mixtures as the separators (mix_dataset.py) but including K=1 (a single clean
utterance). The label is num_speakers-1; cross-entropy loss. The default
speaker-count distribution over-samples low counts because the decision that
matters most for recursive stopping is 1 vs 2 speakers.

Note: this trains on clean mixtures. At inference the classifier is applied to
OR-PIT residual "rest" heads, which carry separation artifacts, so there is a
domain gap; blind-recursion results quantify how much it costs. Refining on
real residuals is a follow-up if the gap is large.

Example:
  python src/train/train_count_classifier.py \
      --source-dir data/LibriSpeech/train-clean-100 \
      --out-dir checkpoints/count_clf --max-steps 6000 \
      --batch-size 8 --lr 1e-3 --device auto --seed 0
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from mix_dataset import MixDataset, collate  # noqa: E402
from train_orpit import resolve_device  # noqa: E402
from wandb_logger import WandbLogger  # noqa: E402
from count_classifier import SpeakerCountCNN, MAX_SPEAKERS  # noqa: E402


def parse_counts(text: str):
    counts = tuple(int(x) for x in text.split(",") if x.strip())
    if not counts:
        raise ValueError("speaker-counts must list at least one integer.")
    return counts


def save_checkpoint(out_dir: Path, step: int, model, optimizer) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ckpt_step{step}.pt"
    torch.save({"step": step, "num_classes": model.num_classes,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, str(path))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the speaker-count / stop classifier.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("checkpoints/count_clf"))
    parser.add_argument("--speaker-counts", default="1,1,2,2,3,4,5",
                        help="Counts sampled per item; low counts repeated to "
                             "balance the crucial 1-vs-2 boundary.")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="If > 0 stop after this many steps. 0 means "
                             "--epochs * --steps-per-epoch.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="voxsplit")
    parser.add_argument("--wandb-mode", default="offline",
                        choices=["offline", "online", "disabled"])
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Source directory not found: {args.source_dir}")
        return 1

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    total_steps = (args.max_steps if args.max_steps > 0
                   else args.epochs * args.steps_per_epoch)
    print(f"Planned total steps: {total_steps}")

    model = SpeakerCountCNN(sample_rate=args.sample_rate,
                            num_classes=MAX_SPEAKERS).to(device)
    model.train()

    counts = parse_counts(args.speaker_counts)
    ds_length = max(total_steps * args.batch_size * 2, args.batch_size * 4)
    dataset = MixDataset(
        source_dir=args.source_dir, sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds, speaker_counts=counts,
        seed=args.seed, length=ds_length)
    print(f"Dataset scanned {len(dataset.speakers)} speakers; "
          f"counts sampled from {counts}.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "train_log.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fh)
    if write_header:
        csv_writer.writerow(["step", "loss", "running_loss", "running_acc"])
        csv_fh.flush()

    wandb_log = WandbLogger(
        enabled=args.wandb, project=args.wandb_project, mode=args.wandb_mode,
        config=vars(args), name=f"count_clf_to{total_steps}")

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    step = 0
    stop = False

    while not stop:
        for batch in loader:
            step += 1
            mixture = batch["mixture"].to(device)                 # [B, T]
            labels = torch.tensor(batch["num_speakers"], device=device) - 1

            logits = model(mixture)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                running_correct += int((pred == labels).sum())
                running_total += labels.shape[0]
            running_loss += float(loss)

            if step % args.log_every == 0:
                mean_loss = running_loss / (args.log_every)
                acc = running_correct / max(running_total, 1)
                print(f"step {step}/{total_steps}  loss={float(loss):.4f}  "
                      f"running_loss={mean_loss:.4f}  acc={acc:.3f}")
                csv_writer.writerow([step, f"{float(loss):.6f}",
                                     f"{mean_loss:.6f}", f"{acc:.6f}"])
                csv_fh.flush()
                wandb_log.log({"loss": float(loss), "running_loss": mean_loss,
                               "acc": acc}, step=step)
                running_loss = 0.0
                running_correct = 0
                running_total = 0

            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                p = save_checkpoint(args.out_dir, step, model, optimizer)
                print(f"Saved checkpoint: {p}")

            if step >= total_steps:
                stop = True
                break

    p = save_checkpoint(args.out_dir, step, model, optimizer)
    print(f"Saved final checkpoint: {p}")
    csv_fh.close()
    wandb_log.finish()
    print(f"Training complete. Log at {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
