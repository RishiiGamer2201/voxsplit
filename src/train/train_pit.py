"""Fixed-N uPIT fine-tuning of a SepFormer model (VoxSplit Phase 3 baseline).

This is the standard permutation invariant training baseline: warm-start from
a pretrained N-output SepFormer (for example speechbrain/sepformer-libri3mix,
which has 3 heads) and fine-tune its encoder, masknet, and decoder with
utterance-level PIT (see pit_loss.py) on on-the-fly N-speaker mixtures (see
mix_dataset.py). Unlike OR-PIT, every mixture has exactly N sources and the N
heads are matched one to one to the N references.

This is plain PyTorch: SpeechBrain is used only to load the pretrained
submodules, not its Brain/hparams training loop.

Example (real training on LibriSpeech train-clean-100, 3 speakers):
  python src/train/train_pit.py \
      --source-dir data/LibriSpeech/train-clean-100 \
      --out-dir checkpoints/pit_libri3mix \
      --init-model speechbrain/sepformer-libri3mix --num-speakers 3 \
      --sample-rate 8000 --segment-seconds 3.0 \
      --batch-size 1 --lr 1e-4 --epochs 1 --steps-per-epoch 1000 \
      --ckpt-every 500 --log-every 20 --device auto --seed 0
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Cross-folder imports like train_orpit.py does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mix_dataset import MixDataset, collate  # noqa: E402
from orpit_loss import si_snr  # noqa: E402
from pit_loss import batch_pit_loss  # noqa: E402
from train_orpit import neutralize_lazy_modules, resolve_device  # noqa: E402
from wandb_logger import WandbLogger  # noqa: E402


def expand_masknet_heads(masknet, num_speakers: int) -> None:
    """Re-shape a SepFormer Dual_Path_Model masknet to num_speakers heads.

    The per-speaker head count lives only in masknet.num_spks and in the
    conv2d output channels (out = base * num_spks, where base is the model
    dimension). To go from the pretrained head count to num_speakers, this
    builds a new conv2d and warm-starts every new head by copying an existing
    head (cycled), so the extra heads start as duplicates of real speaker
    heads rather than random noise, which trains far faster. A no-op if the
    head count already matches.
    """
    import torch.nn as nn

    old = masknet.conv2d
    old_spks = int(masknet.num_spks)
    if old_spks == num_speakers:
        return
    base = old.out_channels // old_spks
    device = old.weight.device
    new = nn.Conv2d(old.in_channels, base * num_speakers,
                    kernel_size=old.kernel_size, stride=old.stride).to(device)
    with torch.no_grad():
        for i in range(num_speakers):
            src = i % old_spks
            new.weight[i * base:(i + 1) * base] = \
                old.weight[src * base:(src + 1) * base]
            if old.bias is not None and new.bias is not None:
                new.bias[i * base:(i + 1) * base] = \
                    old.bias[src * base:(src + 1) * base]
    masknet.conv2d = new
    masknet.num_spks = num_speakers
    print(f"Expanded masknet heads {old_spks} -> {num_speakers} "
          f"(copied existing heads as warm start).")


def load_warm_start(init_model: str, num_speakers: int, device: str):
    """Load the pretrained N-output SepFormer submodules for fine-tuning.

    The SpeechBrain inference wrapper freezes parameters (requires_grad=False),
    so grad is re-enabled here. If the pretrained head count differs from
    num_speakers (for example warm-starting 4- or 5-speaker models from the
    3-head libri3mix), the masknet is expanded first. Returns (encoder,
    masknet, decoder) in train() mode. Imported inside the function so that
    --help stays fast.
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
    expand_masknet_heads(masknet, num_speakers)
    for module in (encoder, masknet, decoder):
        module.train()
        for param in module.parameters():
            param.requires_grad_(True)
    return encoder, masknet, decoder


def sepformer_forward(encoder, masknet, decoder, mix: torch.Tensor,
                      num_spks: int) -> torch.Tensor:
    """SepFormer forward generalized to num_spks heads.

    mix: [B, T] -> est: [B, T2, num_spks]. The masknet output leading dim is
    the actual head count; it is asserted to equal num_spks. T2 can differ
    slightly from T, so the caller truncates est and targets to a common
    length.
    """
    mix_w = encoder(mix)                       # [B, F, L]
    est_mask = masknet(mix_w)                  # [num_spks, B, F, L]
    actual = est_mask.shape[0]
    assert actual == num_spks, (
        f"masknet emitted {actual} heads but num_spks={num_spks}.")
    sep_h = torch.stack([mix_w] * num_spks) * est_mask
    est = torch.cat(
        [decoder(sep_h[i]).unsqueeze(-1) for i in range(num_spks)], dim=-1)
    return est                                 # [B, T2, num_spks]


def si_snri_for_item(outputs: torch.Tensor, targets: torch.Tensor,
                     mixture: torch.Tensor,
                     best_perm) -> float:
    """SI-SNR improvement for one item under its best PIT permutation.

    Matched mean SI-SNR of the winning permutation minus the
    mixture-vs-target baseline. outputs [N, T], targets [N, T], mixture [T].
    """
    n = min(outputs.shape[-1], targets.shape[-1], mixture.shape[-1])
    outputs = outputs[:, :n]
    targets = targets[:, :n]
    mixture = mixture[:n]
    num = outputs.shape[0]

    matched = 0.0
    baseline = 0.0
    for i in range(num):
        ref = targets[best_perm[i]]
        matched += float(si_snr(outputs[i], ref))
        baseline += float(si_snr(mixture, ref))
    matched /= num
    baseline /= num
    return matched - baseline


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
        description="Fixed-N uPIT fine-tuning of SepFormer (plain PyTorch).")
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="Directory of single-speaker source speech "
                             "(searched recursively for .wav and .flac).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("checkpoints/pit_libri3mix"),
                        help="Directory for checkpoints and train_log.csv.")
    parser.add_argument("--init-model",
                        default="speechbrain/sepformer-libri3mix",
                        help="Pretrained N-output SepFormer to warm-start "
                             "from.")
    parser.add_argument("--num-speakers", type=int, default=3,
                        help="Fixed number of sources per mixture and heads.")
    parser.add_argument("--sample-rate", type=int, default=8000,
                        help="Target sample rate in Hz.")
    parser.add_argument("--segment-seconds", type=float, default=3.0,
                        help="Training segment length in seconds.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Mixtures per batch (a 3-output SepFormer on 3 s "
                             "at 8 kHz is memory heavy; raise if you can).")
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

    num_speakers = int(args.num_speakers)

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * args.steps_per_epoch
    print(f"Planned total steps: {total_steps}")

    print(f"Warm-starting from {args.init_model} ...")
    encoder, masknet, decoder = load_warm_start(
        args.init_model, num_speakers, device)

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

    params = (list(encoder.parameters())
              + list(masknet.parameters())
              + list(decoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)

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
        config=vars(args), name=f"pit{num_speakers}spk_to{total_steps}")

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

            est = sepformer_forward(
                encoder, masknet, decoder, mixture, num_speakers)
            # est is [B, T2, N] -> [B, N, T2] for the loss.
            outputs = est.permute(0, 2, 1).contiguous()

            # Truncate targets to the estimate length before scoring.
            length = min(outputs.shape[-1], targets.shape[-1])
            outputs_c = outputs[..., :length]
            targets_c = targets[..., :length]

            loss, best_perms = batch_pit_loss(outputs_c, targets_c)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
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
