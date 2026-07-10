"""Download a LibriSpeech subset for VoxSplit testing and mixture generation.

Default is dev-clean, the smallest clean subset (about 337 MB), which still has
40 speakers, plenty for building 3 to 5 speaker mixtures. Files land in
data/LibriSpeech/<subset>/<speaker>/<chapter>/*.flac (gitignored).

Usage:
  python data/download_librispeech.py            # dev-clean
  python data/download_librispeech.py --subset test-clean
"""
import argparse
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dev-clean",
                    choices=["dev-clean", "test-clean", "train-clean-100"])
    ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parent))
    args = ap.parse_args()

    import torchaudio

    root = pathlib.Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LibriSpeech {args.subset} into {root} ...")
    torchaudio.datasets.LIBRISPEECH(root=str(root), url=args.subset, download=True)

    subset_dir = root / "LibriSpeech" / args.subset
    flacs = list(subset_dir.rglob("*.flac"))
    speakers = sorted({p.name.split("-")[0] for p in flacs})
    print(f"Done. {len(flacs)} utterances from {len(speakers)} speakers at {subset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
