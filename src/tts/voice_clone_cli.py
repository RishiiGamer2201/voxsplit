"""Voice-cloning worker — runs INSIDE the isolated `voxsplit-tts` conda env.

coqui-tts needs transformers < 5, which drags huggingface-hub below what Gradio
requires, so it cannot live in the main `voxsplit` env. It therefore runs in its
own env and is invoked as a subprocess, exactly like separate_mossformer2.py
does for the `clearvoice` env. voice_clone.py calls this and reads the wav back.

Not meant to be run by hand; use VoiceCloner (it shells out here). Direct use:
  conda run -n voxsplit-tts python src/tts/voice_clone_cli.py \
      --text "hello" --ref speaker1.wav --out out.wav --language en
"""
import argparse
import os
from pathlib import Path

# XTTS v2 prompts for its licence on first download, which EOFErrors in a
# subprocess. Setting this accepts the Coqui Public Model License (CPML):
# the XTTS v2 WEIGHTS are NON-COMMERCIAL. Our own code stays under the repo's
# licence; only the downloaded voice-cloning weights carry this restriction.
os.environ.setdefault("COQUI_TOS_AGREED", "1")


def main() -> int:
    parser = argparse.ArgumentParser(description="XTTS v2 cloning worker.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref", required=True, type=Path,
                        help="Reference wav (a separated speaker track).")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu",
                        help="cpu is plenty for a demo sentence; the TTS env "
                             "does not need the CUDA build.")
    args = parser.parse_args()

    if not args.ref.is_file():
        print(f"Reference wav not found: {args.ref}")
        return 1

    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(text=args.text, speaker_wav=str(args.ref),
                    language=args.language, file_path=str(args.out))
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
