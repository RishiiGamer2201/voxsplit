"""Zero-shot voice cloning TTS for VoxSplit (Phase 6 addon).

Uses XTTS v2 (coqui-tts community fork) to synthesize any text in the voice
of a reference speaker, given a short WAV clip of that speaker. In VoxSplit,
the separated per-speaker tracks serve as the reference audio, so no extra
recording is needed.

Usage:
    from src.tts.voice_clone import VoiceCloner
    cloner = VoiceCloner()                        # loads model once
    wav, sr = cloner.clone("Hello world!", "speaker1.wav")

Self-test (no real audio needed):
    python src/tts/voice_clone.py --self-test
"""
import argparse
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Model singleton — load once, reuse across calls
# ---------------------------------------------------------------------------

_CLONER_INSTANCE = None


class VoiceCloner:
    """Wraps XTTS v2 for zero-shot voice cloning.

    The model is heavy (~1.8 GB) and loads in ~30 s on CPU, so we load it
    once and keep it alive for the whole Gradio session.
    """

    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
        "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
    ]

    def __init__(self, device: str = "auto") -> None:
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._tts = None  # lazy load

    def _load(self) -> None:
        """Load XTTS v2 model on first use.

        Distinguishes "not installed" from "installed but its import blew up",
        because the second is the common case: coqui-tts needs transformers < 5,
        and forcing that downgrade drags huggingface-hub below what gradio
        requires. Reporting "not installed" for that would send you in circles.
        """
        if self._tts is not None:
            return
        import importlib.util
        if importlib.util.find_spec("TTS") is None:
            raise ImportError(
                "coqui-tts is not installed (voice cloning is opt-in).\n"
                "    pip install coqui-tts\n"
                "See the warning in requirements.txt first: it pins "
                "transformers/torch and can break this env.")
        try:
            from TTS.api import TTS
        except Exception as exc:
            raise ImportError(
                f"coqui-tts IS installed but failed to import: {exc}\n"
                f"Most likely a dependency clash — coqui-tts wants "
                f"transformers < 5, while gradio needs huggingface-hub >= 1.2 "
                f"(which pulls transformers 5). Voice cloning is optional; "
                f"install it in a SEPARATE env rather than downgrading this "
                f"one, and re-check `python src/check_env.py`.") from exc
        print("Loading XTTS v2 model (first use, ~30 s on CPU) ...")
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            self.device
        )
        print("XTTS v2 ready.")

    def clone(
        self,
        text: str,
        reference_wav: str,
        language: str = "en",
    ) -> Tuple[np.ndarray, int]:
        """Synthesise `text` in the voice captured in `reference_wav`.

        Args:
            text: The text to speak.
            reference_wav: Path to a WAV file containing the target voice.
                           Should be at least 3 s long; 6-10 s is ideal.
            language: BCP-47 language code, e.g. "en", "es", "fr".

        Returns:
            (samples, sample_rate) as (np.ndarray float32, int).
        """
        import soundfile as sf

        self._load()

        if language not in self.SUPPORTED_LANGUAGES:
            print(
                f"Warning: language '{language}' not in XTTS supported list; "
                f"falling back to 'en'."
            )
            language = "en"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        self._tts.tts_to_file(
            text=text,
            speaker_wav=reference_wav,
            language=language,
            file_path=out_path,
        )

        wav, sr = sf.read(out_path, dtype="float32")
        Path(out_path).unlink(missing_ok=True)
        return wav, sr

    def clone_to_numpy(
        self,
        text: str,
        reference_wav: str,
        language: str = "en",
    ) -> Tuple[np.ndarray, int]:
        """Convenience alias for clone() — same signature."""
        return self.clone(text, reference_wav, language)


def get_cloner(device: str = "auto") -> VoiceCloner:
    """Return the global VoiceCloner singleton (creates it on first call)."""
    global _CLONER_INSTANCE
    if _CLONER_INSTANCE is None:
        _CLONER_INSTANCE = VoiceCloner(device=device)
    return _CLONER_INSTANCE


# ---------------------------------------------------------------------------
# Self-test / CLI
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Verify the VoiceCloner class can be imported and instantiated.

    Does NOT load the heavy XTTS model (requires network + ~30 s).  It just
    checks that the import chain is intact and the object can be built.
    """
    cloner = VoiceCloner(device="cpu")
    assert cloner.device == "cpu"
    assert "en" in cloner.SUPPORTED_LANGUAGES
    print("VoiceCloner self-test passed (model NOT loaded - skipped for speed).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Voice clone a text string using a reference WAV."
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run sanity checks and exit."
    )
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--reference-wav", type=str, default=None)
    parser.add_argument(
        "--language", type=str, default="en", help="Language code (default: en)"
    )
    parser.add_argument("--out", type=str, default="cloned_output.wav")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if not args.text or not args.reference_wav:
        parser.error("--text and --reference-wav are required (or use --self-test)")

    cloner = VoiceCloner(device=args.device)
    wav, sr = cloner.clone(args.text, args.reference_wav, args.language)

    import soundfile as sf
    sf.write(args.out, wav, sr)
    print(f"Saved cloned audio to: {args.out}  (sr={sr}, samples={len(wav)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
