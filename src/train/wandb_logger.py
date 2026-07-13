"""Optional Weights & Biases logging for the VoxSplit trainers.

Off by default. The committed CSV logs stay the source of truth; W&B is an
extra view you can turn on with --wandb. If wandb is not installed or init
fails, training continues and only the CSV is written. Default mode is
"offline" so it works with no account or API key (runs land in ./wandb/).

Run this file directly for a self-test (disabled logger is a no-op):
  python src/train/wandb_logger.py
"""
from typing import Dict, Optional


class WandbLogger:
    """Thin wrapper that no-ops unless explicitly enabled and importable."""

    def __init__(self, enabled: bool = False, project: str = "voxsplit",
                 mode: str = "offline", config: Optional[Dict] = None,
                 name: Optional[str] = None) -> None:
        self.run = None
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb
            self._wandb = wandb
            self.run = wandb.init(project=project, mode=mode,
                                  config=config or {}, name=name)
            print(f"wandb logging on (project={project}, mode={mode}).")
        except Exception as exc:  # pragma: no cover - env dependent
            self.run = None
            print(f"wandb disabled ({exc}); CSV logging continues.")

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self.run is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self._wandb.finish()


def _self_test() -> int:
    log = WandbLogger(enabled=False)
    assert log.run is None
    log.log({"loss": 1.0}, step=1)  # must not raise when disabled
    log.finish()
    print("WandbLogger disabled-path self-test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
