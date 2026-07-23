"""CPU-friendly observation domain randomization for robustness probes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DomainRandomizationConfig:
    """Light RGB perturbations. Keep mild when probing pretrained ACT."""

    enabled: bool = False
    brightness_jitter: float = 0.15
    noise_std: float = 8.0
    seed: int | None = None


class DomainRandomizer:
    """Applies optional brightness and Gaussian noise to HWC uint8 frames."""

    def __init__(self, config: DomainRandomizationConfig | None = None) -> None:
        self.config = config or DomainRandomizationConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def maybe_perturb_rgb(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return frame
        out = frame.astype(np.float32)
        if self.config.brightness_jitter > 0:
            delta = float(self._rng.uniform(-self.config.brightness_jitter, self.config.brightness_jitter))
            out = out * (1.0 + delta)
        if self.config.noise_std > 0:
            out = out + self._rng.normal(0.0, self.config.noise_std, size=out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
