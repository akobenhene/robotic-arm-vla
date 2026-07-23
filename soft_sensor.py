"""Soft-sensor head: infer TransferCube reward stage from RGB (CPU-friendly)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_STAGES = 5  # reward levels 0..4


class RewardStageSoftSensor(nn.Module):
    """Tiny CNN mapping NCHW RGB -> 5-way reward-stage logits."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Linear(64 * 4 * 4, NUM_STAGES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h.flatten(1))

    @torch.inference_mode()
    def predict_stage(self, rgb_nchw: torch.Tensor) -> tuple[int, float]:
        """Return (argmax stage, success probability = P(stage==4))."""
        if rgb_nchw.ndim == 3:
            rgb_nchw = rgb_nchw.unsqueeze(0)
        logits = self.forward(rgb_nchw)
        probs = F.softmax(logits, dim=-1)[0]
        stage = int(torch.argmax(probs).item())
        success_p = float(probs[4].item())
        return stage, success_p


def preprocess_frame(frame_hwc: np.ndarray, size: tuple[int, int] = (120, 160)) -> torch.Tensor:
    """HWC uint8 -> float NCHW in [0, 1], optionally resized."""
    import torch.nn.functional as tF

    t = torch.from_numpy(np.ascontiguousarray(frame_hwc)).permute(2, 0, 1).float() / 255.0
    t = t.unsqueeze(0)
    t = tF.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t.squeeze(0)


def save_soft_sensor(model: RewardStageSoftSensor, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)
    return path


def load_soft_sensor(path: Path, device: str = "cpu") -> RewardStageSoftSensor:
    model = RewardStageSoftSensor()
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
