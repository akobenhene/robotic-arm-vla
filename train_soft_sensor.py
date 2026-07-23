"""Collect ACT frames labeled by current reward; train CPU soft-sensor head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from env_wrapper import RoboticsEnvWrapper
from evaluate import parse_seed_list
from policy import build_policy
from soft_sensor import (
    RewardStageSoftSensor,
    preprocess_frame,
    save_soft_sensor,
)


def collect_dataset(
    seeds: list[int],
    steps: int,
    device: str,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    env = RoboticsEnvWrapper(device=device, max_episode_steps=max(steps, 400))
    policy = build_policy(
        action_dim=env.action_dim,
        action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
        action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
        image_size=None,
        device=device,
        policy_type="act",
    )
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    try:
        for seed in seeds:
            obs, _info = env.reset(seed=seed)
            if hasattr(policy, "reset"):
                policy.reset()
            for t in range(steps):
                action = policy.predict(obs, text="Transfer the cube between the Aloha arms")
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                obs, reward, terminated, truncated, _info = env.step(action)
                if t % stride == 0:
                    frame = obs["rgb"]
                    xs.append(preprocess_frame(frame))
                    ys.append(int(max(0, min(4, round(float(reward))))))
                if terminated or truncated:
                    break
    finally:
        env.close()
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train reward-stage soft sensor on CPU.")
    parser.add_argument("--seeds", type=str, default="0-4")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="outputs/soft_sensor.pt")
    parser.add_argument("--metrics-json", type=str, default="outputs/soft_sensor_metrics.json")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    print(f"Collecting frames from ACT seeds={seeds} steps={args.steps} ...")
    x, y = collect_dataset(seeds, args.steps, args.device, args.stride)
    print(f"Dataset: {len(x)} frames | stage counts={torch.bincount(y, minlength=5).tolist()}")

    n = len(x)
    n_val = max(1, n // 5)
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_loader = DataLoader(
        TensorDataset(x[train_idx], y[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = RewardStageSoftSensor().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        print(f"epoch {epoch+1}/{args.epochs} train_loss={total / max(len(train_idx), 1):.4f}")

    model.eval()
    with torch.inference_mode():
        pred = model(x[val_idx].to(args.device)).argmax(dim=-1).cpu()
        acc = float((pred == y[val_idx]).float().mean().item())
    metrics = {
        "n_frames": n,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_accuracy": round(acc, 4),
        "seeds": seeds,
        "steps": args.steps,
    }
    out = save_soft_sensor(model.cpu(), Path(args.out))
    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {out} | val_accuracy={acc:.3f} | {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
