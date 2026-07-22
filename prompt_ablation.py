"""Prove language conditioning: same observation, different prompts → different actions.

Writes ``prompt_ablation.json`` with L1 / L2 action deltas for SmolVLA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from env_wrapper import RoboticsEnvWrapper
from policy import LeRobotSmolVLAPolicy, build_policy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the prompt ablation."""
    parser = argparse.ArgumentParser(description="SmolVLA prompt ablation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--prompt-a",
        type=str,
        default="Transfer the cube between the Aloha arms",
    )
    parser.add_argument(
        "--prompt-b",
        type=str,
        default="Do nothing and keep both arms still",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prompt_ablation.json",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=LeRobotSmolVLAPolicy.DEFAULT_REPO_ID,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Compare first-step actions under two language instructions."""
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    env = RoboticsEnvWrapper(
        env_id=RoboticsEnvWrapper.DEFAULT_ENV_ID,
        image_size=None,
        device=args.device,
        max_episode_steps=50,
    )
    policy = build_policy(
        action_dim=env.action_dim,
        action_low=torch.as_tensor(env.action_space.low),
        action_high=torch.as_tensor(env.action_space.high),
        device=args.device,
        policy_type="smolvla",
        lerobot_repo_id=args.repo_id,
    )

    obs, _ = env.reset(seed=args.seed)
    policy.reset()
    action_a = policy.predict(obs, args.prompt_a)
    policy.reset()
    action_b = policy.predict(obs, args.prompt_b)
    env.close()

    delta = action_a - action_b
    report = {
        "policy_type": "smolvla",
        "repo_id": args.repo_id,
        "seed": args.seed,
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "action_a": action_a.detach().cpu().tolist(),
        "action_b": action_b.detach().cpu().tolist(),
        "l1_delta": float(delta.abs().mean().item()),
        "l2_delta": float(torch.linalg.vector_norm(delta).item()),
        "language_sensitive": bool(delta.abs().mean().item() > 1e-4),
    }
    out = Path(args.output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out.resolve()}")
    return 0 if report["language_sensitive"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
