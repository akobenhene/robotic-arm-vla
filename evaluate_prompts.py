"""Multi-prompt evaluation: same seed, many instructions (SmolVLA / ACT)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env_wrapper import RoboticsEnvWrapper
from main import run_episode
from policy import build_policy
from prompt_bank import list_prompts


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a prompt bank on one seed.")
    parser.add_argument("--policy", choices=("act", "smolvla"), default="smolvla")
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="prompt_bank_results.json")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    env = RoboticsEnvWrapper(device=args.device, max_episode_steps=max(args.steps, 400))
    policy = build_policy(
        action_dim=env.action_dim,
        action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
        action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
        image_size=None,
        device=args.device,
        policy_type=args.policy,
    )

    rows = []
    try:
        for name, prompt in list_prompts():
            if hasattr(policy, "reset"):
                policy.reset()
            ep = run_episode(
                env,
                policy,
                prompt=prompt,
                num_steps=args.steps,
                seed=args.seed,
                deterministic=True,
                stop_on_success=True,
                show_progress=False,
            )
            row = {
                "prompt_id": name,
                "prompt": prompt,
                "seed": args.seed,
                "steps": ep.steps,
                "max_reward": ep.max_reward,
                "success": ep.success,
            }
            rows.append(row)
            print(f"{name:12} max_r={ep.max_reward:.0f} success={ep.success} steps={ep.steps}")
    finally:
        env.close()

    report = {
        "policy": args.policy,
        "seed": args.seed,
        "steps_limit": args.steps,
        "device": args.device,
        "prompts": rows,
        "note": "Short horizon on CPU; use --steps 400 on GPU/overnight for task success.",
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
