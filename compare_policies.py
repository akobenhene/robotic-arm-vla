"""Generate ACT vs SmolVLA side-by-side comparison GIF (same seed).

Also refreshes ``demo_output.gif`` from the ACT rollout (success path).

Usage
-----
    .\\.venv\\Scripts\\python.exe compare_policies.py --seed 36 --steps 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from env_wrapper import RoboticsEnvWrapper
from main import RolloutResult, save_frames
from policy import build_policy
from viz import overlay_hud, side_by_side


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the dual-policy comparison."""
    parser = argparse.ArgumentParser(description="ACT vs SmolVLA side-by-side GIF.")
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transfer the cube between the Aloha arms",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_act_vs_smolvla.gif",
    )
    parser.add_argument(
        "--act-gif",
        type=str,
        default="demo_output.gif",
        help="Also save the ACT episode here (hero demo).",
    )
    parser.add_argument(
        "--smolvla-gif",
        type=str,
        default="demo_smolvla.gif",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="comparison_report.json",
    )
    return parser.parse_args(argv)


def run_hud_episode(
    env: RoboticsEnvWrapper,
    policy,
    *,
    policy_name: str,
    prompt: str,
    seed: int,
    num_steps: int,
    stop_on_success: bool,
) -> RolloutResult:
    """Roll out one episode and burn HUD text onto every frame."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    frames: list[np.ndarray] = [
        overlay_hud(
            obs["rgb"],
            title=policy_name,
            seed=seed,
            step=0,
            reward=0.0,
            max_reward=0.0,
            success=False,
        )
    ]
    rewards: list[float] = []
    max_reward = 0.0
    success = False

    for step in tqdm(range(num_steps), desc=policy_name, unit="step"):
        action = policy.predict(obs, prompt, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        max_reward = max(max_reward, float(reward))
        success = bool(info.get("is_success", False)) or float(reward) >= 4.0
        frames.append(
            overlay_hud(
                obs["rgb"],
                title=policy_name,
                seed=seed,
                step=step + 1,
                reward=float(reward),
                max_reward=max_reward,
                success=success,
            )
        )
        if stop_on_success and success:
            break
        if terminated or truncated:
            break

    return RolloutResult(
        frames=frames,
        rewards=rewards,
        max_reward=max_reward,
        success=success,
        steps=len(rewards),
        seed=seed,
    )


def _pad_to_length(frames: list[np.ndarray], length: int) -> list[np.ndarray]:
    """Pad with the last frame so two rollouts share a common timeline."""
    if len(frames) >= length:
        return frames[:length]
    if not frames:
        raise ValueError("Empty frame list")
    return frames + [frames[-1]] * (length - len(frames))


def main(argv: Sequence[str] | None = None) -> int:
    """Run ACT + SmolVLA on one seed and write comparison artifacts."""
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable - using CPU.")
        args.device = "cpu"

    env = RoboticsEnvWrapper(
        env_id=RoboticsEnvWrapper.DEFAULT_ENV_ID,
        image_size=None,
        device=args.device,
        max_episode_steps=max(args.steps, 400),
    )
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

    results: dict[str, RolloutResult] = {}
    for kind, title, stop in (
        ("act", "ACT (task success)", True),
        ("smolvla", "SmolVLA (language)", False),
    ):
        print(f"\n=== Rolling out {kind} | seed={args.seed} ===")
        policy = build_policy(
            action_dim=env.action_dim,
            action_low=action_low,
            action_high=action_high,
            image_size=None,
            device=args.device,
            policy_type=kind,  # type: ignore[arg-type]
        )
        results[kind] = run_hud_episode(
            env,
            policy,
            policy_name=title,
            prompt=args.prompt,
            seed=args.seed,
            num_steps=args.steps,
            stop_on_success=stop,
        )
        del policy

    act = results["act"]
    smol = results["smolvla"]
    n = max(len(act.frames), len(smol.frames))
    left = _pad_to_length(act.frames, n)
    right = _pad_to_length(smol.frames, n)
    comparison = [side_by_side(l, r) for l, r in zip(left, right)]

    act_path = save_frames(act.frames, Path(args.act_gif), fps=args.fps)
    smol_path = save_frames(smol.frames, Path(args.smolvla_gif), fps=args.fps)
    cmp_path = save_frames(comparison, Path(args.output), fps=args.fps)

    report = {
        "seed": args.seed,
        "prompt": args.prompt,
        "act": {
            "success": act.success,
            "max_reward": act.max_reward,
            "steps": act.steps,
            "mean_reward": float(np.mean(act.rewards)) if act.rewards else 0.0,
            "gif": str(act_path),
        },
        "smolvla": {
            "success": smol.success,
            "max_reward": smol.max_reward,
            "steps": smol.steps,
            "mean_reward": float(np.mean(smol.rewards)) if smol.rewards else 0.0,
            "gif": str(smol_path),
        },
        "comparison_gif": str(cmp_path),
    }
    report_path = Path(args.report_json)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Comparison summary ===")
    print(json.dumps(report, indent=2))
    print(f"Wrote {cmp_path}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
