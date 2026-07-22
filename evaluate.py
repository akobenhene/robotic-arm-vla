"""Seed sweep / eval for LeRobot ACT on Aloha TransferCube.

Finds episodes where the cube transfer succeeds (reward == 4 / is_success),
writes ``eval_results.json``, and copies the best rollout to ``demo_output.gif``.

Usage
-----
    .\\.venv\\Scripts\\python.exe evaluate.py --seeds 0-19 --steps 400
    .\\.venv\\Scripts\\python.exe evaluate.py --seeds 0,3,7,12 --steps 400 --output-gif demo_output.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from env_wrapper import RoboticsEnvWrapper
from main import RolloutResult, run_episode, save_frames
from policy import build_policy


def parse_seed_list(spec: str) -> list[int]:
    """Parse ``0-19`` or ``0,3,7`` into a list of ints."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid seed range: {part}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    # Preserve order, drop duplicates
    seen: set[int] = set()
    ordered: list[int] = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            ordered.append(seed)
    return ordered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for multi-seed evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate LeRobot ACT across seeds; export best GIF."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-19",
        help="Seed list/ranges, e.g. '0-19' or '0,3,7,12'.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Torch device.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="act",
        choices=("act", "smolvla"),
        help="Policy backend to evaluate.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Optional Hugging Face repo override.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transfer the cube between the Aloha arms",
        help="Task description string (API compatibility).",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default="eval_results.json",
        help="Path for the JSON metrics report.",
    )
    parser.add_argument(
        "--output-gif",
        type=str,
        default="demo_output.gif",
        help="GIF path for the best (prefer success) episode.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="GIF frame rate.",
    )
    parser.add_argument(
        "--stop-after-success",
        action="store_true",
        default=True,
        help="Stop the seed sweep after the first successful episode.",
    )
    parser.add_argument(
        "--continue-after-success",
        action="store_true",
        help="Evaluate all seeds even after a success is found.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Sweep seeds, report success rate, write best GIF + JSON."""
    args = parse_args(argv)
    seeds = parse_seed_list(args.seeds)
    stop_sweep_on_success = not args.continue_after_success

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable - falling back to CPU.")
        args.device = "cpu"

    env = RoboticsEnvWrapper(
        env_id=RoboticsEnvWrapper.DEFAULT_ENV_ID,
        image_size=None,
        device=args.device,
        max_episode_steps=max(args.steps, 400),
    )
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)
    policy = build_policy(
        action_dim=env.action_dim,
        action_low=action_low,
        action_high=action_high,
        image_size=None,
        device=args.device,
        policy_type=args.policy,
        lerobot_repo_id=args.repo_id,
    )

    results: list[dict[str, float | int | bool]] = []
    best: RolloutResult | None = None

    try:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            episode = run_episode(
                env,
                policy,  # type: ignore[arg-type]
                prompt=args.prompt,
                num_steps=args.steps,
                seed=seed,
                deterministic=True,
                stop_on_success=True,
                show_progress=True,
            )
            row = {
                "seed": episode.seed,
                "steps": episode.steps,
                "max_reward": episode.max_reward,
                "mean_reward": float(np.mean(episode.rewards)) if episode.rewards else 0.0,
                "success": episode.success,
            }
            results.append(row)
            print(
                f"seed={seed:>3} | steps={episode.steps:>3} | "
                f"max_reward={episode.max_reward:.0f} | success={episode.success}"
            )

            if best is None:
                best = episode
            else:
                # Prefer success; else higher max_reward; else shorter success path.
                if episode.success and not best.success:
                    best = episode
                elif episode.success == best.success and episode.max_reward > best.max_reward:
                    best = episode
                elif (
                    episode.success
                    and best.success
                    and episode.max_reward == best.max_reward
                    and episode.steps < best.steps
                ):
                    best = episode

            if episode.success and stop_sweep_on_success:
                print(f"First success at seed={seed}; stopping sweep.")
                break
    finally:
        env.close()

    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    success_rate = (100.0 * n_success / n) if n else 0.0
    resolved_repo = args.repo_id or getattr(policy, "repo_id", args.policy)
    report = {
        "policy_type": args.policy,
        "policy": resolved_repo,
        "env_id": RoboticsEnvWrapper.DEFAULT_ENV_ID,
        "steps_limit": args.steps,
        "seeds_evaluated": [r["seed"] for r in results],
        "n_episodes": n,
        "n_success": n_success,
        "success_rate_pct": round(success_rate, 2),
        "episodes": results,
        "best_seed": None if best is None else best.seed,
        "best_success": None if best is None else best.success,
        "best_max_reward": None if best is None else best.max_reward,
    }

    results_path = Path(args.results_json).resolve()
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {results_path}")
    print(
        f"Success rate: {n_success}/{n} = {success_rate:.1f}% | "
        f"best_seed={report['best_seed']} | best_max_reward={report['best_max_reward']}"
    )

    if best is not None:
        gif_path = save_frames(best.frames, Path(args.output_gif), fps=args.fps)
        print(
            f"Best episode GIF -> {gif_path} "
            f"(seed={best.seed}, success={best.success}, max_reward={best.max_reward})"
        )

    return 0 if n_success > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
