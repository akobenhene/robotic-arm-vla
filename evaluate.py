"""Seed sweep / eval for LeRobot policies on Aloha TransferCube.

Writes ``eval_results.json`` with Wilson CI, timing, and best GIF.

Usage
-----
    .\\.venv\\Scripts\\python.exe evaluate.py --seeds 0-19 --steps 400
    .\\.venv\\Scripts\\python.exe evaluate.py --seeds 0-9 --continue-after-success
    .\\.venv\\Scripts\\python.exe evaluate.py --seeds 0-4 --domain-rand
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from domain_randomization import DomainRandomizationConfig, DomainRandomizer
from env_wrapper import RoboticsEnvWrapper
from main import RolloutResult, run_episode, save_frames
from metrics_eval import summarize_episodes
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
    seen: set[int] = set()
    ordered: list[int] = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            ordered.append(seed)
    return ordered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LeRobot policies across seeds; export best GIF."
    )
    parser.add_argument("--seeds", type=str, default="0-19")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--policy", type=str, default="act", choices=("act", "smolvla"))
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transfer the cube between the Aloha arms",
    )
    parser.add_argument("--results-json", type=str, default="eval_results.json")
    parser.add_argument("--output-gif", type=str, default="demo_output.gif")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stop-after-success", action="store_true", default=True)
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument(
        "--domain-rand",
        action="store_true",
        help="Mild RGB domain randomization (robustness probe).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
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
    randomizer = DomainRandomizer(
        DomainRandomizationConfig(enabled=args.domain_rand, seed=0)
    )
    if args.domain_rand:
        _orig_format = env._format_observation

        def _format_with_rand(raw_obs, rgb):
            return _orig_format(raw_obs, randomizer.maybe_perturb_rgb(rgb))

        env._format_observation = _format_with_rand  # type: ignore[method-assign]

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

    results: list[dict] = []
    best: RolloutResult | None = None

    try:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.perf_counter()
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
            elapsed = time.perf_counter() - t0
            row = {
                "seed": episode.seed,
                "steps": episode.steps,
                "max_reward": episode.max_reward,
                "mean_reward": float(np.mean(episode.rewards)) if episode.rewards else 0.0,
                "success": episode.success,
                "seconds": round(elapsed, 3),
            }
            results.append(row)
            print(
                f"seed={seed:>3} | steps={episode.steps:>3} | "
                f"max_reward={episode.max_reward:.0f} | success={episode.success} | "
                f"{elapsed:.1f}s"
            )

            if best is None:
                best = episode
            elif episode.success and not best.success:
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

    summary = summarize_episodes(results)
    resolved_repo = args.repo_id or getattr(policy, "repo_id", args.policy)
    report = {
        "policy_type": args.policy,
        "policy": resolved_repo,
        "env_id": RoboticsEnvWrapper.DEFAULT_ENV_ID,
        "steps_limit": args.steps,
        "domain_randomization": bool(args.domain_rand),
        "seeds_evaluated": [r["seed"] for r in results],
        **summary,
        "episodes": results,
        "best_seed": None if best is None else best.seed,
        "best_success": None if best is None else best.success,
        "best_max_reward": None if best is None else best.max_reward,
    }

    results_path = Path(args.results_json).resolve()
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {results_path}")
    print(
        f"Success rate: {summary['n_success']}/{summary['n_episodes']} = "
        f"{summary['success_rate_pct']:.1f}% | "
        f"Wilson95%={summary['wilson_ci95_pct']} | "
        f"mean_steps_to_success={summary['mean_steps_to_success']}"
    )

    if best is not None:
        gif_path = save_frames(best.frames, Path(args.output_gif), fps=args.fps)
        print(
            f"Best episode GIF -> {gif_path} "
            f"(seed={best.seed}, success={best.success}, max_reward={best.max_reward})"
        )

    return 0 if summary["n_success"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
