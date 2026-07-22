"""End-to-end LeRobot ACT manipulation demo with success-aware GIF export.

Usage
-----
    .\\.venv\\Scripts\\python.exe main.py --steps 400 --seed 0
    .\\.venv\\Scripts\\python.exe evaluate.py --seeds 0-19 --steps 400
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from env_wrapper import RoboticsEnvWrapper
from policy import VLAPolicyProtocol, build_policy


@dataclass
class RolloutResult:
    """Metrics and frames from a single closed-loop episode."""

    frames: list[np.ndarray]
    rewards: list[float]
    max_reward: float
    success: bool
    steps: int
    seed: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the manipulation demo."""
    parser = argparse.ArgumentParser(
        description="LeRobot ACT manipulation demo (MuJoCo Aloha + PyTorch)."
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=RoboticsEnvWrapper.DEFAULT_ENV_ID,
        help="Gymnasium env id (default: gym_aloha/AlohaTransferCube-v0).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="Max control steps per episode (Aloha demos typically use 400).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for env reset and torch.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transfer the cube between the Aloha arms",
        help="Task description (logged; classic ACT is vision+state, not text).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Optional RGB resize. Omit for native ACT resolution (480x640).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo_output.gif",
        help="Output GIF / MP4 path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for the saved animation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Torch device for policy inference.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="act",
        choices=("act", "smolvla", "mock"),
        help="Policy backend: act (success), smolvla (language), mock.",
    )
    parser.add_argument(
        "--mock-policy",
        action="store_true",
        help="Alias for --policy mock.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Optional Hugging Face repo override for act/smolvla.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable exploration noise (MockVLAPolicy only).",
    )
    parser.add_argument(
        "--no-stop-on-success",
        action="store_true",
        help="Continue rolling after reward==4 (default: stop and keep success frames).",
    )
    return parser.parse_args(argv)


def save_frames(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int = 20,
) -> Path:
    """Write captured RGB frames to GIF or MP4 via imageio."""
    if not frames:
        raise ValueError("No frames captured - nothing to save.")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    write_frames: list[np.ndarray] = frames
    if suffix == ".gif" and frames[0].shape[0] > 240:
        import torch.nn.functional as F

        resized: list[np.ndarray] = []
        for frame in frames:
            t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            scale = 240.0 / float(frame.shape[0])
            new_w = max(1, int(round(frame.shape[1] * scale)))
            t = F.interpolate(t, size=(240, new_w), mode="bilinear", align_corners=False)
            out = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0).byte().numpy()
            resized.append(out)
        write_frames = resized

    if suffix == ".gif":
        imageio.mimsave(
            output_path,
            write_frames,
            duration=1.0 / float(fps),
            loop=0,
        )
    elif suffix in {".mp4", ".mov", ".avi"}:
        imageio.mimsave(output_path, write_frames, fps=fps)
    else:
        imageio.mimsave(output_path, write_frames, duration=1.0 / float(fps), loop=0)

    return output_path


def run_episode(
    env: RoboticsEnvWrapper,
    policy: VLAPolicyProtocol,
    prompt: str,
    num_steps: int,
    *,
    seed: int,
    deterministic: bool = True,
    stop_on_success: bool = True,
    show_progress: bool = True,
) -> RolloutResult:
    """Run one episode; optionally stop when TransferCube success (reward == 4)."""
    obs, _info = env.reset(seed=seed)
    policy.reset()
    frames: list[np.ndarray] = [obs["rgb"].copy()]
    rewards: list[float] = []
    success = False
    max_reward = 0.0

    iterator = range(num_steps)
    if show_progress:
        iterator = tqdm(iterator, desc=f"seed={seed}", unit="step")

    for _ in iterator:
        action = policy.predict(obs, prompt, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs["rgb"].copy())
        rewards.append(float(reward))
        max_reward = max(max_reward, float(reward))
        success = bool(info.get("is_success", False)) or float(reward) >= 4.0

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


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint: build env + LeRobot ACT, run one episode, save GIF."""
    args = parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable - falling back to CPU.")
        args.device = "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy_type = "mock" if args.mock_policy else args.policy
    use_lerobot = policy_type != "mock"
    image_size = (
        (int(args.image_size[0]), int(args.image_size[1]))
        if args.image_size is not None
        else (None if use_lerobot else (84, 84))
    )

    print(f"Initializing environment: {args.env_id}")
    env = RoboticsEnvWrapper(
        env_id=args.env_id,
        image_size=image_size,
        device=args.device,
        max_episode_steps=max(args.steps, 400),
    )
    print(env)

    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

    print(
        f"Building policy ({policy_type}) | "
        f"action_dim={env.action_dim} | prompt={args.prompt!r}"
    )
    if use_lerobot and not env.is_aloha:
        print(
            "WARNING: pretrained Hub policies expect AlohaTransferCube. "
            "Use the default env id for task success."
        )

    policy = build_policy(
        action_dim=env.action_dim,
        action_low=action_low,
        action_high=action_high,
        image_size=image_size if isinstance(image_size, tuple) else (84, 84),
        device=args.device,
        policy_type=policy_type,
        lerobot_repo_id=args.repo_id,
    )

    try:
        result = run_episode(
            env,
            policy,  # type: ignore[arg-type]
            prompt=args.prompt,
            num_steps=args.steps,
            seed=args.seed,
            deterministic=args.deterministic or use_lerobot,
            stop_on_success=not args.no_stop_on_success,
        )
    finally:
        env.close()

    output_path = save_frames(result.frames, Path(args.output), fps=args.fps)
    mean_r = float(np.mean(result.rewards)) if result.rewards else 0.0
    print(
        f"Saved {len(result.frames)} frames -> {output_path} | "
        f"steps={result.steps} | mean_reward={mean_r:.4f} | "
        f"max_reward={result.max_reward:.1f} | success={result.success}"
    )
    print(f"Policy backend: {policy_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
