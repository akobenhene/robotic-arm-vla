"""Lightweight CI / local smoke tests for the VLA manipulation stack.

Runs:
1. Import sanity checks
2. Short ACT rollout (downloads Hub weights on first run)
3. SmolVLA prompt ablation (language sensitivity)

Usage
-----
    .\\.venv\\Scripts\\python.exe smoke_test.py
    .\\.venv\\Scripts\\python.exe smoke_test.py --skip-smolvla   # faster CI option
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI smoke tests for robotic-arm-vla.")
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--act-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-smolvla",
        action="store_true",
        help="Skip SmolVLA prompt ablation (faster pipeline-only smoke).",
    )
    parser.add_argument(
        "--skip-act",
        action="store_true",
        help="Skip ACT rollout (imports + optional SmolVLA only).",
    )
    return parser.parse_args()


def check_imports() -> None:
    """Fail fast if core packages are missing."""
    import gymnasium  # noqa: F401
    import gym_aloha  # noqa: F401
    import imageio  # noqa: F401
    import numpy  # noqa: F401
    import torch  # noqa: F401

    from env_wrapper import RoboticsEnvWrapper  # noqa: F401
    from policy import LeRobotACTPolicy, LeRobotSmolVLAPolicy, build_policy  # noqa: F401

    print("[ok] imports")


def run_act_smoke(device: str, steps: int, seed: int) -> None:
    """Run a short ACT episode and require a written GIF artifact."""
    from main import main as run_main

    out = Path("smoke_act.gif")
    code = run_main(
        [
            "--policy",
            "act",
            "--steps",
            str(steps),
            "--seed",
            str(seed),
            "--device",
            device,
            "--output",
            str(out),
            "--no-stop-on-success",
        ]
    )
    if code != 0:
        raise RuntimeError(f"ACT smoke main() returned {code}")
    if not out.exists() or out.stat().st_size < 1000:
        raise RuntimeError(f"ACT smoke GIF missing or too small: {out}")
    print(f"[ok] ACT smoke -> {out} ({out.stat().st_size} bytes)")


def run_smolvla_ablation(device: str, seed: int) -> None:
    """Require prompt ablation to report language_sensitive=true."""
    from prompt_ablation import main as ablation_main

    out = Path("smoke_prompt_ablation.json")
    code = ablation_main(
        [
            "--seed",
            str(seed),
            "--device",
            device,
            "--output",
            str(out),
        ]
    )
    if code != 0:
        raise RuntimeError(f"prompt_ablation returned {code}")
    report = json.loads(out.read_text(encoding="utf-8"))
    if not report.get("language_sensitive"):
        raise RuntimeError(f"Expected language_sensitive=true, got {report}")
    print(
        f"[ok] SmolVLA ablation l1_delta={report['l1_delta']:.4f} "
        f"language_sensitive={report['language_sensitive']}"
    )


def main() -> int:
    """Entrypoint for CI smoke tests."""
    args = _parse_args()
    print("=== smoke_test: imports ===")
    check_imports()

    if not args.skip_act:
        print("=== smoke_test: ACT short rollout ===")
        run_act_smoke(args.device, args.act_steps, args.seed)

    if not args.skip_smolvla:
        print("=== smoke_test: SmolVLA prompt ablation ===")
        run_smolvla_ablation(args.device, args.seed)

    print("=== smoke_test: PASS ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - surface clear CI failure
        print(f"=== smoke_test: FAIL ===\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
