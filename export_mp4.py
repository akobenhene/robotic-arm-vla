"""Export demo videos as MP4 (portfolio / LinkedIn friendly).

Reads an existing GIF or re-runs ACT seed 36 and writes ``demo_output.mp4``.

Usage
-----
    .\\.venv\\Scripts\\python.exe export_mp4.py --from-gif demo_output.gif
    .\\.venv\\Scripts\\python.exe export_mp4.py --rerun-act --seed 36
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for MP4 export."""
    parser = argparse.ArgumentParser(description="Export MP4 demo videos.")
    parser.add_argument(
        "--from-gif",
        type=str,
        default=None,
        help="Convert an existing GIF to MP4.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo_output.mp4",
        help="Destination MP4 path.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--rerun-act",
        action="store_true",
        help="Re-run ACT and write MP4 directly (ignores --from-gif).",
    )
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--also",
        type=str,
        nargs="*",
        default=[],
        help="Extra GIFs to convert (e.g. comparison_act_vs_smolvla.gif).",
    )
    return parser.parse_args(argv)


def gif_to_mp4(gif_path: Path, mp4_path: Path, fps: int) -> Path:
    """Decode GIF frames and write an H.264-friendly MP4 via imageio-ffmpeg."""
    frames = list(imageio.mimread(gif_path))
    if not frames:
        raise ValueError(f"No frames in {gif_path}")
    # Ensure uint8 RGB and H.264 macroblock-friendly size (multiples of 16).
    rgb_frames = [_pad_macroblock(np.asarray(f[..., :3], dtype=np.uint8)) for f in frames]
    mp4_path = mp4_path.resolve()
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(mp4_path, rgb_frames, fps=fps, codec="libx264", quality=8)
    return mp4_path


def _pad_macroblock(frame: np.ndarray, block: int = 16) -> np.ndarray:
    """Pad HWC frame so H and W are divisible by ``block``."""
    h, w = frame.shape[:2]
    pad_h = (block - h % block) % block
    pad_w = (block - w % block) % block
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def rerun_act_mp4(output: Path, *, seed: int, steps: int, device: str, fps: int) -> Path:
    """Roll out ACT once and save MP4 (and a matching GIF for README)."""
    from main import main as run_main

    gif_out = output.with_suffix(".gif")
    code = run_main(
        [
            "--policy",
            "act",
            "--seed",
            str(seed),
            "--steps",
            str(steps),
            "--device",
            device,
            "--output",
            str(gif_out),
            "--fps",
            str(fps),
        ]
    )
    if code != 0:
        raise RuntimeError(f"ACT rollout failed with code {code}")
    return gif_to_mp4(gif_out, output, fps)


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for MP4 export."""
    args = parse_args(argv)
    written: list[Path] = []

    if args.rerun_act:
        path = rerun_act_mp4(
            Path(args.output),
            seed=args.seed,
            steps=args.steps,
            device=args.device,
            fps=args.fps,
        )
        written.append(path)
    else:
        sources = []
        if args.from_gif:
            sources.append((Path(args.from_gif), Path(args.output)))
        else:
            # Default portfolio set
            defaults = [
                ("demo_output.gif", "demo_output.mp4"),
                ("comparison_act_vs_smolvla.gif", "comparison_act_vs_smolvla.mp4"),
                ("demo_smolvla.gif", "demo_smolvla.mp4"),
            ]
            for src, dst in defaults:
                if Path(src).exists():
                    sources.append((Path(src), Path(dst)))
        for extra in args.also:
            src = Path(extra)
            sources.append((src, src.with_suffix(".mp4")))

        if not sources:
            raise FileNotFoundError(
                "No GIFs found. Pass --from-gif PATH or --rerun-act."
            )
        for src, dst in sources:
            written.append(gif_to_mp4(src, dst, args.fps))

    for path in written:
        print(f"Wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
