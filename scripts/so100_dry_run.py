"""Hardware-free SO-100 deployment dry-run / checklist.

Validates that the LeRobot stack imports and prints the record→train→deploy plan
without requiring a physical arm.
"""

from __future__ import annotations

import json
from pathlib import Path


CHECKLIST = [
    ("robot", "SO-100/SO-101 powered, motor IDs set, e-stop tested"),
    ("camera", "OpenCV/RealSense index known; 640x480 @ 30fps reachable"),
    ("host", "Ubuntu preferred; USB serial permissions for /dev/ttyACM*"),
    ("dataset", "Plan HF repo id: <user>/so100_cube_demo"),
    ("policy", "Do NOT load Aloha ACT zero-shot; finetune on SO-100 demos"),
    ("safety", "Workspace box + velocity/torque clamps before policy control"),
]


def main() -> int:
    print("SO-100 dry-run (no hardware required)\n")
    import_ok = True
    try:
        import lerobot  # noqa: F401

        print("[ok] lerobot import")
    except Exception as exc:  # pragma: no cover
        import_ok = False
        print(f"[warn] lerobot import failed: {exc}")

    rows = []
    for key, text in CHECKLIST:
        print(f"[ ] {key:8} {text}")
        rows.append({"item": key, "detail": text, "done": False})

    plan = {
        "import_ok": import_ok,
        "checklist": rows,
        "commands": {
            "record": "lerobot-record --robot.type=so100_follower ...",
            "train": "lerobot-train --policy.path=lerobot/smolvla_base --policy.device=cuda ...",
            "docs": "docs/DEPLOYMENT_SO100.md",
        },
        "note": "Complete checklist on site; train needs GPU or cloud.",
    }
    out = Path("outputs/so100_dry_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("See docs/DEPLOYMENT_SO100.md for full hardware path.")
    return 0 if import_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
