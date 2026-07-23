"""Print a short summary of outputs/run_log.jsonl for debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize run_log.jsonl")
    parser.add_argument("--log", type=str, default="outputs/run_log.jsonl")
    parser.add_argument("--last", type=int, default=30)
    args = parser.parse_args()
    path = Path(args.log)
    if not path.exists():
        print(f"No log yet: {path.resolve()}")
        print("Run Streamlit once first.")
        return 1

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    print(f"Log: {path.resolve()}")
    print(f"Total events: {len(lines)}")
    print("--- last events ---")
    for line in lines[-args.last :]:
        rec = json.loads(line)
        event = rec.get("event")
        if event == "config":
            print(
                f"[{rec['run_id']}] CONFIG policy={rec.get('policy')} "
                f"seed={rec.get('seed')} A={rec.get('prompt_a')!r} B={rec.get('prompt_b')!r}"
            )
        elif event == "prompt_ablation":
            print(
                f"[{rec['run_id']}] ABLATION policy={rec.get('policy')} "
                f"l1={rec.get('l1_delta'):.5f} sensitive={rec.get('language_sensitive')} "
                f"prompts_equal={rec.get('prompts_equal')}"
            )
        elif event == "episode":
            print(
                f"[{rec['run_id']}] EPISODE success={rec.get('success')} "
                f"max_reward={rec.get('max_reward')} steps={rec.get('steps')}"
            )
        elif event == "predict":
            act = rec.get("action") or []
            head = [round(float(x), 4) for x in act[:3]]
            print(
                f"[{rec['run_id']}] PREDICT {rec.get('which')} "
                f"task={rec.get('task_sent_to_model')!r} action[:3]={head}"
            )
        else:
            print(f"[{rec.get('run_id')}] {event}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
