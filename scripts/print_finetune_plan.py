"""Print the SmolVLA TransferCube finetune plan (no training / no GPU required)."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    """Emit a machine-readable finetune plan for docs / CI sanity."""
    plan = {
        "goal": "Finetune SmolVLA so language conditioning and TransferCube success align",
        "base_policy": "lerobot/smolvla_base",
        "dataset_repo_id": "lerobot/aloha_sim_transfer_cube_human",
        "env_id": "gym_aloha/AlohaTransferCube-v0",
        "recommended_steps": 20000,
        "recommended_batch_size": 8,
        "device": "cuda",
        "success_metric": "reward == 4 (is_success)",
        "acceptance": {
            "min_success_rate_seeds_0_19": 0.4,
            "prompt_ablation_language_sensitive": True,
            "compare_seed_36_smolvla_success": True,
        },
        "launchers": [
            "scripts/finetune_smolvla.sh",
            "scripts/finetune_smolvla.ps1",
        ],
        "docs": "docs/FINETUNE_SMOLVLA.md",
        "eval_after_train": [
            "python main.py --policy smolvla --repo-id <ckpt> --seed 36 --steps 400",
            "python evaluate.py --policy smolvla --repo-id <ckpt> --seeds 0-19 --continue-after-success",
            "python prompt_ablation.py --repo-id <ckpt>",
        ],
    }
    print(json.dumps(plan, indent=2))
    docs = Path("docs/FINETUNE_SMOLVLA.md")
    if not docs.exists():
        raise SystemExit(f"Missing {docs}")
    print(f"[ok] playbook present: {docs.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
