"""Append-only JSONL run logger for Streamlit / CLI demos."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

DEFAULT_LOG_PATH = Path("outputs") / "run_log.jsonl"


def _to_list(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_list(v) for k, v in x.items()}
    return x


class RunLogger:
    """Writes one JSON object per line for post-mortem checks."""

    def __init__(self, path: Path | str = DEFAULT_LOG_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._t0 = time.perf_counter()

    def log(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            "elapsed_s": round(time.perf_counter() - self._t0, 3),
            **{k: _to_list(v) for k, v in payload.items()},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        return record

    def log_config(self, **config: Any) -> dict[str, Any]:
        return self.log("config", **config)

    def log_prompt_ablation(
        self,
        *,
        policy: str,
        prompt_a: str,
        prompt_b: str,
        action_a: torch.Tensor,
        action_b: torch.Tensor,
        seed: int,
    ) -> dict[str, Any]:
        delta = action_a - action_b
        l1 = float(delta.abs().mean().item())
        l2 = float(torch.linalg.vector_norm(delta).item())
        return self.log(
            "prompt_ablation",
            policy=policy,
            seed=seed,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            prompt_a_len=len(prompt_a),
            prompt_b_len=len(prompt_b),
            prompts_equal=prompt_a.strip() == prompt_b.strip(),
            action_a=_to_list(action_a),
            action_b=_to_list(action_b),
            l1_delta=l1,
            l2_delta=l2,
            language_sensitive=l1 > 1e-4,
            expected_sensitive_if_smolvla=policy == "smolvla",
        )

    def log_episode(
        self,
        *,
        policy: str,
        prompt: str,
        seed: int,
        steps: int,
        max_reward: float,
        success: bool,
        first_action: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return self.log(
            "episode",
            policy=policy,
            prompt=prompt,
            seed=seed,
            steps=steps,
            max_reward=max_reward,
            success=success,
            first_action=_to_list(first_action) if first_action is not None else None,
        )
