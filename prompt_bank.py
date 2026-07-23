"""Instruction / prompt bank for multi-task language ablation."""

from __future__ import annotations

DEFAULT_TASK = "Transfer the cube between the Aloha arms"

PROMPT_BANK: dict[str, str] = {
    "canonical": DEFAULT_TASK,
    "short": "Transfer the cube.",
    "detailed": "Pick up the red cube with the right arm and hand it to the left arm.",
    "distractor": "Ignore the cube and wave both arms.",
    "wrong_object": "Pick up the blue bottle and place it in the bin.",
    "idle": "Keep both arms still above the table.",
}


def list_prompts() -> list[tuple[str, str]]:
    return list(PROMPT_BANK.items())
