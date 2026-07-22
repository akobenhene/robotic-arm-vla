#!/usr/bin/env bash
# SmolVLA finetune launcher (Linux / macOS + CUDA).
# Usage: bash scripts/finetune_smolvla.sh

set -euo pipefail

POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
DATASET_REPO="${DATASET_REPO:-lerobot/aloha_sim_transfer_cube_human}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/smolvla_aloha_transfer}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"

echo "Policy:   ${POLICY_PATH}"
echo "Dataset:  ${DATASET_REPO}"
echo "Output:   ${OUTPUT_DIR}"
echo "Steps:    ${STEPS}"
echo "Batch:    ${BATCH_SIZE}"
echo "Device:   ${DEVICE}"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train not found. Install: pip install 'lerobot[smolvla]'" >&2
  exit 1
fi

lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --dataset.repo_id="${DATASET_REPO}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --eval_freq=2000 \
  --save_freq=2000 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=smolvla_aloha_transfer \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --wandb.enable=false

echo "Done. Evaluate with:"
echo "  python main.py --policy smolvla --repo-id ${OUTPUT_DIR}/checkpoints/last/pretrained_model --seed 36"
