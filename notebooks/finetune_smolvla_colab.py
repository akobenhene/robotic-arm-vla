# SmolVLA finetune on Google Colab / cloud GPU
# Copy this file into a Colab notebook (one cell per section) when you have a GPU.
#
# Runtime: GPU (T4/A10/L4). Do not expect this to finish on CPU.

# %% [markdown]
# # SmolVLA finetune — Aloha TransferCube
# Dataset: `lerobot/aloha_sim_transfer_cube_human`
# Base: `lerobot/smolvla_base`

# %%
# Cell 1 — install (Colab)
# !pip install -U pip
# !pip install "lerobot[smolvla]" torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# %%
# Cell 2 — sanity
import torch

assert torch.cuda.is_available(), "Enable Colab GPU Runtime before training"
print("cuda:", torch.cuda.get_device_name(0))

# %%
# Cell 3 — train (adjust batch_size if OOM)
# !lerobot-train \
#   --policy.path=lerobot/smolvla_base \
#   --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
#   --batch_size=4 \
#   --steps=20000 \
#   --eval_freq=2000 \
#   --save_freq=2000 \
#   --output_dir=outputs/train/smolvla_aloha_transfer \
#   --job_name=smolvla_aloha_transfer \
#   --policy.device=cuda \
#   --policy.push_to_hub=false \
#   --wandb.enable=false

# %%
# Cell 4 — after training, download checkpoint and evaluate locally:
# .\.venv\Scripts\python.exe evaluate.py --policy smolvla --repo-id <your_ckpt_or_local> --seeds 0-49 --continue-after-success
# .\.venv\Scripts\python.exe compare_policies.py --seed 36

print("Uncomment Colab cells when GPU runtime is available. See docs/FINETUNE_SMOLVLA.md")
