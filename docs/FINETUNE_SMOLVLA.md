# SmolVLA Finetune Playbook — Aloha TransferCube

Goal: close the gap where **ACT solves** TransferCube but community **SmolVLA** is language-sensitive yet rarely succeeds — by finetuning SmolVLA on the official Aloha transfer demonstrations.

> Full finetuning needs a **GPU** (recommended: ≥16 GB VRAM).  
> **No local GPU?** Use Google Colab / RunPod with [`notebooks/finetune_smolvla_colab.py`](../notebooks/finetune_smolvla_colab.py).  
> On CPU, keep shipping ACT success + SmolVLA language ablation; see [`docs/NEXT_LEVEL_CPU.md`](NEXT_LEVEL_CPU.md).

---

## 1. Why finetune?

| Checkpoint | Language? | Solves TransferCube? |
|------------|-----------|----------------------|
| `lerobot/act_aloha_sim_transfer_cube_human` | No | Yes (~50% local CPU / ~83% Hub GPU) |
| `crislmfroes/smolvla-aloha-sim-transfer-cube-scripted` | Yes | Rarely (scripted / weak) |
| **Your finetuned SmolVLA** (this playbook) | Yes | Target: match ACT success *with* prompts |

---

## 2. Dataset

Use the LeRobot Hub dataset aligned with gym-aloha TransferCube:

- **Dataset:** [`lerobot/aloha_sim_transfer_cube_human`](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)
- **Base policy:** [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)  
  or continue from `crislmfroes/smolvla-aloha-sim-transfer-cube-scripted` if observation keys already match (`observation.images.top`, state 14-D).

Verify camera / state keys in the dataset card before training. If keys differ from gym-aloha, add a LeRobot rename map in the train config.

---

## 3. Environment

```bash
# Linux GPU node recommended
python -m venv .venv-train
source .venv-train/bin/activate
pip install -U pip
pip install "lerobot[smolvla]" torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
```

Windows CUDA works if you already have a CUDA PyTorch build; prefer Linux for long jobs.

---

## 4. Train command

```bash
# From repo root (or any dir with lerobot CLI on PATH)
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --batch_size=8 \
  --steps=20000 \
  --eval_freq=2000 \
  --save_freq=2000 \
  --output_dir=outputs/train/smolvla_aloha_transfer \
  --job_name=smolvla_aloha_transfer \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

### Suggested knobs

| Flag | Starter value | Notes |
|------|---------------|-------|
| `--steps` | `20000` | ~paper-scale light finetune; scale up if reward plateaus |
| `--batch_size` | `4–16` | Lower if OOM |
| `--policy.chunk_size` / `n_action_steps` | keep defaults | Match deployment horizon |
| Learning rate | SmolVLA default | Prefer `train_expert_only=true` (config default) |

Helper scripts in this repo:

- `scripts/finetune_smolvla.sh` (Linux/macOS)
- `scripts/finetune_smolvla.ps1` (Windows)

Dry-run config print (no training):

```powershell
.\.venv\Scripts\python.exe scripts\print_finetune_plan.py
```

---

## 5. Evaluate in this repo

After training, point the demo at your checkpoint directory or Hub repo:

```powershell
.\.venv\Scripts\python.exe main.py --policy smolvla --repo-id ./outputs/train/smolvla_aloha_transfer/checkpoints/last/pretrained_model --seed 36 --steps 400
.\.venv\Scripts\python.exe evaluate.py --policy smolvla --repo-id <your-hf-repo> --seeds 0-19 --steps 400 --continue-after-success
.\.venv\Scripts\python.exe prompt_ablation.py --repo-id <your-hf-repo>
```

Success criterion unchanged: Aloha TransferCube **reward == 4**.

---

## 6. Acceptance bar (portfolio)

Ship a finetune when:

1. `evaluate.py` SmolVLA success rate **≥ 40%** on seeds 0–19 (CPU or GPU)  
2. `prompt_ablation.py` still reports `language_sensitive: true`  
3. `compare_policies.py` shows SmolVLA panel reaching **SUCCESS** on seed 36  

Until then, keep **ACT as the hero GIF** and SmolVLA as the language-ablation story (current `v1.0.0` release).

---

## 7. Cost / time (approximate)

| Hardware | 20k steps |
|----------|-----------|
| 1× A100 40GB | ~3–5 hours |
| 1× RTX 4090 24GB | ~6–10 hours |
| CPU only | not recommended |

---

## References

- SmolVLA blog: https://huggingface.co/blog/smolvla  
- LeRobot SmolVLA docs: https://huggingface.co/docs/lerobot/smolvla  
- ACT TransferCube model card: https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human  
