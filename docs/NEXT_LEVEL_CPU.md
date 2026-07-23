# Next-level roadmap (CPU now, GPU later)

You can implement **all next levels** without a local GPU by splitting work into:

| Track | Runs on CPU today | Needs cloud GPU later |
|-------|-------------------|------------------------|
| Stronger evaluation | Yes | Optional speed-up |
| Domain randomization | Yes | — |
| Prompt bank / multi-instruction eval | Yes | — |
| Soft-sensor confidence head | Yes (small CNN) | Optional larger models |
| Streamlit demo dashboard | Yes | — |
| Docker packaging | Yes (Linux image) | — |
| SO-100 / sim-to-real scaffolding | Docs + dry-run | Hardware + optional CUDA |
| SmolVLA full finetune | Colab/cloud recipe only | **Required for real train** |
| Isaac / Omniverse twin | Scaffold docs | GPU workstation/cloud |

## What was added in this repo

| Path | Purpose |
|------|---------|
| `metrics_eval.py` | Wilson CI, steps-to-success, latency aggregates |
| `evaluate.py` | Extended JSON metrics (CI, timing) |
| `domain_randomization.py` | CPU observation noise / brightness jitter |
| `prompt_bank.py` + `evaluate_prompts.py` | Multi-instruction ablation |
| `soft_sensor.py` + `train_soft_sensor.py` | Reward-stage soft sensor from RGB |
| `app_streamlit.py` | Interactive demo dashboard |
| `Dockerfile` + `docker-compose.yml` | One-command Linux demo |
| `notebooks/finetune_smolvla_colab.py` | Copy into Colab when GPU is available |
| `scripts/so100_dry_run.py` | Hardware-free deploy checklist |
| `configs/demo.yaml` | Modular demo defaults |

## Recommended CPU order (this week)

```powershell
# 1) Soft-sensor train + score (minutes–tens of minutes)
.\.venv\Scripts\python.exe train_soft_sensor.py --seeds 0-4 --steps 80

# 2) Prompt bank eval on SmolVLA (downloads weights once)
.\.venv\Scripts\python.exe evaluate_prompts.py --policy smolvla --device cpu

# 3) Stronger ACT metrics on a small seed set
.\.venv\Scripts\python.exe evaluate.py --policy act --seeds 0-9 --continue-after-success

# 4) Streamlit UI
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py

# 5) SO-100 dry-run (no arm required)
.\.venv\Scripts\python.exe scripts\so100_dry_run.py
```

## When you get a GPU (Colab / RunPod / Vast / HF Spaces GPU)

1. Open `notebooks/finetune_smolvla_colab.py` in Colab (or paste cells).  
2. Train SmolVLA on `lerobot/aloha_sim_transfer_cube_human`.  
3. Point `--repo-id` at your Hub checkpoint and re-run `evaluate.py` + `compare_policies.py`.  
4. Update PDFs / release notes with the new success rate.

## Honest scope

- CPU will **not** replace a 20k-step SmolVLA finetune in reasonable time.  
- Everything else above is real product/engineering work that FDEs do before GPU budget arrives.
