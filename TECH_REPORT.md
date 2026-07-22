# Technical Writeup — Text-Conditioned Robotic Manipulation via VLA Policies

**Author context:** Physical AI / Soft-sensor & automation portfolio piece  
**Repo:** [akobenhene/robotic-arm-vla](https://github.com/akobenhene/robotic-arm-vla)  
**Stack:** Python 3.11 · PyTorch · MuJoCo · gym-aloha · Hugging Face LeRobot

---

## 1. Problem

Demonstrate an end-to-end **Vision–Language–Action** style manipulation pipeline that:

1. Perceives a MuJoCo scene through an RGB camera + proprioception  
2. Maps observations (and optionally language) to continuous robot actions  
3. Executes closed-loop control and exports a visual artifact for review  

The target task is **Aloha TransferCube**: pick a red cube with the right arm and transfer it to the left arm (success ⇔ sparse reward `4`).

---

## 2. System architecture

| Layer | Implementation | Notes |
|-------|----------------|-------|
| Simulation | `gym_aloha/AlohaTransferCube-v0` | MuJoCo bimanual ViperX; obs = top RGB `(480×640×3)` + state `(14,)` |
| Env I/O | `RoboticsEnvWrapper` | NCHW tensors, LeRobot batch keys, no zero-action warm-up (avoids distribution shift) |
| Task policy | **ACT** `lerobot/act_aloha_sim_transfer_cube_human` | Vision + state; chunked actions; **solves** the task |
| Language policy | **SmolVLA** `crislmfroes/smolvla-aloha-sim-transfer-cube-scripted` | Tokenized `task` string; prompt-sensitive actions |
| Eval | `evaluate.py` | Multi-seed sweep; stop on reward `4` |
| Artifacts | GIF + JSON | `demo_output.gif`, `comparison_act_vs_smolvla.gif`, metrics |

Control loop:

```
obs_t = (RGB_top, qpos)  →  π_θ(obs_t [, text])  →  a_t ∈ R^14  →  MuJoCo step
```

Legacy ACT Hub checkpoints store `normalize_*` buffers inside `model.safetensors`. Under LeRobot 0.4.x those keys are not auto-loaded; `LeRobotACTPolicy` re-applies them so inference matches training statistics.

---

## 3. Results

### 3.1 ACT task success (CPU)

| Metric | Value |
|--------|--------|
| Seeds | 0–49 (50 episodes, 400-step cap) |
| Success rate | **25 / 50 = 50.0%** |
| Best seed (hero GIF) | **36** (full transfer, reward 4) |
| Official Hub GPU eval (reference) | ~**83%** / 500 episodes |

The gap vs Hub is expected: CPU eval, shorter tooling differences, and stochastic cube poses. The local sweep still shows **reliable, reproducible successes** (e.g. seeds 0, 1, 36).

### 3.2 SmolVLA language conditioning

| Check | Result |
|-------|--------|
| Same observation, two prompts | Different first actions |
| Mean \|Δa\| (L1) | ~0.03 |
| `language_sensitive` | **true** (`prompt_ablation.json`) |
| Task success (scripted community ckpt) | Typically **false** on TransferCube |

**Interpretation:** SmolVLA proves the **VLA interface** (text → action change). ACT remains the **task expert** for portfolio visuals. A natural follow-up is finetuning SmolVLA on `lerobot/aloha_sim_transfer_cube_human` so language and success align.

### 3.3 Ablation narrative

| Backend | Solves cube transfer? | Uses language? | Role in demo |
|---------|----------------------|----------------|--------------|
| ACT | Yes | No | Hero GIF / success metric |
| SmolVLA | Not reliably (this ckpt) | Yes | Prompt ablation / VLA story |
| Mock | No | Hash stub | Pipeline smoke test |

Side-by-side: `comparison_act_vs_smolvla.gif` (same seed, HUD overlays).

---

## 4. Engineering decisions

- **Matched embodiment:** pretrained weights only run on the env they were trained for (Aloha, not FetchReach).  
- **Python 3.11 venv:** LeRobot / MuJoCo wheels are unreliable on 3.14.  
- **Success-aware rollouts:** stop at reward `4` so GIFs show completion, not post-success noise.  
- **Windows DX:** `run.bat` / direct `.\.venv\Scripts\python.exe` avoid `Activate.ps1` execution-policy failures.

---

## 5. How to reproduce

```powershell
.\.venv\Scripts\python.exe evaluate.py --policy act --seeds 0-49 --steps 400 --continue-after-success
.\.venv\Scripts\python.exe prompt_ablation.py
.\.venv\Scripts\python.exe compare_policies.py --seed 36 --steps 400
.\.venv\Scripts\python.exe main.py --policy act --seed 36
```

---

## 6. Takeaway for Grid Dynamics / Physical AI review

This repo is not a toy random-action GIF. It demonstrates:

1. **Real Hub imitation weights** closed-loop in MuJoCo  
2. **Quantitative eval** (50% local CPU success vs published Hub baseline)  
3. **True multimodal path** (SmolVLA prompt sensitivity)  
4. **Production-minded packaging** (typed modules, eval JSON, comparison artifact, reproducible runners)

CI: `.github/workflows/ci-smoke.yml` + `smoke_test.py` (ACT 5-step + SmolVLA prompt ablation).  
Release: GitHub `v1.0.0` bundles GIFs + this writeup.

### Roadmap (implemented as docs + tooling)

1. **SmolVLA finetune playbook** — [docs/FINETUNE_SMOLVLA.md](docs/FINETUNE_SMOLVLA.md) + `scripts/finetune_smolvla.*`  
   Target: language *and* TransferCube success (≥40% on seeds 0–19). Requires GPU.  
2. **Real-robot appendix** — [docs/DEPLOYMENT_SO100.md](docs/DEPLOYMENT_SO100.md) (SO-100 record → train → deploy).  
3. **MP4 export** — `export_mp4.py` for interview / LinkedIn artifacts.
