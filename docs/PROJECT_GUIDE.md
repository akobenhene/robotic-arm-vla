# Complete Project Guide — Text-Conditioned Robotic Manipulation via VLA Policies

**Repository:** [github.com/akobenhene/robotic-arm-vla](https://github.com/akobenhene/robotic-arm-vla)  
**Release:** [v1.0.0](https://github.com/akobenhene/robotic-arm-vla/releases/tag/v1.0.0)  
**Audience:** engineers, hiring managers, and **Grid Dynamics** Physical AI / Forward Deployed Engineering reviewers  

This document explains **what was built**, **what every important term means**, **why it matters**, and **how the same skills map to Grid Dynamics’ Physical AI, digital twin, and industrial automation work**.

---

## Table of contents

1. [Executive summary](#1-executive-summary)  
2. [What the project does (plain language)](#2-what-the-project-does-plain-language)  
3. [System walkthrough](#3-system-walkthrough)  
4. [Glossary of concepts (deep explanations)](#4-glossary-of-concepts-deep-explanations)  
5. [Repository map](#5-repository-map)  
6. [Results and how to read them](#6-results-and-how-to-read-them)  
7. [Relevance to Grid Dynamics](#7-relevance-to-grid-dynamics)  
8. [Link to soft sensors, hydrodynamics, and industrial estimation](#8-link-to-soft-sensors-hydrodynamics-and-industrial-estimation)  
9. [How to talk about this in an interview](#9-how-to-talk-about-this-in-an-interview)  
10. [Limitations and honest next steps](#10-limitations-and-honest-next-steps)  
11. [Quick reproduce commands](#11-quick-reproduce-commands)  

---

## 1. Executive summary

This project is a **portfolio-grade Physical AI demo**: a robot in simulation looks at the scene with a camera, optionally reads a language instruction, outputs motor commands, and we measure whether the task succeeded.

It proves four delivery-relevant capabilities:

| Capability | Evidence in this repo |
|------------|------------------------|
| Closed-loop visuomotor control | ACT policy drives Aloha arms until cube transfer succeeds |
| Multimodal (vision + language) policy interface | SmolVLA changes actions when the prompt changes |
| Quantitative evaluation | 50-seed CPU eval → **25/50 = 50%** success |
| Engineering packaging | typed modules, CI smoke tests, release artifacts, MP4/GIF, docs |

That combination matches how Grid Dynamics positions **GAIN Physical AI**: simulate → validate → refine foundation / imitation models → deploy with measurable KPIs — not “a random arm GIF.”

---

## 2. What the project does (plain language)

Imagine a tabletop with **two robot arms** and a **red cube**. The job is:

> Pick the cube with the right arm and hand it to the left arm so the left arm holds it off the table.

A neural network policy (the “brain”) repeatedly:

1. Sees an RGB image from above  
2. Reads joint positions (where the arms currently are)  
3. Optionally reads a sentence like *“Transfer the cube between the Aloha arms”*  
4. Outputs continuous joint/gripper commands  
5. The physics engine moves the robots one small step  

We record the camera frames into a GIF/MP4 and count **success** when the environment’s reward reaches level **4**.

Two brains are compared:

- **ACT** — strong at *doing the task* (hero demo)  
- **SmolVLA** — strong at *listening to language* (prompt ablation), weaker at full success until further finetuning  

---

## 3. System walkthrough

```
Language instruction (optional)
            │
            ▼
┌───────────────────────┐
│  Multimodal Policy π  │  ACT or SmolVLA (PyTorch / LeRobot)
│  image + state [+text]│
└───────────┬───────────┘
            │ action a_t ∈ R^14
            ▼
┌───────────────────────┐
│  Gymnasium + gym-aloha│  Aloha TransferCube task
│  MuJoCo physics       │
└───────────┬───────────┘
            │ new RGB + state + reward
            ▼
     GIF / MP4 / eval JSON
```

**Closed loop** means the policy’s previous action changes the next image and state — unlike offline “predict once” demos.

---

## 4. Glossary of concepts (deep explanations)

Each entry has: **definition → why it appears here → why Grid Dynamics cares**.

### 4.1 Physical AI

**Definition:** AI that **senses and acts in the physical world** (robots, factories, warehouses), not only text or recommendation systems.

**Here:** MuJoCo robot + camera → policy → motor commands.

**Grid Dynamics:** Core of [GAIN Physical AI](https://www.griddynamics.com/physical-ai): robotic manipulation, inspection, IoT quality, digital twins. Forward Deployed Engineers build these systems inside client environments.

---

### 4.2 Vision-Language-Action (VLA) model

**Definition:** A model that takes **vision** (images) + **language** (instructions) and outputs **actions** (robot controls).

**Here:** SmolVLA is a true VLA. ACT is vision–action (language API kept for interface compatibility).

**Grid Dynamics:** Clients want robots that follow natural-language work orders (“pick SKU A into bin B”) without rewriting controllers per SKU. VLA is the product pattern.

---

### 4.3 Imitation learning / Behavioral Cloning (BC)

**Definition:** Learn a policy by imitating expert demonstrations (human teleop or scripted experts), usually by regressing actions from observations — not by trial-and-error reward maximization alone.

**Here:** ACT and SmolVLA checkpoints were trained on demonstration datasets on the Hugging Face Hub.

**Grid Dynamics:** Factory skills are often easier to *show* than to *reward-engineer*. BC/VLA finetuning on client demos is a standard delivery path.

---

### 4.4 Policy π (pi)

**Definition:** The decision function `a = π(observation [, language])`.

**Here:** `policy.py` wraps ACT and SmolVLA behind `predict(observation, text)`.

**Grid Dynamics:** Clean policy interfaces let you swap models (ACT ↔ SmolVLA ↔ client finetune) without rewriting the robot loop — essential for modular GAIN-style stacks.

---

### 4.5 Observation / state / proprioception

**Definition:**

- **Observation:** everything the agent receives (images, vectors, language).  
- **Proprioception / state:** the robot’s own joint positions, gripper openings, etc.  
- **Exteroception:** external sensors (cameras, force/torque, lidar).

**Here:** `observation.images.top` (camera) + `observation.state` (14-D joints/grippers).

**Grid Dynamics:** Real cells mix cameras, encoders, PLC tags. Designing observation schemas and normalization is most of the integration work.

---

### 4.6 Action space

**Definition:** The set of valid control outputs. Continuous Box spaces use real-valued vectors (joint targets, deltas, gripper).

**Here:** Aloha actions are **14-dimensional** continuous commands for two arms.

**Grid Dynamics:** Action dimensions must match embodiment (Aloha ≠ SO-100 ≠ UR5). Embodiment mismatch is why we document finetune + hardware adaptation instead of claiming zero-shot transfer.

---

### 4.7 MuJoCo

**Definition:** A fast, accurate **physics engine** for robotics/biomechanics (contacts, joints, actuators).

**Here:** Backend physics for the Aloha arms and cube.

**Grid Dynamics:** Same role as NVIDIA Isaac / Omniverse in client digital twins — validate motion and contacts before risking hardware.

---

### 4.8 Gymnasium (and Gymnasium-Robotics)

**Definition:** Standard Python API for RL/control environments: `reset`, `step`, `observation_space`, `action_space`.

**Here:** `gym_aloha` registers `AlohaTransferCube-v0` as a Gymnasium env.

**Grid Dynamics:** A common “adapter layer” so research policies plug into industrial sims with a stable contract.

---

### 4.9 gym-aloha / ALOHA

**Definition:** **ALOHA** = low-cost **bi-manual** teleop platform; `gym-aloha` is its MuJoCo simulation tasks (transfer cube, insertion).

**Here:** Task env matching public LeRobot ACT weights.

**Grid Dynamics:** Shows you can align **policy training distribution** with **deployment env** — a recurring client failure mode when those diverge.

---

### 4.10 TransferCube task & sparse reward levels

**Definition:** Task success is staged:

| Reward | Meaning |
|--------|---------|
| 0 | No useful contact |
| 1 | Right gripper touches cube |
| 2 | Right lifts cube off table |
| 3 | Left gripper contacts cube (transfer attempt) |
| **4** | **Left holds cube off table → SUCCESS** |

**Here:** `is_success` when reward == 4; eval stops early on success for clean GIFs.

**Grid Dynamics:** Industrial KPIs are often sparse (pick success, cycle complete). Designing staged rewards/metrics is how you debug policies in the field.

---

### 4.11 Hugging Face Hub / LeRobot

**Definition:**

- **Hugging Face Hub:** host for models/datasets.  
- **LeRobot:** HF’s robotics library for datasets, policies (ACT, Diffusion, SmolVLA), train/eval/record.

**Here:** Load ACT & SmolVLA from Hub; evaluate in our wrapper; CI caches Hub downloads.

**Grid Dynamics:** Modern delivery uses shared model hubs + internal registries. Knowing LeRobot is knowing the open Physical AI toolchain clients increasingly ask about (alongside NVIDIA).

---

### 4.12 ACT (Action Chunking with Transformers)

**Definition:** Imitation policy that predicts a **chunk** of future actions at once (not one joint tick only), reducing compounding error.

**Here:** Official `lerobot/act_aloha_sim_transfer_cube_human` — **solves** TransferCube in our demos (seed 36 success).

**Grid Dynamics:** Strong baseline for short-horizon manipulation skills when language is not required.

---

### 4.13 Action chunking

**Definition:** Output `T` future actions per forward pass; execute them sequentially; replan when the queue empties.

**Here:** ACT/SmolVLA maintain an action queue inside `select_action`.

**Grid Dynamics:** Improves smoothness and latency tradeoffs on real robots (fewer GPU forwards per second of motion).

---

### 4.14 SmolVLA

**Definition:** Compact (~450M) open **VLA** from Hugging Face — vision backbone + language tokens + action expert.

**Here:** Community Aloha-matched checkpoint; **prompt changes actions** (`language_sensitive: true`), full task success needs finetune (playbook provided).

**Grid Dynamics:** Shows you can productize “instruction-following robots” and measure language sensitivity separately from task success — mature evaluation thinking.

---

### 4.15 Tokenizer / language conditioning

**Definition:** Convert text to token IDs the model embeds; conditioning means those embeddings influence action decoding.

**Here:** SmolVLA preprocessor tokenizes the `task` string (newline-terminated).

**Grid Dynamics:** Same pattern as grounding LLMs to tools — but the “tool” is a robot.

---

### 4.16 Normalization / denormalization

**Definition:** Scale inputs/outputs using dataset statistics (mean/std or min/max) so training is stable; invert for real units at deploy time.

**Here:** ACT Hub weights embed normalize buffers; we re-apply them under LeRobot 0.4.x. SmolVLA uses Hub preprocessor/postprocessor pipelines.

**Grid Dynamics:** Silent normalization bugs are a top cause of “model works in notebook, fails on robot.” Explicit handling is a senior signal.

---

### 4.17 Distribution shift

**Definition:** Deploy-time inputs differ from train-time inputs (camera pose, lighting, zero-action warm-up, different embodiment).

**Here:** We removed zero-action warm-up after reset because it desynced ACT from its training distribution and killed success.

**Grid Dynamics:** FDEs spend weeks on shift: camera mounts, latency, domain randomization, finetune on site data.

---

### 4.18 Seed / stochastic evaluation

**Definition:** A seed fixes randomness (cube pose sampling, etc.) for reproducibility; multi-seed eval estimates success rate.

**Here:** Seeds 0–49 → 50% CPU success; Hub reports ~83% on GPU/official eval.

**Grid Dynamics:** Clients need **rates**, not one lucky video. Multi-seed eval is the professional standard.

---

### 4.19 Digital twin (related concept)

**Definition:** A virtual replica of a physical system used to test control/AI before deployment.

**Here:** MuJoCo Aloha is a lightweight task twin. Client twins may use Omniverse/Isaac (Grid Dynamics’ NVIDIA partnership).

**Grid Dynamics:** [Digital twin solutions](https://www.griddynamics.com/solutions/digital-twin) validate schedules and robot motion in sim — same philosophy as this repo, different fidelity stack.

---

### 4.20 Sim-to-real

**Definition:** Transfer policies from simulation to hardware.

**Here:** Documented in `docs/DEPLOYMENT_SO100.md` — record demos → finetune → deploy; **no false claim** that Aloha ACT zero-shots to SO-100.

**Grid Dynamics:** Honest sim-to-real narrative is exactly how FDEs should speak with manufacturing clients.

---

### 4.21 Soft sensor (cross-link to your thesis theme)

**Definition:** Infer hard-to-measure quantities from easy sensors + models (e.g. mixing quality from images/pressure; 3D flow fields from 2D views).

**Here (analogy):** The policy is a **control soft sensor + actuator**: it infers “what to do next” from cameras/state — the control counterpart to estimating hidden process variables.

**Grid Dynamics:** Soft sensing (vision QC, anomaly scores) and Physical AI control often share the same stack: multimodal models, edge inference, digital twins, KPI dashboards.

---

### 4.22 MLOps / CI for robotics

**Definition:** Automate test/build of ML systems so regressions are caught early.

**Here:** GitHub Actions smoke: imports + 5-step ACT + SmolVLA prompt ablation.

**Grid Dynamics:** Enterprise clients expect CI/CD for AI, not laptop demos. Robotics CI with sim smoke tests is a reusable blueprint.

---

### 4.23 Forward Deployed Engineer (FDE)

**Definition:** Grid Dynamics’ model of embedding engineers with the client to build against real constraints.

**Here (skill signal):** You can own a vertical slice — env adapters, Hub policies, eval KPIs, packaging, docs — the same ownership FDEs need on factory projects.

---

## 5. Repository map

| Path | Role |
|------|------|
| `env_wrapper.py` | Gymnasium/Aloha → tensors + LeRobot batches |
| `policy.py` | ACT, SmolVLA, Mock policies |
| `main.py` | Single-episode rollout + GIF |
| `evaluate.py` | Multi-seed success sweep |
| `prompt_ablation.py` | Language sensitivity proof |
| `compare_policies.py` / `viz.py` | Side-by-side HUD comparison |
| `export_mp4.py` | Portfolio MP4s |
| `smoke_test.py` + `.github/workflows/` | CI |
| `docs/FINETUNE_SMOLVLA.md` | GPU finetune recipe |
| `docs/DEPLOYMENT_SO100.md` | Hardware path |
| `TECH_REPORT.md` | Short technical summary |
| `demo_output.gif/.mp4` | ACT success hero |
| `comparison_act_vs_smolvla.gif/.mp4` | Ablation visual |

---

## 6. Results and how to read them

### 6.1 ACT task success

- **Local CPU eval:** 25 successes / 50 seeds = **50.0%** (`eval_results.json`)  
- **Hub reference:** ~**83%** / 500 episodes (official GPU eval)  
- **Hero episode:** seed **36**, reward 4, full transfer (`demo_output.gif` / `.mp4`)

Gap vs Hub is expected (CPU, tooling differences). The point is **reproducible nonzero success**, not matching the paper number on a laptop.

### 6.2 SmolVLA language conditioning

- Same image/state, two prompts → different actions  
- L1 action delta ≈ **0.03**  
- `language_sensitive: true` in `prompt_ablation.json`  
- Task success not reliable yet → finetune playbook ready  

### 6.3 Ablation story (interview-ready)

| Backend | Solves task? | Uses language? | Use in pitch |
|---------|--------------|----------------|--------------|
| ACT | Yes | No | Reliability / KPI |
| SmolVLA | Not yet | Yes | Instruction interface |
| Finetuned SmolVLA (future) | Target yes | Yes | Production VLA |

---

## 7. Relevance to Grid Dynamics

Grid Dynamics publicly invests in **Physical AI**, **digital twins**, **NVIDIA partnerships**, and **Forward Deployed Engineering** for manufacturing, logistics, and warehouse automation ([Physical AI / GAIN](https://www.griddynamics.com/physical-ai), [IoT & robotics services](https://www.griddynamics.com/services/iot-and-edge-computing), [digital twins](https://www.griddynamics.com/solutions/digital-twin)).

### 7.1 Direct skill alignment

| Grid Dynamics theme | This project demonstrates |
|---------------------|---------------------------|
| Robotic manipulation solutions | Closed-loop ACT on bimanual transfer |
| Validate in simulation before hardware | MuJoCo + multi-seed success metrics |
| Foundation / imitation model refinement | Hub load + SmolVLA finetune playbook |
| Agentic / instruction-driven workflows | Language-conditioned SmolVLA interface |
| Production engineering | Typed code, CI, releases, artifacts, docs |
| NVIDIA / modern AI stack adjacency | Same problem class as Isaac/Omniverse twins (different engine) |

### 7.2 How it can be of use on client work

1. **Discovery / PoC sprint**  
   Spin up a task twin, plug a Hub policy, measure success rate in days — not months of greenfield RL.

2. **KPI-driven demos for stakeholders**  
   GIF/MP4 + `eval_results.json` is how you convince plant managers (visual) and CTOs (numbers).

3. **Modular policy swap**  
   ACT today, client-finetuned SmolVLA tomorrow, same `predict()` loop — matches GAIN’s modular platform idea.

4. **FDE onboarding pattern**  
   Env wrapper + eval harness + CI smoke is a template for embedding into a client’s ROS/edge stack.

5. **Cross-sell into vision QC / soft sensing**  
   Same multimodal + edge inference skills apply to Metropolis-style inspection and process soft sensors.

### 7.3 What you are *not* claiming (important)

- Not claiming Omniverse-scale fidelity  
- Not claiming Aloha ACT zero-shots onto SO-100  
- Not claiming 83% on a laptop  

That honesty is an asset for Grid Dynamics clients who have been burned by overpromised robotics demos.

---

## 8. Link to soft sensors, hydrodynamics, and industrial estimation

Your broader research theme — **AI soft sensors, deep learning, physics-informed networks, mixing state estimation, 3D reconstruction from 2D** — connects cleanly:

| Soft-sensor / process AI idea | Robotics analogue in this repo |
|-------------------------------|--------------------------------|
| Infer hidden state from cheap sensors | Infer actions from RGB + proprioception |
| Physics-informed constraints | MuJoCo contact/joint physics as ground truth in sim |
| 2D→3D / spatial understanding | Camera → spatial manipulation of a cube |
| Online estimation loop | Closed-loop `obs → π → action → obs'` |
| Industrial automation KPI | Success rate, cycle completion (reward 4) |

At Grid Dynamics you can position yourself as someone who spans **perception → estimation → action**: soft sensing for *visibility*, Physical AI for *actuation*, digital twins for *safe validation*.

---

## 9. How to talk about this in an interview

**30-second pitch**

> I built an end-to-end Physical AI pipeline: MuJoCo Aloha simulation, Hugging Face LeRobot ACT for successful bimanual cube transfer, and SmolVLA to show language-conditioned actions. I evaluated 50 seeds at 50% CPU success, packaged CI and a release, and documented finetune + SO-100 deployment paths. It’s the same simulate → measure → refine loop Grid Dynamics uses in GAIN Physical AI engagements.

**Questions you can answer**

- Why ACT succeeded after removing reset warm-up (distribution shift)  
- Why normalization buffers matter  
- Why language sensitivity ≠ task success  
- How you’d finetune SmolVLA on client demos  
- How you’d move to SO-100 without lying about embodiment  

**Artifacts to send**

1. Release: https://github.com/akobenhene/robotic-arm-vla/releases/tag/v1.0.0  
2. Hero MP4: `demo_output.mp4`  
3. Ablation: `comparison_act_vs_smolvla.mp4`  
4. This guide + `TECH_REPORT.md`  

---

## 10. Limitations and honest next steps

| Limitation | Mitigation / next step |
|------------|------------------------|
| SmolVLA weak on full transfer | GPU finetune (`docs/FINETUNE_SMOLVLA.md`) |
| CPU eval below Hub 83% | GPU eval; more seeds; domain randomization |
| Sim-only | SO-100 record/finetune path documented |
| Single task | Add insertion / pick-place envs with matched checkpoints |
| No force/torque | Add tactile/wrench for contact-rich client cells |

---

## 11. Quick reproduce commands

```powershell
.\.venv\Scripts\python.exe main.py --policy act --seed 36
.\.venv\Scripts\python.exe prompt_ablation.py
.\.venv\Scripts\python.exe compare_policies.py --seed 36
.\.venv\Scripts\python.exe evaluate.py --policy act --seeds 0-49 --continue-after-success
.\.venv\Scripts\python.exe export_mp4.py
.\.venv\Scripts\python.exe smoke_test.py
.\.venv\Scripts\python.exe scripts\print_finetune_plan.py
```

---

## Closing

This project is a **miniature Physical AI delivery**: perception, multimodal policy, closed-loop control, quantitative eval, CI, and documentation.  

For **Grid Dynamics**, it is evidence you can contribute immediately to GAIN-style engagements — digital twin validation, imitation/VLA policy integration, KPI-driven demos, and the engineering discipline required to take AI from a notebook to a plant floor.
