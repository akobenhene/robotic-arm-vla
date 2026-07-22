# Full Report: Text-Conditioned Robotic Manipulation via Vision–Language–Action Policies

**Project title:** AI-enabled Physical AI demo — closed-loop bimanual manipulation with imitation and VLA policies  
**Repository:** [github.com/akobenhene/robotic-arm-vla](https://github.com/akobenhene/robotic-arm-vla)  
**Release:** [v1.0.0](https://github.com/akobenhene/robotic-arm-vla/releases/tag/v1.0.0)  
**Stack:** Python 3.11 · PyTorch · MuJoCo · gym-aloha · Hugging Face LeRobot  
**Audience:** technical reviewers, hiring managers, and **Grid Dynamics** Physical AI / Forward Deployed Engineering (FDE) teams  
**Related research theme:** AI-based soft sensors, deep learning, and physics-informed methods for industrial estimation and automation  

---

## Abstract

This report presents an end-to-end **Physical AI** system for tabletop robotic manipulation. A bimanual Aloha robot in a MuJoCo simulation observes the scene through an RGB camera and proprioceptive joint state, optionally conditioned on a natural-language instruction, and outputs continuous motor commands in a closed control loop. Two policy families from Hugging Face **LeRobot** are integrated and compared: **ACT** (Action Chunking with Transformers), which reliably solves the TransferCube task, and **SmolVLA**, a compact vision–language–action model that demonstrably changes actions when the language prompt changes.

Quantitative evaluation over 50 random seeds yields a **50%** success rate for ACT on CPU (25/50), with a reference Hub GPU evaluation near **83%**. The system is packaged with multi-seed evaluation, prompt ablation, side-by-side visualization, CI smoke tests, GIF/MP4 artifacts, and documentation for SmolVLA finetuning and SO-100 hardware deployment. The work is framed as a miniature delivery of the simulate → validate → refine → package loop that underpins Grid Dynamics’ **GAIN Physical AI**, digital-twin, and industrial automation engagements.

---

## Framework diagrams (Figma / FigJam)

Editable board with all architecture and process frameworks:

**[Open FigJam board](https://www.figma.com/board/CmvFbnixCtXsehlEUMbEnZ)**

**Print PDFs (frameworks embedded):**

| PDF | Use |
|-----|-----|
| [Full_Project_Report.pdf](pdf/Full_Project_Report.pdf) | Complete report with all Figma figures |
| [Grid_Dynamics_Relevance_Brief.pdf](pdf/Grid_Dynamics_Relevance_Brief.pdf) | Screening / FDE brief |
| [Frameworks_Atlas.pdf](pdf/Frameworks_Atlas.pdf) | Diagram-only atlas |

| Diagram | Purpose |
|---------|---------|
| Physical AI System Architecture | CLI → rollout → policies → MuJoCo / Hub / artifacts |
| Closed-Loop Control Framework | Sense → policy → step → success check |
| Grid Dynamics GAIN Delivery Loop | Discover → simulate → KPI → refine → FDE deploy |
| Evaluation and Ablation Framework | ACT success track vs SmolVLA language track |
| Soft Sensor to Physical AI Mapping | Thesis soft sensing ↔ this robotics stack |
| TransferCube Reward State Machine | Sparse reward ladder 0 → 4 |

---

## Table of contents

1. [Introduction](#1-introduction)  
2. [Relevance for Grid Dynamics](#2-relevance-for-grid-dynamics)  
3. [Background and terminology](#3-background-and-terminology)  
4. [Objectives and scope](#4-objectives-and-scope)  
5. [System design and methodology](#5-system-design-and-methodology)  
6. [Experimental setup](#6-experimental-setup)  
7. [Results](#7-results)  
8. [Engineering lessons](#8-engineering-lessons)  
9. [Link to soft sensing and industrial estimation](#9-link-to-soft-sensing-and-industrial-estimation)  
10. [Limitations and future work](#10-limitations-and-future-work)  
11. [Conclusions](#11-conclusions)  
12. [Artifacts and reproduction](#12-artifacts-and-reproduction)  
13. [References and further reading](#13-references-and-further-reading)  

---

## 1. Introduction

### 1.1 Motivation

Industrial automation is shifting from fixed, hand-programmed robot trajectories toward **perception-driven, learning-based control**. Clients in manufacturing, logistics, and warehousing increasingly ask for systems that:

- see the scene with cameras,  
- understand optional language or work-order context,  
- act safely and measurably in simulation before hardware,  
- and improve from demonstration data rather than only from handcrafted rewards.

This shift is often called **Physical AI**: artificial intelligence that not only analyzes data, but **senses and acts in the physical world**. It sits at the intersection of computer vision, multimodal foundation models, robotics, digital twins, and MLOps.

At the same time, research and industry practice have produced reusable open stacks — notably **MuJoCo** for physics simulation, **Gymnasium** for environment APIs, and **Hugging Face LeRobot** for datasets and policies such as ACT and SmolVLA. These tools make it possible to assemble a credible end-to-end demo that mirrors how consulting and product-engineering teams (including Grid Dynamics) run Physical AI proof-of-concepts: start in simulation, measure KPIs, refine models, then plan hardware transfer.

### 1.2 Problem statement

The concrete problem addressed by this project is:

> Build a reproducible, closed-loop pipeline in which a simulated bimanual robot completes (or attempts) a cube-transfer task using pretrained imitation / VLA policies, with quantitative success metrics, language-conditioning evidence, and portfolio-grade engineering packaging.

The chosen task is **Aloha TransferCube**: pick a red cube with the right arm and transfer it to the left arm so that the left arm holds the cube off the table. Success is defined by the environment’s sparse reward reaching level **4**.

### 1.3 Why this problem is hard (and interesting)

Several difficulties make the problem representative of real client work:

1. **Embodiment matching.** Pretrained weights only work on the robot and observation layout they were trained for. Using the wrong environment (e.g. FetchReach with Aloha weights) fails silently or obviously.  
2. **Distribution shift.** Small differences between training and inference (e.g. injecting zero-action warm-up after reset) can destroy success even when the “right” checkpoint is loaded.  
3. **Normalization.** Policies expect inputs scaled with dataset statistics; incorrect normalization yields plausible but wrong motions.  
4. **Multimodal evaluation.** Language sensitivity and task success are different claims; both must be measured separately.  
5. **Delivery expectations.** Stakeholders need both a video and a success rate — not a notebook that “sometimes works.”

### 1.4 Contribution of this work

This repository contributes:

| Contribution | Evidence |
|--------------|----------|
| Closed-loop visuomotor control on Aloha TransferCube | ACT rollouts reaching reward 4 |
| Multimodal (vision + language) policy interface | SmolVLA prompt ablation (`language_sensitive: true`) |
| Quantitative multi-seed evaluation | 25/50 = 50% ACT success on CPU |
| Comparative ablation narrative | Side-by-side ACT vs SmolVLA GIF/MP4 |
| Production-minded packaging | CI smoke, release v1.0.0, finetune & SO-100 docs |

### 1.5 Report structure

Section 2 maps the project to Grid Dynamics’ Physical AI and FDE practice. Section 3 defines terms. Sections 4–7 cover objectives, design, experiments, and results. Sections 8–11 discuss engineering lessons, soft-sensor links, limitations, and conclusions.

---

## 2. Relevance for Grid Dynamics

### 2.1 Strategic context

Grid Dynamics publicly invests in:

- **Physical AI / GAIN** — robotic manipulation, inspection, and AI systems that operate on factory and warehouse floors ([Physical AI](https://www.griddynamics.com/physical-ai)),  
- **Digital twins** — virtual replicas used to validate schedules, motion, and AI before deployment ([Digital twin solutions](https://www.griddynamics.com/solutions/digital-twin)),  
- **IoT, edge, and robotics services** — sensing, edge inference, and automation ([IoT & edge](https://www.griddynamics.com/services/iot-and-edge-computing)),  
- **NVIDIA partnerships** — high-fidelity simulation and accelerated AI (Isaac / Omniverse class platforms),  
- **Forward Deployed Engineering (FDE)** — embedding engineers with clients to ship against real constraints, not only lab demos.

This project is not a claim to have built Omniverse-scale twins or a production cell. It is a **portable demonstration of the same delivery loop** Grid Dynamics sells: simulate → measure → refine foundation/imitation models → package for stakeholders → plan hardware.

### 2.2 Capability alignment matrix

| Grid Dynamics theme | Capability shown in this project | Client-facing value |
|---------------------|----------------------------------|---------------------|
| Robotic manipulation solutions | Closed-loop ACT on bimanual TransferCube | PoC of pick/transfer skills in sim |
| Validate before hardware | MuJoCo + multi-seed success metrics | Risk reduction before robot cells |
| Foundation / imitation model refinement | Hub load + SmolVLA finetune playbook | Path from open weights to client demos |
| Instruction-driven / agentic workflows | Language-conditioned SmolVLA interface | Natural-language work orders |
| Production engineering discipline | Typed modules, CI, releases, artifacts | Enterprise-ready delivery habits |
| Digital twin philosophy | Task-level physics twin in MuJoCo | Same idea as Isaac/Omniverse at lighter fidelity |
| FDE ownership model | Env adapter → policy → KPI → docs | Vertical-slice ownership on site |

### 2.3 How the project can be of use on client engagements

**1. Discovery and PoC sprints (1–2 weeks)**  
Many clients need a fast answer: “Can learning-based control handle *this* skill in simulation?” This repo shows a reusable pattern: choose a matched sim task, load a Hub or internal policy, wrap observations, run multi-seed eval, export video + JSON. That pattern transfers to client sims (Isaac, custom Gym envs, or digital twins) even when the physics engine changes.

**2. KPI-driven stakeholder communication**  
Plant managers respond to motion videos; CTOs and quality leads respond to rates. Shipping both `demo_output.mp4` and `eval_results.json` mirrors how FDEs should communicate: visual proof + measured success rate, with honest gaps vs published Hub numbers.

**3. Modular policy platforms (GAIN-style)**  
A stable `predict(observation, text)` interface allows swapping ACT today for a client-finetuned SmolVLA tomorrow without rewriting the control loop. That modularity is essential when Grid Dynamics maintains a platform layer and client-specific adapters.

**4. FDE onboarding and delivery template**  
New engineers can reuse the structure:

- environment wrapper (schema + normalization),  
- policy factory,  
- evaluation harness,  
- CI smoke tests,  
- deployment appendix.

This is how consulting teams industrialize “AI robotics” beyond single-hero notebooks.

**5. Bridge to vision QC and soft sensing**  
The same multimodal and edge-inference skills apply to inspection (defect scores from cameras) and process soft sensors (inferring mixing quality, fill level, or anomaly scores). Physical AI control and soft sensing share perception stacks; this project positions the author on the **perception → estimation → action** continuum.

### 2.4 Positioning for interview / FDE screening

A concise positioning statement:

> This project is a miniature Physical AI delivery: closed-loop manipulation in a digital-twin-style simulation, imitation and VLA policy integration via LeRobot, multi-seed KPI evaluation, language-conditioning ablation, CI packaging, and an honest sim-to-real path. It demonstrates the engineering habits Grid Dynamics needs on GAIN and manufacturing engagements — not only model curiosity, but measurable, explainable, swappable systems.

### 2.5 What this report does *not* claim

Honesty matters for Grid Dynamics clients who have been burned by overpromised robotics demos:

- No claim of Omniverse / Isaac-level digital-twin fidelity.  
- No claim that Aloha ACT zero-shots onto SO-100 or UR arms.  
- No claim that laptop CPU eval matches the Hub’s ~83% GPU reference.  
- No claim that the community SmolVLA checkpoint is production-ready for TransferCube success without finetuning.

These boundaries are strengths: they show mature scoping.

---

## 3. Background and terminology

This section defines terms used throughout the report. Each concept is stated briefly; a longer glossary lives in [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

| Term | Definition |
|------|------------|
| **Physical AI** | AI systems that sense and act in the physical world (robots, factories, warehouses). |
| **VLA (Vision–Language–Action)** | Model that maps images + language instructions to robot actions. |
| **Imitation learning / Behavioral Cloning (BC)** | Learn a policy by imitating expert demonstrations (teleop or scripted). |
| **Policy π** | Decision function `a = π(observation [, language])`. |
| **Observation** | All inputs to the agent (images, vectors, text). |
| **Proprioception** | Robot’s own state (joint positions, gripper openings). |
| **Action space** | Set of valid controls; here a 14-D continuous vector for two arms. |
| **MuJoCo** | Physics engine for contacts, joints, and actuators. |
| **Gymnasium** | Standard Python API (`reset`, `step`) for control environments. |
| **ALOHA / gym-aloha** | Low-cost bimanual teleop platform and its MuJoCo task suite. |
| **TransferCube** | Pick cube with right arm, transfer to left; success at reward 4. |
| **ACT** | Action Chunking with Transformers — imitation policy predicting action sequences. |
| **Action chunking** | Predict multiple future actions per forward pass to reduce compounding error. |
| **SmolVLA** | Compact open VLA (~450M parameters) from Hugging Face. |
| **LeRobot** | HF robotics library for datasets, policies, train/eval/record. |
| **Normalization** | Scaling inputs/outputs with dataset statistics for stable inference. |
| **Distribution shift** | Deploy-time inputs differ from train-time inputs. |
| **Digital twin** | Virtual replica used to test AI/control before hardware. |
| **Sim-to-real** | Transferring policies from simulation to physical robots. |
| **Soft sensor** | Infer hard-to-measure quantities from easy sensors + models. |
| **FDE** | Forward Deployed Engineer — embedded delivery engineer at the client. |

---

## 4. Objectives and scope

### 4.1 Primary objectives

1. Integrate a MuJoCo Aloha environment with a clean observation/action wrapper.  
2. Run a pretrained **ACT** policy that solves TransferCube in closed loop.  
3. Integrate **SmolVLA** and prove language conditioning via prompt ablation.  
4. Evaluate ACT success over many seeds and report a rate, not a single episode.  
5. Package artifacts (GIF/MP4, JSON, CI, docs) suitable for portfolio and client demos.

### 4.2 Non-objectives (out of scope for v1.0.0)

- Full GPU retraining of SmolVLA to Hub-level success (playbook provided; run pending CUDA).  
- Zero-shot transfer to SO-100 hardware.  
- Force/torque or tactile sensing.  
- Multi-task or multi-robot fleets.  
- Omniverse-scale plant digital twins.

### 4.3 Success criteria for the project itself

| Criterion | Status |
|-----------|--------|
| ACT reaches reward 4 on at least one documented seed | Met (seed 36 hero) |
| Multi-seed success rate reported | Met (50%) |
| SmolVLA language sensitivity proven | Met |
| Side-by-side comparison artifact | Met |
| CI smoke + release packaging | Met |
| Finetune + hardware path documented | Met |

---

## 5. System design and methodology

### 5.1 Architecture overview

```
Language instruction (optional)
            │
            ▼
┌───────────────────────────┐
│  Multimodal Policy π      │  ACT or SmolVLA (PyTorch / LeRobot)
│  RGB + proprio [+ text]   │
└─────────────┬─────────────┘
              │ action a_t ∈ R^14
              ▼
┌───────────────────────────┐
│  Gymnasium + gym-aloha    │  AlohaTransferCube-v0
│  MuJoCo physics           │
└─────────────┬─────────────┘
              │ RGB' + state' + reward
              ▼
     GIF / MP4 / eval JSON / ablation JSON
```

The loop is **closed**: each action changes the next image and state. This differs from offline “predict once from a still image” demos.

### 5.2 Simulation and task

- **Environment ID:** `gym_aloha/AlohaTransferCube-v0`  
- **Physics:** MuJoCo  
- **Observation:** top RGB `(480×640×3)` + state `(14,)`  
- **Action:** continuous `(14,)` joint/gripper commands for two arms  
- **Reward stages:**

| Reward | Meaning |
|--------|---------|
| 0 | No useful contact |
| 1 | Right gripper touches cube |
| 2 | Right lifts cube off table |
| 3 | Left gripper contacts cube |
| **4** | **Left holds cube off table → SUCCESS** |

### 5.3 Software modules

| Module | Responsibility |
|--------|----------------|
| `env_wrapper.py` | Gymnasium/Aloha → contiguous tensors, LeRobot batch keys |
| `policy.py` | `MockVLAPolicy`, `LeRobotACTPolicy`, `LeRobotSmolVLAPolicy`, `build_policy` |
| `main.py` | Single-episode rollout + GIF; early stop on success |
| `evaluate.py` | Multi-seed sweep → `eval_results.json` |
| `prompt_ablation.py` | Same obs, two prompts → language sensitivity |
| `compare_policies.py` / `viz.py` | Side-by-side HUD comparison |
| `export_mp4.py` | Portfolio MP4 export |
| `smoke_test.py` + GitHub Actions | CI regression smoke |
| `docs/FINETUNE_SMOLVLA.md` | GPU finetune recipe |
| `docs/DEPLOYMENT_SO100.md` | Real-robot adaptation path |

### 5.4 Policies

**ACT** — `lerobot/act_aloha_sim_transfer_cube_human`  
Vision + proprioception; action chunking; no language required for success. Used as the **task expert** and hero demo backend.

**SmolVLA** — `crislmfroes/smolvla-aloha-sim-transfer-cube-scripted`  
Vision + language + proprioception; tokenized task string. Used to prove the **VLA interface**. Full TransferCube success with this community checkpoint is not reliable; a finetune path on `lerobot/aloha_sim_transfer_cube_human` is documented.

**Mock** — hash-based stub for pipeline smoke tests without Hub downloads.

### 5.5 Critical implementation choices

1. **Matched embodiment.** Policies run only on Aloha TransferCube, not FetchReach.  
2. **No zero-action warm-up after reset.** Warm-up steps after `reset` shifted the state away from the training distribution and eliminated ACT success; the fix is re-render only.  
3. **Explicit ACT normalization.** LeRobot 0.4.x did not auto-load Hub `normalize_*` buffers; they are re-applied from safetensors.  
4. **Success-aware rollouts.** Evaluation can stop when reward == 4 for clean GIFs while still supporting full-horizon sweeps.  
5. **Separate claims.** Task success (ACT) and language sensitivity (SmolVLA) are evaluated with different protocols.

---

## 6. Experimental setup

### 6.1 Hardware / runtime

- **OS:** Windows 10/11  
- **Python:** 3.11 virtual environment (system 3.14 incompatible with LeRobot/pygame stack)  
- **Device for reported eval:** CPU  
- **Invocation pattern:** `.\.venv\Scripts\python.exe <script>` (avoids PowerShell execution-policy issues with `Activate.ps1`)

### 6.2 ACT success evaluation

| Parameter | Value |
|-----------|--------|
| Script | `evaluate.py` |
| Seeds | 0–49 (50 episodes) |
| Step limit | 400 |
| Success | reward == 4 |
| Policy | `lerobot/act_aloha_sim_transfer_cube_human` |

### 6.3 SmolVLA language ablation

| Parameter | Value |
|-----------|--------|
| Script | `prompt_ablation.py` |
| Protocol | Freeze same observation; vary prompt text |
| Metric | L1 difference of first actions; boolean `language_sensitive` |

### 6.4 Comparative visualization

| Parameter | Value |
|-----------|--------|
| Script | `compare_policies.py` |
| Seed | 36 (ACT hero seed) |
| Output | `comparison_act_vs_smolvla.gif` / `.mp4` |

---

## 7. Results

### 7.1 ACT task success

| Metric | Value |
|--------|--------|
| Episodes | 50 (seeds 0–49) |
| Successes | **25** |
| Success rate | **50.0%** |
| Hero seed | **36** (full transfer, reward 4) |
| Hub reference (GPU, ~500 eps) | ~**83%** |

**Interpretation.** Local CPU evaluation demonstrates **reproducible, nontrivial success**, not a single lucky GIF. The gap versus the Hub’s ~83% is expected given device, tooling, and episode-count differences. For portfolio and PoC purposes, a documented 50% with exported artifacts is stronger than an undocumented claim of “it works.”

### 7.2 SmolVLA language conditioning

| Check | Result |
|-------|--------|
| Same observation, two prompts | Different first actions |
| Mean absolute action delta (L1) | ≈ **0.03** |
| `language_sensitive` | **true** |
| Reliable TransferCube success (this ckpt) | **No** |

**Interpretation.** SmolVLA validates the product-facing claim that **language changes behavior**. It does not yet replace ACT as the success baseline. The correct next engineering step is finetuning on the human TransferCube dataset so that language and success coincide.

### 7.3 Ablation narrative (for stakeholders)

| Backend | Solves transfer? | Uses language? | Role |
|---------|------------------|----------------|------|
| ACT | Yes | No | Reliability / KPI / hero video |
| SmolVLA (current ckpt) | Not reliably | Yes | Instruction interface proof |
| Finetuned SmolVLA (planned) | Target yes | Yes | Production-style VLA |

This separation of concerns is deliberate and interview-ready: it shows evaluation maturity rather than conflating two different capabilities.

### 7.4 Engineering packaging results

- Release **v1.0.0** with demo artifacts and technical writeup  
- CI smoke workflow for imports + short ACT/SmolVLA checks  
- MP4 export path for portfolio and LinkedIn / interview sharing  
- Finetune and SO-100 deployment appendices for next-phase work  

---

## 8. Engineering lessons

These lessons are as important as the success rate for Grid Dynamics FDE work:

1. **Train–eval distribution must match.** Zero-action warm-up after reset looked harmless and destroyed ACT performance.  
2. **Normalization is part of the model.** Missing Hub buffers produced wrong actions despite “loading the checkpoint.”  
3. **Embodiment mismatch is a project killer.** Policy, observation keys, and action dimension must be co-designed.  
4. **Language sensitivity ≠ task competence.** Measure both; report both.  
5. **Artifacts beat anecdotes.** GIF/MP4 + JSON + seed list enable review without re-running the full stack.  
6. **CI for robotics is possible.** Even short smoke tests catch import and interface regressions early.

---

## 9. Link to soft sensing and industrial estimation

The author’s broader research theme — **AI-based soft sensors**, deep learning, physics-informed networks, mixing-state estimation, and 3D reconstruction from 2D data — connects directly to this robotics stack:

| Soft-sensor / process AI idea | Analogue in this project |
|-------------------------------|--------------------------|
| Infer hidden process state from cheap sensors | Infer “next action” from RGB + proprioception |
| Physics-informed constraints | MuJoCo contact/joint physics as sim ground truth |
| 2D → spatial understanding | Camera image → 3D cube manipulation |
| Online estimation loop | `obs → π → action → obs'` closed loop |
| Industrial KPI | Success rate / cycle completion (reward 4) |

At Grid Dynamics, this supports a coherent personal brand:

- **Soft sensing** for visibility into processes that lack expensive instruments,  
- **Physical AI** for actuation and manipulation,  
- **Digital twins** for safe validation before plant deployment.

---

## 10. Limitations and future work

| Limitation | Impact | Next step |
|------------|--------|-----------|
| SmolVLA weak on full transfer | Language story without success KPI | GPU finetune (`docs/FINETUNE_SMOLVLA.md`) |
| CPU eval below Hub 83% | Conservative rate | GPU eval; more seeds |
| Simulation only | No hardware risk proof | Follow SO-100 record → finetune → deploy path |
| Single task | Narrow skill coverage | Add insertion / pick-place with matched weights |
| No force/torque | Limited contact-rich skills | Add wrench/tactile for assembly cells |
| Venv-local SmolVLA import patches | Fresh installs may need re-apply | Document install script / pin LeRobot |

---

## 11. Conclusions

This project delivers a complete, honest Physical AI demonstration:

- a matched MuJoCo Aloha simulation,  
- a successful imitation policy (ACT) with multi-seed success measurement,  
- a language-conditioned VLA interface (SmolVLA) with ablation evidence,  
- and engineering packaging suitable for portfolio review and client PoC storytelling.

For **Grid Dynamics**, the relevance is not that the cube transfer itself is a client SKU, but that the **delivery pattern** matches GAIN Physical AI engagements: twin-style validation, imitation/VLA integration, KPI-first communication, modular policy interfaces, and a credible path from simulation toward hardware — without overclaiming.

The work supports positioning as an engineer who can own a vertical slice from environment adapters through multimodal policies to measurable outcomes — the core expectation of Forward Deployed Engineering on industrial AI programs.

---

## 12. Artifacts and reproduction

### 12.1 Key artifacts

| Artifact | Description |
|----------|-------------|
| `demo_output.gif` / `.mp4` | ACT success hero (seed 36) |
| `comparison_act_vs_smolvla.gif` / `.mp4` | Same-seed policy comparison |
| `eval_results.json` | 50-seed ACT evaluation |
| `prompt_ablation.json` | Language sensitivity metrics |
| `TECH_REPORT.md` | Short technical writeup |
| `docs/PROJECT_GUIDE.md` | Concept glossary + Grid Dynamics mapping |
| This document | Full formal report |

### 12.2 Quick reproduce commands

```powershell
.\.venv\Scripts\python.exe main.py --policy act --seed 36
.\.venv\Scripts\python.exe prompt_ablation.py
.\.venv\Scripts\python.exe compare_policies.py --seed 36
.\.venv\Scripts\python.exe evaluate.py --policy act --seeds 0-49 --continue-after-success
.\.venv\Scripts\python.exe export_mp4.py
.\.venv\Scripts\python.exe smoke_test.py
```

### 12.3 30-second oral summary

> I built an end-to-end Physical AI pipeline: MuJoCo Aloha simulation, Hugging Face LeRobot ACT for successful bimanual cube transfer, and SmolVLA to show language-conditioned actions. I evaluated 50 seeds at 50% CPU success, packaged CI and a release, and documented finetune and SO-100 deployment paths. It follows the same simulate → measure → refine loop Grid Dynamics uses in GAIN Physical AI work.

---

## 13. References and further reading

**Project documents**

- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) — extended glossary and interview narrative  
- [TECH_REPORT.md](../TECH_REPORT.md) — concise technical summary  
- [FINETUNE_SMOLVLA.md](FINETUNE_SMOLVLA.md) — GPU finetune playbook  
- [DEPLOYMENT_SO100.md](DEPLOYMENT_SO100.md) — hardware adaptation appendix  

**External**

- Grid Dynamics Physical AI: https://www.griddynamics.com/physical-ai  
- Grid Dynamics Digital Twin: https://www.griddynamics.com/solutions/digital-twin  
- Hugging Face LeRobot: https://huggingface.co/docs/lerobot  
- ACT (Zhao et al., “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware”)  
- SmolVLA / LeRobot VLA documentation on Hugging Face Hub  
- MuJoCo: https://mujoco.org  
- Gymnasium: https://gymnasium.farama.org  

**Hub checkpoints used**

- `lerobot/act_aloha_sim_transfer_cube_human`  
- `crislmfroes/smolvla-aloha-sim-transfer-cube-scripted`  
- Dataset for planned finetune: `lerobot/aloha_sim_transfer_cube_human`  

---

*End of report.*
