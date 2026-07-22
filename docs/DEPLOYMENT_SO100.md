# Real-Robot Deployment Appendix (SO-100 / LeRobot)

This project’s **primary artifact is sim** (MuJoCo Aloha + Hub policies). The same LeRobot stack is designed to continue onto low-cost hardware — typically the **SO-100 / SO-101** arms used in Hugging Face community datasets.

---

## 1. Sim → real bridge

| Stage | This repo (sim) | Hardware path |
|-------|-----------------|---------------|
| Observations | `observation.images.top` + `observation.state` | Wrist/front cameras + joint state from robot bus |
| Policy | ACT / SmolVLA `select_action` | Same `PreTrainedPolicy` API |
| Actions | 14-D Aloha (bimanual) or adapted dim | Calibrated joint / leader–follower commands |
| Safety | MuJoCo limits | Torque/velocity clamps, e-stop, workspace box |

**Important:** Aloha TransferCube ACT weights are **not** drop-in for a single SO-100 arm (different DoF / cameras). Real deployment means either:

1. Finetune SmolVLA / ACT on **your** SO-100 dataset (`lerobot-record` → Hub → `lerobot-train`), or  
2. Use a Hub policy already trained for SO-100 (e.g. community `svla_so100_*` datasets).

---

## 2. Hardware checklist

- SO-100 or SO-101 arm + motors powered and IDed  
- Cameras calibrated and reachable from LeRobot robot config (`cameras={ front: {...} }`)  
- Host PC: Ubuntu preferred; CUDA optional for SmolVLA  
- Mechanical: clear workspace, soft gripper pads, e-stop tested  

Docs / hardware:

- https://github.com/TheRobotStudio/SO-ARM100  
- https://huggingface.co/docs/lerobot  

---

## 3. Record → train → deploy loop

```bash
# 1) Record demos (example shape — adjust ports / camera ids)
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_so100 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }' \
  --dataset.repo_id=${HF_USER}/so100_cube_demo \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the red cube and place it in the bin"

# 2) Finetune SmolVLA (see docs/FINETUNE_SMOLVLA.md for TransferCube sim analogue)
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/so100_cube_demo \
  --output_dir=outputs/train/smolvla_so100 \
  --steps=20000 \
  --policy.device=cuda

# 3) Deploy / teleop hybrid
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --policy.path=${HF_USER}/smolvla_so100_cube \
  --dataset.repo_id=${HF_USER}/so100_eval_rollouts \
  --dataset.single_task="Pick up the red cube and place it in the bin"
```

Exact CLI flags evolve with LeRobot versions — always run `lerobot-record --help` / `lerobot-train --help` on your installed package.

---

## 4. Mapping from this repository

Reuse without change:

- `policy.py` prediction interface (`predict(observation, text)`)  
- Eval mindset: sparse success metrics + prompt ablation  
- Packaging: typed wrappers, GIF/MP4 artifacts, CI smoke  

Replace for hardware:

- `env_wrapper.py` → LeRobot robot observation adapters  
- `gym_aloha` → physical robot + camera topics  
- Action dim / normalization stats → dataset `meta/stats`  

---

## 5. Safety & ops (non-negotiable)

1. Velocity / torque limits in robot config before enabling policy actions  
2. Manual override / leader arm for the first eval episodes  
3. Never run untested Hub checkpoints near people without a cage / e-stop  
4. Log episodes (`lerobot-record`) so failures are auditable  

---

## 6. Portfolio narrative

For Grid Dynamics / Physical AI interviews, the honest story is:

> We validated closed-loop imitation + VLA language conditioning in **MuJoCo**, measured success rates, and documented the **identical LeRobot path** to SO-100 recording, finetuning, and on-robot rollout — without claiming zero-shot transfer of Aloha ACT weights to a different embodiment.

That is the professional sim-to-real stance.
