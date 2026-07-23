# Higher-fidelity digital twin path (NVIDIA / Isaac) — scaffold

This project uses **MuJoCo + gym-aloha** as a lightweight task twin. Grid Dynamics client work often targets **NVIDIA Isaac Sim / Omniverse**.

## What you can do without a local GPU workstation

1. Keep the **same software contracts**:
   - observation keys (`images`, `state`)
   - action dim / normalize stats
   - `predict(obs, text)` policy interface
   - multi-seed success KPI JSON
2. Document the twin upgrade as a **port**, not a rewrite:
   - replace `RoboticsEnvWrapper` backend with Isaac env adapter
   - keep `policy.py` + `evaluate.py` unchanged
3. Use cloud Isaac / NGC when available (not required for this repo's v1 demos).

## Mapping

| This repo | Isaac-oriented client stack |
|-----------|-----------------------------|
| MuJoCo Aloha | Isaac Manipulator / custom USD scene |
| gymnasium `step/reset` | Isaac gym / gymnasium wrapper |
| LeRobot Hub policies | Same PyTorch policies via ONNX/TensorRT later |
| Streamlit demo | Client dashboard / GAIN UI |

## Non-goals until GPU workstation exists

- Full Omniverse plant model  
- RTX-rendered synthetic data at scale  
- TensorRT deployment  

CPU progress (Streamlit, eval KPIs, soft sensor, prompt bank) still transfers 1:1 into an Isaac adapter later.
