"""Streamlit demo dashboard for CPU-friendly Physical AI rollouts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import torch

from env_wrapper import RoboticsEnvWrapper
from main import run_episode, save_frames
from policy import build_policy
from prompt_bank import DEFAULT_TASK, PROMPT_BANK

st.set_page_config(page_title="Robotic Arm VLA Demo", layout="wide")
st.title("Physical AI demo — Aloha TransferCube")
st.caption("CPU-friendly Streamlit UI. Keep steps low on first run.")

with st.sidebar:
    policy_type = st.selectbox("Policy", ["act", "smolvla", "mock"], index=0)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=36, step=1)
    steps = st.slider("Max steps", min_value=5, max_value=400, value=40, step=5)
    prompt_key = st.selectbox("Prompt preset", list(PROMPT_BANK.keys()), index=0)
    prompt = st.text_area("Prompt", value=PROMPT_BANK[prompt_key], height=80)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    run = st.button("Run episode", type="primary")

col1, col2 = st.columns(2)

if run:
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA unavailable — using CPU.")
        device = "cpu"
    with st.spinner("Running closed-loop rollout..."):
        env = RoboticsEnvWrapper(device=device, max_episode_steps=max(int(steps), 400))
        policy = build_policy(
            action_dim=env.action_dim,
            action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
            action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
            image_size=None if policy_type != "mock" else (84, 84),
            device=device,
            policy_type=policy_type,
        )
        try:
            ep = run_episode(
                env,
                policy,
                prompt=prompt or DEFAULT_TASK,
                num_steps=int(steps),
                seed=int(seed),
                deterministic=True,
                stop_on_success=True,
                show_progress=False,
            )
        finally:
            env.close()

        out = Path(tempfile.gettempdir()) / "streamlit_demo.gif"
        save_frames(ep.frames, out, fps=12)

    with col1:
        st.subheader("Rollout")
        st.image(str(out), caption=f"seed={ep.seed}")
    with col2:
        st.subheader("Metrics")
        st.metric("Success", str(ep.success))
        st.metric("Max reward", f"{ep.max_reward:.0f}")
        st.metric("Steps", ep.steps)
        st.json(
            {
                "policy": policy_type,
                "prompt": prompt,
                "seed": ep.seed,
                "success": ep.success,
                "max_reward": ep.max_reward,
                "steps": ep.steps,
            }
        )
else:
    st.info(
        "Configure the sidebar and click **Run episode**. "
        "First ACT/SmolVLA run downloads Hugging Face weights."
    )
    st.markdown(
        "- Full report: `docs/FULL_REPORT.md`\n"
        "- CPU next-level guide: `docs/NEXT_LEVEL_CPU.md`\n"
        "- FigJam: https://www.figma.com/board/CmvFbnixCtXsehlEUMbEnZ"
    )
