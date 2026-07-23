"""Streamlit demo: ACT for cube transfer, SmolVLA for language ablation."""

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

mode = st.radio(
    "What do you want to demo?",
    [
        "Pick the red cube (ACT — ignores prompt)",
        "Change action with language (SmolVLA)",
    ],
    index=0,
)
is_language_mode = mode.startswith("Change")

with st.sidebar:
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=36, step=1)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)

    if is_language_mode:
        st.markdown("### Prompts (SmolVLA)")
        st.caption("Same camera + joints; only the text changes.")
        keys = list(PROMPT_BANK.keys())
        key_a = st.selectbox("Prompt A", keys, index=keys.index("canonical"))
        key_b = st.selectbox("Prompt B", keys, index=keys.index("idle"))
        if st.session_state.get("_ka") != key_a:
            st.session_state["_ka"] = key_a
            st.session_state["text_a"] = PROMPT_BANK[key_a]
        if st.session_state.get("_kb") != key_b:
            st.session_state["_kb"] = key_b
            st.session_state["text_b"] = PROMPT_BANK[key_b]
        prompt_a = st.text_area("Text A", key="text_a", height=80)
        prompt_b = st.text_area("Text B", key="text_b", height=80)
        steps = st.slider("Optional GIF steps", 5, 80, 20, 5)
        policy_type = "smolvla"
    else:
        st.markdown("### ACT rollout")
        st.caption("Prompt is ignored on purpose — ACT has no language input.")
        prompt_a = DEFAULT_TASK
        prompt_b = DEFAULT_TASK
        steps = st.slider("Max steps", 50, 400, 300, 10)
        policy_type = "act"

    run = st.button("Run", type="primary")

if not is_language_mode:
    st.warning(
        "You are in **ACT** mode. Editing any prompt will **not** change the motion. "
        "Switch the radio to **Change action with language (SmolVLA)**."
    )
else:
    st.success(
        "SmolVLA mode: we always compare Prompt A vs B on the **same** first frame "
        "and report L1 action delta. Do not judge language from the GIF alone."
    )

if run:
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA unavailable — using CPU.")
        device = "cpu"

    prompt_a = (prompt_a or DEFAULT_TASK).strip()
    prompt_b = (prompt_b or DEFAULT_TASK).strip()

    with st.spinner("Running..."):
        env = RoboticsEnvWrapper(device=device, max_episode_steps=max(int(steps), 400))
        policy = build_policy(
            action_dim=env.action_dim,
            action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
            action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
            image_size=None,
            device=device,
            policy_type=policy_type,
        )
        try:
            obs, _ = env.reset(seed=int(seed))
            policy.reset()
            action_a = policy.predict(obs, prompt_a)
            policy.reset()
            action_b = policy.predict(obs, prompt_b)
            delta = action_a - action_b
            l1 = float(delta.abs().mean().item())
            sensitive = l1 > 1e-4

            policy.reset()
            ep = run_episode(
                env,
                policy,
                prompt=prompt_a,
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

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rollout GIF (Prompt A)")
        st.image(str(out), caption=f"{policy_type} | seed={ep.seed}")
        st.metric("Success", str(ep.success))
        st.metric("Max reward", f"{ep.max_reward:.0f}")
    with c2:
        st.subheader("First-action prompt test")
        st.write(f"**A:** {prompt_a}")
        st.write(f"**B:** {prompt_b}")
        st.metric("L1 |aA - aB|", f"{l1:.5f}")
        st.metric("Actions differ?", str(sensitive))
        if policy_type == "act":
            st.error("ACT ignores text — L1 should be ~0. Switch to SmolVLA mode.")
        elif sensitive:
            st.success("Language changed the action (this is the proof).")
        else:
            st.error("No difference — SmolVLA may have failed to load.")
        st.json(
            {
                "policy": policy_type,
                "l1_delta": l1,
                "language_sensitive": sensitive,
                "action_a_first_4": action_a.detach().cpu().tolist()[:4],
                "action_b_first_4": action_b.detach().cpu().tolist()[:4],
            }
        )
else:
    st.markdown(
        """
**Pick cube:** choose ACT mode → seed 36 → steps ~300 → Run → watch GIF / success.

**Change action with prompt:** choose SmolVLA mode → Prompt A=`canonical`, Prompt B=`idle` → Run  
→ look at **L1 |aA - aB|** (should be > 0). The GIF may still look similar.
"""
    )
