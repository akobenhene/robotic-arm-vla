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
st.caption(
    "Language changes actions only with **SmolVLA**. "
    "ACT ignores the prompt (vision + state only)."
)

with st.sidebar:
    policy_type = st.selectbox(
        "Policy",
        ["smolvla", "act", "mock"],
        index=0,
        help="Use SmolVLA to see prompt effects. ACT will look identical for any text.",
    )
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=36, step=1)
    steps = st.slider("Max steps", min_value=5, max_value=400, value=40, step=5)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)

    st.markdown("### Language prompt")
    prompt_key = st.selectbox("Prompt preset", list(PROMPT_BANK.keys()), index=0)
    # Keep text area in sync when the preset changes (Streamlit session-state gotcha).
    if st.session_state.get("_prompt_key") != prompt_key:
        st.session_state["_prompt_key"] = prompt_key
        st.session_state["prompt_text"] = PROMPT_BANK[prompt_key]
    prompt = st.text_area("Prompt", key="prompt_text", height=100)

    compare = st.checkbox(
        "Compare two prompts (first-action delta)",
        value=policy_type == "smolvla",
        help="Frozen observation, two prompts — proves language sensitivity.",
    )
    prompt_b_key = st.selectbox(
        "Second prompt (compare)",
        list(PROMPT_BANK.keys()),
        index=list(PROMPT_BANK.keys()).index("idle"),
        disabled=not compare,
    )
    run = st.button("Run", type="primary")

if policy_type == "act":
    st.warning(
        "ACT does **not** use language. Changing the prompt will not change the motion. "
        "Switch policy to **smolvla** to test prompts."
    )

col1, col2 = st.columns(2)

if run:
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA unavailable — using CPU.")
        device = "cpu"

    prompt_a = (prompt or DEFAULT_TASK).strip()
    prompt_b = PROMPT_BANK[prompt_b_key]

    with st.spinner("Loading policy / running rollout..."):
        env = RoboticsEnvWrapper(device=device, max_episode_steps=max(int(steps), 400))
        policy = build_policy(
            action_dim=env.action_dim,
            action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
            action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
            image_size=None if policy_type != "mock" else (84, 84),
            device=device,
            policy_type=policy_type,
        )

        ablation = None
        try:
            if compare:
                obs, _ = env.reset(seed=int(seed))
                policy.reset()
                action_a = policy.predict(obs, prompt_a)
                policy.reset()
                action_b = policy.predict(obs, prompt_b)
                delta = action_a - action_b
                ablation = {
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "l1_delta": float(delta.abs().mean().item()),
                    "l2_delta": float(torch.linalg.vector_norm(delta).item()),
                    "language_sensitive": bool(delta.abs().mean().item() > 1e-4),
                    "action_a": action_a.detach().cpu().tolist(),
                    "action_b": action_b.detach().cpu().tolist(),
                }

            if hasattr(policy, "reset"):
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

    with col1:
        st.subheader("Rollout GIF")
        st.image(str(out), caption=f"policy={policy_type} | seed={ep.seed}")
        st.caption(
            "Full GIFs can look similar even when first actions differ — "
            "use the comparison panel for a clear language check."
        )
    with col2:
        st.subheader("Metrics")
        st.metric("Success", str(ep.success))
        st.metric("Max reward", f"{ep.max_reward:.0f}")
        st.metric("Steps", ep.steps)
        st.json(
            {
                "policy": policy_type,
                "uses_language": policy_type == "smolvla",
                "prompt": prompt_a,
                "seed": ep.seed,
                "success": ep.success,
                "max_reward": ep.max_reward,
                "steps": ep.steps,
            }
        )
        if ablation is not None:
            st.subheader("Prompt comparison (same image/state)")
            st.metric("L1 action delta", f"{ablation['l1_delta']:.5f}")
            st.metric("Language sensitive", str(ablation["language_sensitive"]))
            if policy_type != "smolvla":
                st.info("With ACT/Mock, L1 delta is expected near 0.")
            elif ablation["language_sensitive"]:
                st.success("SmolVLA changed the action when the prompt changed.")
            else:
                st.error("No action change detected — check SmolVLA load path.")
            st.json(ablation)
else:
    st.info(
        "1. Set policy to **smolvla**\n"
        "2. Pick two different presets (e.g. `canonical` vs `idle`)\n"
        "3. Enable **Compare two prompts** and click Run\n\n"
        "Expect L1 delta > 0 (often ~0.01–0.05), not a totally different GIF."
    )
