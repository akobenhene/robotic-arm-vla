"""Streamlit demo with append-only run logging (outputs/run_log.jsonl)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import torch

from env_wrapper import RoboticsEnvWrapper
from main import run_episode, save_frames
from policy import build_policy
from prompt_bank import DEFAULT_TASK, PROMPT_BANK
from run_logger import DEFAULT_LOG_PATH, RunLogger

st.set_page_config(page_title="Robotic Arm VLA Demo", layout="wide")
st.title("Physical AI demo — Aloha TransferCube")
st.caption(f"Every Run appends to `{DEFAULT_LOG_PATH}` (open that file to verify).")

mode = st.radio(
    "What do you want to demo?",
    [
        "Pick the red cube (ACT — ignores prompt)",
        "Change action with language (SmolVLA)",
    ],
    index=1,
)
is_language_mode = mode.startswith("Change")

with st.sidebar:
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=36, step=1)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    log_path = st.text_input("Log file", value=str(DEFAULT_LOG_PATH))

    if is_language_mode:
        st.markdown("### Prompts (SmolVLA)")
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
        st.caption("Prompt is ignored — ACT has no language input.")
        prompt_a = DEFAULT_TASK
        prompt_b = "THIS_PROMPT_IS_IGNORED_BY_ACT"
        steps = st.slider("Max steps", 50, 400, 300, 10)
        policy_type = "act"

    run = st.button("Run", type="primary")

if not is_language_mode:
    st.warning(
        "ACT mode: prompts do **not** change actions. "
        "Switch to **Change action with language (SmolVLA)**."
    )
else:
    st.success(
        "SmolVLA mode: log will record Prompt A/B, both action vectors, and L1 delta."
    )

if run:
    logger = RunLogger(log_path)
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA unavailable — using CPU.")
        device = "cpu"

    prompt_a = (prompt_a or DEFAULT_TASK).strip()
    prompt_b = (prompt_b or DEFAULT_TASK).strip()

    logger.log_config(
        ui_mode=mode,
        policy=policy_type,
        device=device,
        seed=int(seed),
        steps=int(steps),
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        log_path=str(Path(log_path).resolve()),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
    )

    with st.spinner("Running (also writing log)..."):
        env = RoboticsEnvWrapper(device=device, max_episode_steps=max(int(steps), 400))
        policy = build_policy(
            action_dim=env.action_dim,
            action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
            action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
            image_size=None,
            device=device,
            policy_type=policy_type,
        )
        logger.log(
            "policy_loaded",
            policy=policy_type,
            repo_id=getattr(policy, "repo_id", None),
            class_name=type(policy).__name__,
        )
        try:
            obs, _ = env.reset(seed=int(seed))
            logger.log(
                "env_reset",
                seed=int(seed),
                rgb_shape=list(obs["rgb"].shape),
                state_dim=int(obs["vector"].shape[0]) if "vector" in obs else None,
            )

            policy.reset()
            action_a = policy.predict(obs, prompt_a)
            logger.log(
                "predict",
                which="A",
                policy=policy_type,
                prompt=prompt_a,
                task_sent_to_model=getattr(policy, "last_task", None),
                action=action_a,
                task_key_used_by_smolvla=policy_type == "smolvla",
            )

            policy.reset()
            action_b = policy.predict(obs, prompt_b)
            logger.log(
                "predict",
                which="B",
                policy=policy_type,
                prompt=prompt_b,
                task_sent_to_model=getattr(policy, "last_task", None),
                action=action_b,
                task_key_used_by_smolvla=policy_type == "smolvla",
            )

            ablation = logger.log_prompt_ablation(
                policy=policy_type,
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                action_a=action_a,
                action_b=action_b,
                seed=int(seed),
            )

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
            logger.log_episode(
                policy=policy_type,
                prompt=prompt_a,
                seed=ep.seed,
                steps=ep.steps,
                max_reward=ep.max_reward,
                success=ep.success,
                first_action=action_a,
            )
        finally:
            env.close()
            logger.log("env_closed")

        out = Path(tempfile.gettempdir()) / "streamlit_demo.gif"
        gif = save_frames(ep.frames, out, fps=12)
        logger.log("gif_saved", path=str(gif))

    resolved_log = str(Path(log_path).resolve())
    st.code(resolved_log, language=None)
    st.download_button(
        "Download run_log.jsonl",
        data=Path(log_path).read_text(encoding="utf-8"),
        file_name="run_log.jsonl",
        mime="application/jsonl",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rollout GIF (Prompt A)")
        st.image(str(out), caption=f"{policy_type} | seed={ep.seed}")
        st.metric("Success", str(ep.success))
        st.metric("Max reward", f"{ep.max_reward:.0f}")
    with c2:
        st.subheader("Logged prompt test")
        st.write(f"**A:** {prompt_a}")
        st.write(f"**B:** {prompt_b}")
        l1 = float(ablation["l1_delta"])
        st.metric("L1 |aA - aB|", f"{l1:.5f}")
        st.metric("language_sensitive", str(ablation["language_sensitive"]))
        if policy_type == "act":
            st.error("ACT ignores text — expect L1 ~ 0. Use SmolVLA mode.")
        elif ablation["language_sensitive"]:
            st.success("Log confirms actions differ. Open run_log.jsonl for full vectors.")
        else:
            st.error("Log shows no action difference — paste the latest log lines here.")
        st.json(
            {
                "run_id": logger.run_id,
                "log_file": resolved_log,
                "policy": policy_type,
                "l1_delta": l1,
                "language_sensitive": ablation["language_sensitive"],
                "action_a_first_4": action_a.detach().cpu().tolist()[:4],
                "action_b_first_4": action_b.detach().cpu().tolist()[:4],
            }
        )

    with st.expander("Latest log lines"):
        lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
        st.code("\n".join(lines[-8:]), language="json")
else:
    st.markdown(
        f"""
1. Choose **Change action with language (SmolVLA)** (default now).  
2. Prompt A = `canonical`, Prompt B = `idle`.  
3. Click **Run**.  
4. Open **`{DEFAULT_LOG_PATH}`** — look for `"event": "prompt_ablation"` and `"language_sensitive": true`.

If you stay in ACT mode, the log will show `"language_sensitive": false` / L1 ~ 0 — that is correct.
"""
    )
