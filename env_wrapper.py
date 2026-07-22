"""Gymnasium / gym-aloha environment wrappers for VLA-style observation pipelines.

Supports:
* Gymnasium-Robotics FetchReach (MuJoCo) for mock / prototyping rollouts
* gym-aloha ``AlohaTransferCube-v0`` for pretrained Hugging Face LeRobot ACT
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# Optional robotics stack registration (Fetch, etc.)
try:
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
except ImportError:  # pragma: no cover
    gymnasium_robotics = None  # type: ignore[assignment]

# Aloha bimanual MuJoCo tasks used by LeRobot pretrained checkpoints
try:
    import gym_aloha  # noqa: F401
except ImportError:  # pragma: no cover
    gym_aloha = None  # type: ignore[assignment]


class RoboticsEnvWrapper:
    """OO wrapper around MuJoCo manipulation envs with RGB + state tensors.

    Responsibilities
    ----------------
    * Construct Fetch or Aloha envs with ``render_mode="rgb_array"``.
    * Reset / step while always returning an RGB frame for GIF capture.
    * Convert HWC uint8 frames into NCHW float32 tensors for PyTorch policies.
    * Expose proprioceptive state when available (Fetch goals / Aloha agent_pos).

    Parameters
    ----------
    env_id:
        Gymnasium env id. Defaults to Aloha transfer cube (LeRobot ACT target).
    image_size:
        Spatial resize target ``(H, W)`` for policy RGB tensors.
        Use ``None`` to keep the native camera resolution (required for ACT).
    device:
        Torch device string for observation tensors (``"cpu"`` or ``"cuda"``).
    max_episode_steps:
        Optional override for the env's time limit.
    obs_type:
        Aloha-only observation mode. ``pixels_agent_pos`` matches LeRobot ACT.
    """

    DEFAULT_ENV_ID: str = "gym_aloha/AlohaTransferCube-v0"
    FETCH_ENV_ID: str = "FetchReachDense-v4"

    def __init__(
        self,
        env_id: str = DEFAULT_ENV_ID,
        image_size: tuple[int, int] | None = None,
        device: str = "cpu",
        max_episode_steps: int | None = None,
        obs_type: str = "pixels_agent_pos",
    ) -> None:
        self.env_id = env_id
        self.image_size = image_size  # (H, W) or None = native
        self.device = torch.device(device)
        self.obs_type = obs_type
        self.is_aloha = env_id.startswith("gym_aloha/")

        kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
        if self.is_aloha:
            if gym_aloha is None:
                raise ImportError(
                    "gym-aloha is required for Aloha envs. "
                    "Install with: pip install gym-aloha"
                )
            kwargs["obs_type"] = obs_type
        if max_episode_steps is not None:
            kwargs["max_episode_steps"] = int(max_episode_steps)

        self.env: gym.Env = gym.make(env_id, **kwargs)

        self.action_space: spaces.Box = self._as_box(self.env.action_space)
        self.action_dim: int = int(np.prod(self.action_space.shape))
        self._last_rgb: np.ndarray | None = None  # shape: (H_raw, W_raw, 3)

    @staticmethod
    def _as_box(space: spaces.Space) -> spaces.Box:
        """Ensure the action space is a continuous Box."""
        if not isinstance(space, spaces.Box):
            raise TypeError(
                f"Expected continuous Box action space, got {type(space).__name__}"
            )
        return space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the simulator and return the first observation + info.

        Returns
        -------
        obs:
            Dict containing:
            - ``"rgb"``: uint8 array, shape ``(H_raw, W_raw, 3)``
            - ``"rgb_tensor"``: float32 tensor, shape ``(1, 3, H, W)`` in [0, 1]
            - ``"vector"`` / ``"vector_tensor"``: proprio / goal state if present
            - ``"raw"``: original Gymnasium observation
            - ``"lerobot"``: LeRobot-format batch (Aloha) for ACT select_action
        info:
            Environment info dict from ``env.reset``.
        """
        raw_obs, info = self.env.reset(seed=seed, options=options)
        rgb = self.render_rgb()
        # Aloha can return a black first frame; re-render only (do not step —
        # zero-actions would desync the episode from ACT training distribution).
        if self.is_aloha and float(np.mean(rgb)) < 1.0:
            for _ in range(5):
                rgb = self.render_rgb()
                if float(np.mean(rgb)) >= 1.0:
                    break
        return self._format_observation(raw_obs, rgb), info

    def step(
        self, action: np.ndarray | torch.Tensor
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply an action, advance physics one step, and return formatted obs."""
        action_np = self._to_numpy_action(action)
        raw_obs, reward, terminated, truncated, info = self.env.step(action_np)
        rgb = self.render_rgb()
        return (
            self._format_observation(raw_obs, rgb),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render_rgb(self) -> np.ndarray:
        """Capture the current MuJoCo RGB frame from the env camera.

        Returns
        -------
        rgb:
            uint8 image array, shape ``(H_raw, W_raw, 3)``, channel order RGB.
        """
        frame = self.env.render()
        if frame is None:
            # Fall back to pixels in the observation if render is unavailable.
            raise RuntimeError(
                f"env.render() returned None — ensure render_mode='rgb_array' "
                f"for '{self.env_id}'."
            )
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame (H, W, 3), got shape {rgb.shape}")
        rgb = np.ascontiguousarray(rgb)
        self._last_rgb = rgb
        return rgb  # shape: (H_raw, W_raw, 3)

    def get_last_rgb(self) -> np.ndarray:
        """Return the most recently captured RGB frame without re-rendering."""
        if self._last_rgb is None:
            return self.render_rgb()
        return self._last_rgb

    def observation_to_tensor(
        self,
        rgb: np.ndarray,
        *,
        batch: bool = True,
    ) -> torch.Tensor:
        """Convert an HWC uint8 RGB frame to an NCHW float tensor in [0, 1].

        Parameters
        ----------
        rgb:
            uint8 array, shape ``(H_raw, W_raw, 3)``.
        batch:
            If True, prepend a batch dimension → ``(1, 3, H, W)``.

        Returns
        -------
        tensor:
            float32 tensor on ``self.device``, values in ``[0, 1]``.
        """
        hwc = torch.from_numpy(rgb).float() / 255.0  # shape: (H_raw, W_raw, 3)
        chw = hwc.permute(2, 0, 1).unsqueeze(0)  # shape: (1, 3, H_raw, W_raw)
        if self.image_size is not None:
            target_h, target_w = self.image_size
            if chw.shape[-2:] != (target_h, target_w):
                chw = torch.nn.functional.interpolate(
                    chw,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )  # shape: (1, 3, H, W)
        if not batch:
            chw = chw.squeeze(0)  # shape: (3, H, W)
        return chw.to(self.device)

    def _extract_top_rgb(self, raw_obs: Any, fallback_rgb: np.ndarray) -> np.ndarray:
        """Prefer Aloha ``pixels['top']`` when present; else use render frame."""
        if isinstance(raw_obs, dict):
            pixels = raw_obs.get("pixels")
            if isinstance(pixels, dict) and "top" in pixels:
                return np.ascontiguousarray(np.asarray(pixels["top"], dtype=np.uint8))
            if isinstance(pixels, np.ndarray) and pixels.ndim == 3:
                return np.ascontiguousarray(pixels.astype(np.uint8))
        return fallback_rgb

    def _format_observation(
        self,
        raw_obs: Any,
        rgb: np.ndarray,
    ) -> dict[str, Any]:
        """Package raw env obs + RGB into a policy-ready dictionary."""
        cam_rgb = self._extract_top_rgb(raw_obs, rgb)
        formatted: dict[str, Any] = {
            "rgb": cam_rgb,  # shape: (H_raw, W_raw, 3), uint8
            "rgb_tensor": self.observation_to_tensor(cam_rgb, batch=True),  # (1, 3, H, W)
            "raw": raw_obs,
        }

        vector: np.ndarray | None = None
        if isinstance(raw_obs, dict):
            if "agent_pos" in raw_obs:
                vector = np.asarray(raw_obs["agent_pos"], dtype=np.float32).ravel()
            else:
                vector_parts: list[np.ndarray] = []
                for key in ("observation", "achieved_goal", "desired_goal"):
                    if key in raw_obs:
                        vector_parts.append(
                            np.asarray(raw_obs[key], dtype=np.float32).ravel()
                        )
                if vector_parts:
                    vector = np.concatenate(vector_parts, axis=0)  # shape: (D,)
        elif isinstance(raw_obs, np.ndarray):
            vector = np.asarray(raw_obs, dtype=np.float32).ravel()

        if vector is not None:
            formatted["vector"] = vector
            formatted["vector_tensor"] = (
                torch.from_numpy(vector).float().unsqueeze(0).to(self.device)
            )  # shape: (1, D)

        # LeRobot batch used by ACTPolicy.select_action (Aloha path).
        if self.is_aloha and isinstance(raw_obs, dict) and "agent_pos" in raw_obs:
            from lerobot.envs.utils import preprocess_observation

            formatted["lerobot"] = preprocess_observation(raw_obs)

        return formatted

    def _to_numpy_action(self, action: np.ndarray | torch.Tensor) -> np.ndarray:
        """Normalize policy output to a 1-D float32 numpy action within bounds."""
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)

        action_np = action_np.astype(np.float32).reshape(-1)
        if action_np.shape[0] != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: got {action_np.shape[0]}, "
                f"expected {self.action_dim}"
            )

        low = self.action_space.low.astype(np.float32).reshape(-1)
        high = self.action_space.high.astype(np.float32).reshape(-1)
        return np.clip(action_np, low, high)

    def close(self) -> None:
        """Release MuJoCo / Gymnasium resources."""
        self.env.close()

    def __enter__(self) -> RoboticsEnvWrapper:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"RoboticsEnvWrapper(env_id={self.env_id!r}, "
            f"action_dim={self.action_dim}, image_size={self.image_size}, "
            f"device={self.device})"
        )
