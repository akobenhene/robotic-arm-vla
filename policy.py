"""Multimodal Vision-Language-Action (VLA) / imitation policy modules.

Production paths
----------------
* ``act`` — Hugging Face LeRobot ACT on AlohaTransferCube (task success).
* ``smolvla`` — language-conditioned SmolVLA finetuned for Aloha transfer cube.
* ``mock`` — lightweight CNN+text hash for FetchReach prototyping.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

PolicyKind = Literal["act", "smolvla", "mock"]


class VLAPolicyProtocol(Protocol):
    """Structural interface shared by mock and production policies."""

    def predict(
        self,
        observation: dict[str, Any],
        text: str = "",
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Map an observation (+ optional language) to a continuous action."""
        ...

    def reset(self) -> None:
        """Clear temporal state (action queues / ensembles) on env reset."""
        ...


class MockVLAPolicy(nn.Module):
    """Lightweight mock VLA for FetchReach-style pipeline demos.

    Not a trained imitation policy. Kept as ``--mock-policy`` fallback.
    """

    def __init__(
        self,
        action_dim: int,
        image_size: tuple[int, int] = (84, 84),
        vision_dim: int = 128,
        text_dim: int = 64,
        action_low: float | torch.Tensor = -1.0,
        action_high: float | torch.Tensor = 1.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.image_size = image_size
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        # Input expected shape: (B, 3, H, W)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, vision_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, vision_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vision_dim, vision_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(vision_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        low = torch.as_tensor(action_low, dtype=torch.float32).view(-1)
        high = torch.as_tensor(action_high, dtype=torch.float32).view(-1)
        if low.numel() == 1:
            low = low.expand(action_dim)
        if high.numel() == 1:
            high = high.expand(action_dim)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reset(self) -> None:
        """No temporal state for the mock policy."""
        return None

    @staticmethod
    def encode_text(text: str, text_dim: int, device: torch.device) -> torch.Tensor:
        """Deterministic mock text encoder (hash → embedding)."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        block = digest
        while len(values) < text_dim:
            for byte in block:
                u = (byte + 0.5) / 256.0
                values.append(float((u - 0.5) * 2.0 * 1.732))
                if len(values) >= text_dim:
                    break
            block = hashlib.sha256(block).digest()
        emb = torch.tensor(values[:text_dim], dtype=torch.float32, device=device)
        return emb.unsqueeze(0)  # shape: (1, text_dim)

    def forward(self, image: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Forward: image ``(B, 3, H, W)`` + text ``(B, text_dim)`` → ``(B, A)``."""
        if image.ndim != 4:
            raise ValueError(f"Expected image (B, 3, H, W), got {tuple(image.shape)}")
        if text_emb.ndim == 1:
            text_emb = text_emb.unsqueeze(0)
        batch = image.shape[0]
        if text_emb.shape[0] == 1 and batch > 1:
            text_emb = text_emb.expand(batch, -1)

        vision_feat = self.vision_encoder(image)  # (B, vision_dim)
        language_feat = self.text_proj(text_emb)  # (B, vision_dim)
        fused = torch.cat([vision_feat, language_feat], dim=-1)
        normalized = self.action_head(fused)  # (B, action_dim) in [-1, 1]
        mid = (self.action_high + self.action_low) * 0.5
        half = (self.action_high - self.action_low) * 0.5
        return mid + half * normalized

    @torch.inference_mode()
    def predict(
        self,
        observation: dict[str, Any],
        text: str = "",
        *,
        deterministic: bool = False,
        exploration_std: float = 0.05,
    ) -> torch.Tensor:
        """Inference from formatted env observation + optional text."""
        image = observation["rgb_tensor"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(next(self.parameters()).device)
        text_emb = self.encode_text(text, self.text_dim, image.device)
        action = self.forward(image, text_emb)
        if not deterministic and exploration_std > 0.0:
            action = action + torch.randn_like(action) * exploration_std
            action = torch.max(torch.min(action, self.action_high), self.action_low)
        if action.shape[0] == 1:
            return action.squeeze(0)
        return action


class LeRobotACTPolicy(nn.Module):
    """Production wrapper around Hugging Face LeRobot ``ACTPolicy``.

    Loads ``lerobot/act_aloha_sim_transfer_cube_human`` (Action Chunking Transformer)
    trained on gym-aloha ``AlohaTransferCube-v0``. Expected inputs:

    * ``observation.images.top`` — shape ``(B, 3, 480, 640)``, float in [0, 1]
    * ``observation.state`` — shape ``(B, 14)`` joint / gripper proprioception
    * Output action — shape ``(B, 14)`` (or ``(14,)`` after squeeze)

    Modern LeRobot moved dataset normalization out of the module weights. Older Hub
    checkpoints still store ``normalize_*`` / ``unnormalize_*`` buffers in
    ``model.safetensors``; this adapter re-applies them so inference matches
    training-time statistics.

    Parameters
    ----------
    repo_id:
        Hugging Face Hub repo id for the ACT checkpoint.
    device:
        Torch device string.
    """

    DEFAULT_REPO_ID: str = "lerobot/act_aloha_sim_transfer_cube_human"

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.device_str = device
        self.device = torch.device(device)

        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "lerobot is required. Install with: pip install lerobot"
            ) from exc

        # --- Real LeRobot weight loading ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._policy = ACTPolicy.from_pretrained(repo_id)
        self._policy.to(self.device)
        self._policy.eval()

        self._stats = self._load_norm_stats(repo_id)
        self.action_dim = int(self._policy.config.output_features["action"].shape[0])

    @staticmethod
    def _load_norm_stats(repo_id: str) -> dict[str, torch.Tensor]:
        """Load legacy normalize / unnormalize buffers from the Hub safetensors."""
        path = hf_hub_download(repo_id, "model.safetensors")
        stats: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                if "normalize" in key or "unnormalize" in key:
                    stats[key] = handle.get_tensor(key)
        required = (
            "normalize_inputs.buffer_observation_images_top.mean",
            "normalize_inputs.buffer_observation_images_top.std",
            "normalize_inputs.buffer_observation_state.mean",
            "normalize_inputs.buffer_observation_state.std",
            "unnormalize_outputs.buffer_action.mean",
            "unnormalize_outputs.buffer_action.std",
        )
        missing = [k for k in required if k not in stats]
        if missing:
            raise RuntimeError(
                f"Checkpoint {repo_id!r} is missing normalization buffers: {missing}"
            )
        return stats

    def reset(self) -> None:
        """Clear ACT action-chunk queue (call on every env reset)."""
        self._policy.reset()

    def _normalize_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply training-time (x - mean) / std normalization to images + state."""
        stats = self._stats
        device = self.device
        img = batch["observation.images.top"].to(device)  # (B, 3, H, W)
        state = batch["observation.state"].to(device)  # (B, 14)

        img_mean = stats["normalize_inputs.buffer_observation_images_top.mean"].to(device)
        img_std = stats["normalize_inputs.buffer_observation_images_top.std"].to(device)
        st_mean = stats["normalize_inputs.buffer_observation_state.mean"].to(device)
        st_std = stats["normalize_inputs.buffer_observation_state.std"].to(device)

        return {
            "observation.images.top": (img - img_mean) / (img_std + 1e-8),
            "observation.state": (state - st_mean) / (st_std + 1e-8),
        }

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map normalized action back to env scale using checkpoint stats."""
        mean = self._stats["unnormalize_outputs.buffer_action.mean"].to(action.device)
        std = self._stats["unnormalize_outputs.buffer_action.std"].to(action.device)
        return action * std + mean

    def _build_lerobot_batch(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Build ACT batch from env-wrapper observation dict."""
        if "lerobot" in observation:
            batch = observation["lerobot"]
            return {
                "observation.images.top": batch["observation.images.top"],
                "observation.state": batch["observation.state"],
            }

        image = observation["rgb_tensor"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # ACT expects native 480x640; resize if the wrapper downscaled.
        if image.shape[-2:] != (480, 640):
            image = torch.nn.functional.interpolate(
                image, size=(480, 640), mode="bilinear", align_corners=False
            )
        if "vector_tensor" not in observation:
            raise KeyError(
                "LeRobot ACT requires proprioceptive state "
                "(observation['vector_tensor'] or observation['lerobot'])."
            )
        state = observation["vector_tensor"]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return {
            "observation.images.top": image,  # (B, 3, 480, 640)
            "observation.state": state,  # (B, 14)
        }

    @torch.inference_mode()
    def predict(
        self,
        observation: dict[str, Any],
        text: str = "",
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Run ACT ``select_action`` on a formatted observation.

        Parameters
        ----------
        observation:
            Dict from ``RoboticsEnvWrapper`` (must include Aloha / LeRobot fields).
        text:
            Optional language instruction. Classic ACT is not language-conditioned;
            the string is accepted for API compatibility with VLA-style callers
            (swap to SmolVLA / PI0 for true text conditioning).
        deterministic:
            Unused for ACT (policy is deterministic given the chunk queue).

        Returns
        -------
        action:
            float32 tensor, shape ``(action_dim,)``.
        """
        _ = text, deterministic  # API compatibility with VLA callers
        batch = self._normalize_batch(self._build_lerobot_batch(observation))
        action = self._policy.select_action(batch)  # shape: (B, action_dim)
        action = self._unnormalize_action(action)
        if action.ndim == 2 and action.shape[0] == 1:
            return action.squeeze(0)  # shape: (action_dim,)
        return action


class LeRobotSmolVLAPolicy(nn.Module):
    """Language-conditioned SmolVLA wrapper for Aloha TransferCube.

    Loads a Hub SmolVLA checkpoint whose observation layout matches gym-aloha
    (``observation.images.top`` + 14-D state → 14-D action) and routes the
    natural-language ``task`` string through LeRobot's tokenizer preprocessor.

    Default repo: ``crislmfroes/smolvla-aloha-sim-transfer-cube-scripted``.
    """

    DEFAULT_REPO_ID: str = "crislmfroes/smolvla-aloha-sim-transfer-cube-scripted"

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        device: str = "cpu",
        cache_dir: str | Path = "checkpoints",
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.device_str = device
        self.device = torch.device(device)

        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "lerobot + transformers are required for SmolVLA. "
                "Install with: pip install lerobot transformers"
            ) from exc

        local_dir = self._prepare_local_checkpoint(repo_id, Path(cache_dir), SmolVLAConfig)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._policy = SmolVLAPolicy.from_pretrained(str(local_dir))
        self._policy.to(self.device)
        self._policy.eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config,
            str(local_dir),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.action_dim = int(self._policy.config.output_features["action"].shape[0])

    @staticmethod
    def _prepare_local_checkpoint(
        repo_id: str,
        cache_root: Path,
        config_cls: type,
    ) -> Path:
        """Download Hub weights and drop config keys unsupported by local LeRobot."""
        safe_name = repo_id.replace("/", "__")
        target = cache_root / safe_name
        target.mkdir(parents=True, exist_ok=True)
        marker = target / ".ready"
        if not marker.exists():
            src = Path(snapshot_download(repo_id))
            for item in src.iterdir():
                dest = target / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)

            cfg_path = target / "config.json"
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            valid = {f.name for f in fields(config_cls)} | {"type"}
            cleaned = {k: v for k, v in raw.items() if k in valid}
            cfg_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
            marker.write_text("ok", encoding="utf-8")
        return target

    def reset(self) -> None:
        """Clear SmolVLA action chunk queue."""
        self._policy.reset()

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        """Ensure the instruction ends with a newline (SmolVLA tokenizer convention)."""
        prompt = text.strip()
        if not prompt:
            prompt = "Transfer the cube between the Aloha arms"
        if not prompt.endswith("\n"):
            prompt += "\n"
        return prompt

    def _build_batch(self, observation: dict[str, Any], text: str) -> dict[str, Any]:
        """Pack env observation + language task for the SmolVLA preprocessor."""
        if "lerobot" in observation:
            batch: dict[str, Any] = dict(observation["lerobot"])
        else:
            image = observation["rgb_tensor"]
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.shape[-2:] != (480, 640):
                image = torch.nn.functional.interpolate(
                    image, size=(480, 640), mode="bilinear", align_corners=False
                )
            state = observation["vector_tensor"]
            if state.ndim == 1:
                state = state.unsqueeze(0)
            batch = {
                "observation.images.top": image,
                "observation.state": state,
            }
        batch["task"] = self._normalize_prompt(text)
        return batch

    @torch.inference_mode()
    def predict(
        self,
        observation: dict[str, Any],
        text: str = "",
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Run language-conditioned SmolVLA ``select_action``.

        Parameters
        ----------
        observation:
            Dict from ``RoboticsEnvWrapper``.
        text:
            Natural-language instruction (conditioned into the VLM backbone).
        deterministic:
            Unused (SmolVLA decoding is deterministic given the chunk queue).

        Returns
        -------
        action:
            float32 tensor, shape ``(action_dim,)``.
        """
        _ = deterministic
        batch = self._build_batch(observation, text)
        processed = self._preprocessor(batch)
        action = self._policy.select_action(processed)
        action = self._postprocessor(action)
        if isinstance(action, torch.Tensor):
            out = action
        else:
            out = torch.as_tensor(action, dtype=torch.float32)
        if out.ndim == 2 and out.shape[0] == 1:
            return out.squeeze(0)
        return out


def build_policy(
    action_dim: int,
    *,
    action_low: torch.Tensor | float,
    action_high: torch.Tensor | float,
    image_size: tuple[int, int] | None = (84, 84),
    device: str = "cpu",
    policy_type: PolicyKind = "act",
    use_lerobot: bool | None = None,
    lerobot_repo_id: str | None = None,
) -> nn.Module:
    """Factory for ACT / SmolVLA / Mock policies.

    Parameters
    ----------
    policy_type:
        ``"act"`` (default success demo), ``"smolvla"`` (language-conditioned),
        or ``"mock"``.
    use_lerobot:
        Deprecated alias: ``False`` forces ``mock``; ``True`` keeps ``policy_type``.
    """
    if use_lerobot is False:
        policy_type = "mock"

    if policy_type == "act":
        return LeRobotACTPolicy(
            repo_id=lerobot_repo_id or LeRobotACTPolicy.DEFAULT_REPO_ID,
            device=device,
        )
    if policy_type == "smolvla":
        return LeRobotSmolVLAPolicy(
            repo_id=lerobot_repo_id or LeRobotSmolVLAPolicy.DEFAULT_REPO_ID,
            device=device,
        )

    size = image_size if image_size is not None else (84, 84)
    policy = MockVLAPolicy(
        action_dim=action_dim,
        image_size=size,
        action_low=action_low,
        action_high=action_high,
    )
    return policy.to(device).eval()
