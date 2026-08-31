from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import torch


@dataclass(frozen=True)
class RolloutGraphContext:
    """Graph features computed for the current rollout decision state."""

    graph_embedding: torch.Tensor
    candidate_features: torch.Tensor

    def __post_init__(self):
        if self.graph_embedding.dim() != 2:
            raise ValueError(
                "graph_embedding must have shape [batch, embedding]"
            )
        if self.candidate_features.dim() != 3:
            raise ValueError(
                "candidate_features must have shape "
                "[batch, candidates, embedding]"
            )
        if self.graph_embedding.size(0) != self.candidate_features.size(0):
            raise ValueError("Rollout graph context batch sizes must match")
        if self.graph_embedding.size(-1) != self.candidate_features.size(-1):
            raise ValueError("Rollout graph context embedding sizes must match")

    def selected_action_feature(self, action_index: int) -> torch.Tensor:
        if self.graph_embedding.size(0) != 1:
            raise ValueError("Rollout cache only supports one actor observation")
        if not isinstance(action_index, int) or isinstance(action_index, bool):
            raise ValueError("action_index must be an integer")
        if not 0 <= action_index < self.candidate_features.size(1):
            raise ValueError(
                f"action_index {action_index} is outside the candidate list"
            )
        return self.candidate_features[:, action_index, :]


@dataclass(frozen=True)
class RolloutHistoryEntry:
    episode_id: int
    policy_version: int
    transition_id: int
    graph_embedding: torch.Tensor
    action_feature: torch.Tensor
    token: torch.Tensor


@dataclass(frozen=True)
class RolloutHistorySequence:
    """A left-padded sequence of finalized rollout history tokens."""

    tokens: torch.Tensor
    padding_mask: torch.Tensor
    episode_id: int
    decision_step: int
    transition_ids: Tuple[Optional[int], ...]

    @property
    def batch_size(self):
        return self.tokens.size(0)

    @property
    def sequence_length(self):
        return self.tokens.size(1)


class RolloutHistoryCache:
    """Bounded, episode-local cache of finalized temporal history tokens."""

    def __init__(
        self,
        sequence_length: int,
        graph_embedding_dim: int,
        token_dim: int,
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if graph_embedding_dim <= 0 or token_dim <= 0:
            raise ValueError("cache embedding dimensions must be positive")

        self.sequence_length = sequence_length
        self.graph_embedding_dim = graph_embedding_dim
        self.token_dim = token_dim
        self._entries: Deque[RolloutHistoryEntry] = deque(
            maxlen=sequence_length
        )
        self._episode_id = None
        self._policy_version = 0
        self._device = None
        self._dtype = None
        self._use_mamba = None
        self._active = False

    @property
    def entries(self):
        return tuple(self._entries)

    @property
    def policy_version(self):
        return self._policy_version

    @property
    def active(self):
        return self._active

    def __len__(self):
        return len(self._entries)

    def invalidate(self):
        self._entries.clear()
        self._episode_id = None
        self._device = None
        self._dtype = None
        self._use_mamba = None
        self._active = False

    def on_policy_update(self, policy_version: int):
        self._validate_non_negative_int(policy_version, "policy_version")
        self.invalidate()
        self._policy_version = policy_version

    def start_episode(
        self,
        episode_id: int,
        policy_version: int,
        device,
        dtype: torch.dtype,
        use_mamba: bool,
        sequence_length: Optional[int] = None,
        graph_embedding_dim: Optional[int] = None,
        token_dim: Optional[int] = None,
    ):
        self._validate_non_negative_int(episode_id, "episode_id")
        self._validate_non_negative_int(policy_version, "policy_version")
        requested_configuration = (
            self.sequence_length
            if sequence_length is None
            else sequence_length,
            self.graph_embedding_dim
            if graph_embedding_dim is None
            else graph_embedding_dim,
            self.token_dim if token_dim is None else token_dim,
        )
        expected_configuration = (
            self.sequence_length,
            self.graph_embedding_dim,
            self.token_dim,
        )
        if requested_configuration != expected_configuration:
            self.invalidate()
            raise RuntimeError(
                "Rollout cache sequence or embedding dimensions changed: "
                f"cache={expected_configuration}, "
                f"requested={requested_configuration}"
            )
        if not use_mamba:
            self.invalidate()
            raise ValueError("Rollout history cache requires USE_MAMBA=True")
        if not isinstance(dtype, torch.dtype):
            raise ValueError("dtype must be a torch.dtype")

        self.invalidate()
        self._episode_id = episode_id
        self._policy_version = policy_version
        self._device = torch.device(device)
        self._dtype = dtype
        self._use_mamba = True
        self._active = True

    def end_episode(self, episode_id: int):
        self._ensure_active()
        if episode_id != self._episode_id:
            self.invalidate()
            raise RuntimeError(
                "Rollout cache episode mismatch at episode end: "
                f"active={self._episode_id}, received={episode_id}"
            )
        self.invalidate()

    def is_valid(
        self,
        episode_id: int,
        policy_version: int,
        decision_step: int,
        device,
        dtype: torch.dtype,
        use_mamba: bool,
    ) -> bool:
        if not self._active:
            return False
        expected_first_id = max(0, decision_step - len(self._entries))
        transition_ids = [
            entry.transition_id
            for entry in self._entries
        ]
        return (
            self._episode_id == episode_id
            and self._policy_version == policy_version
            and self._device == torch.device(device)
            and self._dtype == dtype
            and self._use_mamba == use_mamba
            and transition_ids
            == list(range(expected_first_id, decision_step))
        )

    def append(
        self,
        episode_id: int,
        policy_version: int,
        transition_id: int,
        graph_embedding: torch.Tensor,
        action_feature: torch.Tensor,
        token: torch.Tensor,
    ):
        self._ensure_active()
        self._validate_non_negative_int(transition_id, "transition_id")
        if episode_id != self._episode_id:
            self.invalidate()
            raise RuntimeError(
                "Rollout cache cannot cross episode boundaries"
            )
        if policy_version != self._policy_version:
            self.invalidate()
            raise RuntimeError(
                "Rollout cache policy version changed before append"
            )

        expected_transition_id = (
            self._entries[-1].transition_id + 1
            if self._entries
            else 0
        )
        if transition_id != expected_transition_id:
            self.invalidate()
            raise RuntimeError(
                "Rollout cache requires continuous transition ids: "
                f"expected={expected_transition_id}, "
                f"received={transition_id}"
            )

        graph_embedding = self._validate_feature(
            graph_embedding,
            self.graph_embedding_dim,
            "graph_embedding",
        )
        action_feature = self._validate_feature(
            action_feature,
            self.graph_embedding_dim,
            "action_feature",
        )
        token = self._validate_feature(
            token,
            self.token_dim,
            "token",
        )

        self._entries.append(
            RolloutHistoryEntry(
                episode_id=episode_id,
                policy_version=policy_version,
                transition_id=transition_id,
                graph_embedding=graph_embedding.detach(),
                action_feature=action_feature.detach(),
                token=token.detach(),
            )
        )

    def get_sequence(
        self,
        episode_id: int,
        policy_version: int,
        decision_step: int,
        device,
        dtype: torch.dtype,
        use_mamba: bool,
    ) -> RolloutHistorySequence:
        if not self.is_valid(
            episode_id=episode_id,
            policy_version=policy_version,
            decision_step=decision_step,
            device=device,
            dtype=dtype,
            use_mamba=use_mamba,
        ):
            self.invalidate()
            raise RuntimeError(
                "Rollout history cache metadata or transition order is invalid"
            )

        valid_tokens = [entry.token for entry in self._entries]
        padding_length = self.sequence_length - len(valid_tokens)
        padding_tokens = [
            torch.zeros(
                self.token_dim,
                device=self._device,
                dtype=self._dtype,
            )
            for _ in range(padding_length)
        ]
        tokens = torch.stack(padding_tokens + valid_tokens, dim=0).unsqueeze(0)
        padding_mask = torch.tensor(
            [True] * padding_length + [False] * len(valid_tokens),
            device=self._device,
            dtype=torch.bool,
        ).unsqueeze(0)
        transition_ids = (
            (None,) * padding_length
            + tuple(entry.transition_id for entry in self._entries)
        )
        return RolloutHistorySequence(
            tokens=tokens,
            padding_mask=padding_mask,
            episode_id=episode_id,
            decision_step=decision_step,
            transition_ids=transition_ids,
        )

    def _validate_feature(
        self,
        feature: torch.Tensor,
        expected_dim: int,
        name: str,
    ) -> torch.Tensor:
        if feature.shape == (1, expected_dim):
            feature = feature.squeeze(0)
        if feature.shape != (expected_dim,):
            self.invalidate()
            raise RuntimeError(
                f"{name} must have shape ({expected_dim},), "
                f"got {tuple(feature.shape)}"
            )
        if feature.device != self._device or feature.dtype != self._dtype:
            self.invalidate()
            raise RuntimeError(
                f"{name} device/dtype changed while the cache was active"
            )
        return feature

    def _ensure_active(self):
        if not self._active:
            raise RuntimeError("Rollout history cache is not active")

    @staticmethod
    def _validate_non_negative_int(value: int, name: str):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
