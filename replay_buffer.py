import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, NamedTuple, Optional, Sequence, Tuple

import torch


Observation = Tuple[torch.Tensor, ...]


def detach_observation(observation: Sequence[torch.Tensor]) -> Observation:
    """Remove the worker batch dimension and keep replay data on CPU."""
    if len(observation) != 6:
        raise ValueError(f"Expected 6 observation tensors, got {len(observation)}")

    detached = []
    for tensor in observation:
        if tensor.size(0) != 1:
            raise ValueError(
                "Worker observations must have a singleton batch dimension, "
                f"got shape {tuple(tensor.shape)}"
            )
        detached.append(tensor.detach().cpu().squeeze(0))
    return tuple(detached)


@dataclass(frozen=True)
class TrajectoryStep:
    observation: Observation
    action: int
    reward: float
    next_observation: Observation
    robot_pose: torch.Tensor
    delta_coverage: float
    delta_frontier: int
    done: bool
    step_index: int

    def __post_init__(self):
        if len(self.observation) != 6 or len(self.next_observation) != 6:
            raise ValueError("Trajectory observations must contain 6 tensors")
        if self.robot_pose.shape != (3,):
            raise ValueError(
                f"robot_pose must have shape (3,), got {tuple(self.robot_pose.shape)}"
            )
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")


@dataclass
class EpisodeTrajectory:
    episode_id: int
    steps: List[TrajectoryStep] = field(default_factory=list)

    def append(self, step: TrajectoryStep):
        expected_index = len(self.steps)
        if step.step_index != expected_index:
            raise ValueError(
                f"Expected step_index {expected_index}, got {step.step_index}"
            )
        self.steps.append(step)

    def __len__(self):
        return len(self.steps)


@dataclass(frozen=True)
class SequenceBatch:
    observations: Observation
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: Observation
    robot_pose: torch.Tensor
    delta_coverage: torch.Tensor
    delta_frontier: torch.Tensor
    dones: torch.Tensor
    episode_ids: torch.Tensor
    start_steps: torch.Tensor

    @property
    def batch_size(self):
        return self.actions.size(0)

    @property
    def sequence_length(self):
        return self.actions.size(1)

    def to(self, device):
        return SequenceBatch(
            observations=tuple(tensor.to(device) for tensor in self.observations),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_observations=tuple(
                tensor.to(device) for tensor in self.next_observations
            ),
            robot_pose=self.robot_pose.to(device),
            delta_coverage=self.delta_coverage.to(device),
            delta_frontier=self.delta_frontier.to(device),
            dones=self.dones.to(device),
            episode_ids=self.episode_ids.to(device),
            start_steps=self.start_steps.to(device),
        )


class HistorySequenceBatch(NamedTuple):
    """A left-padded history ending immediately before a decision state."""

    observations: Observation
    actions: torch.Tensor
    rewards: torch.Tensor
    robot_pose: torch.Tensor
    delta_coverage: torch.Tensor
    delta_frontier: torch.Tensor
    padding_mask: torch.Tensor
    episode_ids: torch.Tensor
    decision_steps: torch.Tensor

    @property
    def batch_size(self):
        return self.actions.size(0)

    @property
    def sequence_length(self):
        return self.actions.size(1)

    def to(self, device):
        return HistorySequenceBatch(
            observations=tuple(
                tensor.to(device) for tensor in self.observations
            ),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            robot_pose=self.robot_pose.to(device),
            delta_coverage=self.delta_coverage.to(device),
            delta_frontier=self.delta_frontier.to(device),
            padding_mask=self.padding_mask.to(device),
            episode_ids=self.episode_ids.to(device),
            decision_steps=self.decision_steps.to(device),
        )


class AlignedTransitionBatch(NamedTuple):
    """SAC transitions with separately aligned current and next histories."""

    observations: Observation
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_observations: Observation
    current_history: HistorySequenceBatch
    next_history: HistorySequenceBatch
    episode_ids: torch.Tensor
    step_indices: torch.Tensor

    @property
    def batch_size(self):
        return self.actions.size(0)

    def to(self, device):
        return AlignedTransitionBatch(
            observations=tuple(
                tensor.to(device) for tensor in self.observations
            ),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            next_observations=tuple(
                tensor.to(device) for tensor in self.next_observations
            ),
            current_history=self.current_history.to(device),
            next_history=self.next_history.to(device),
            episode_ids=self.episode_ids.to(device),
            step_indices=self.step_indices.to(device),
        )


class TrajectoryReplayBuffer:
    """Episode-aware replay storage for contiguous temporal sequences."""

    def __init__(self, capacity_steps: int):
        if capacity_steps <= 0:
            raise ValueError("capacity_steps must be positive")
        self.capacity_steps = capacity_steps
        self.episodes: Deque[EpisodeTrajectory] = deque()
        self.num_steps = 0

    def __len__(self):
        return self.num_steps

    @property
    def num_episodes(self):
        return len(self.episodes)

    def add_episode(self, trajectory: EpisodeTrajectory):
        if len(trajectory) == 0:
            return
        if len(trajectory) > self.capacity_steps:
            raise ValueError(
                "A single episode cannot exceed trajectory replay capacity"
            )

        self.episodes.append(trajectory)
        self.num_steps += len(trajectory)

        while self.num_steps > self.capacity_steps:
            removed = self.episodes.popleft()
            self.num_steps -= len(removed)

    def available_windows(self, sequence_length: int):
        self._validate_sequence_length(sequence_length)
        return sum(
            max(0, len(episode) - sequence_length + 1)
            for episode in self.episodes
        )

    @property
    def available_transitions(self):
        return self.num_steps

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
        seed: Optional[int] = None,
    ) -> SequenceBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._validate_sequence_length(sequence_length)

        windows = []
        for episode in self.episodes:
            for start in range(len(episode) - sequence_length + 1):
                windows.append((episode, start))

        if not windows:
            raise ValueError(
                f"No episode contains a sequence of length {sequence_length}"
            )

        rng = random.Random(seed)
        if batch_size <= len(windows):
            selected = rng.sample(windows, batch_size)
        else:
            selected = rng.choices(windows, k=batch_size)

        return self._collate(selected, sequence_length)

    def sample_aligned_transitions(
        self,
        batch_size: int,
        sequence_length: int,
        seed: Optional[int] = None,
    ) -> AlignedTransitionBatch:
        """Sample transition_t with history_t and one-step-shifted history_t+1."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._validate_sequence_length(sequence_length)

        transitions = [
            (episode, step_index)
            for episode in self.episodes
            for step_index in range(len(episode))
        ]
        if not transitions:
            raise ValueError("No transitions are available")

        rng = random.Random(seed)
        if batch_size <= len(transitions):
            selected = rng.sample(transitions, batch_size)
        else:
            selected = rng.choices(transitions, k=batch_size)

        return self._collate_aligned_transitions(
            selected,
            sequence_length,
        )

    @staticmethod
    def _validate_sequence_length(sequence_length: int):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

    @staticmethod
    def _stack_observations(
        selected: Sequence[Tuple[EpisodeTrajectory, int]],
        sequence_length: int,
        use_next: bool,
    ) -> Observation:
        components = []
        for component_index in range(6):
            batch_component = []
            for episode, start in selected:
                steps = episode.steps[start:start + sequence_length]
                tensors = [
                    (
                        step.next_observation[component_index]
                        if use_next
                        else step.observation[component_index]
                    )
                    for step in steps
                ]
                batch_component.append(torch.stack(tensors, dim=0))
            components.append(torch.stack(batch_component, dim=0))
        return tuple(components)

    @staticmethod
    def _stack_transition_observations(
        selected: Sequence[Tuple[EpisodeTrajectory, int]],
        use_next: bool,
    ) -> Observation:
        return tuple(
            torch.stack(
                [
                    (
                        episode.steps[step_index].next_observation[
                            component_index
                        ]
                        if use_next
                        else episode.steps[step_index].observation[
                            component_index
                        ]
                    )
                    for episode, step_index in selected
                ],
                dim=0,
            )
            for component_index in range(6)
        )

    @staticmethod
    def _zero_observation(reference: Observation) -> Observation:
        return tuple(torch.zeros_like(tensor) for tensor in reference)

    @classmethod
    def _collate_histories(
        cls,
        selected: Sequence[Tuple[EpisodeTrajectory, int]],
        sequence_length: int,
    ) -> HistorySequenceBatch:
        """Build histories before each decision_step without crossing episodes."""
        history_rows = []
        observation_rows = [[] for _ in range(6)]
        actions = []
        rewards = []
        robot_pose = []
        delta_coverage = []
        delta_frontier = []
        padding_masks = []

        for episode, decision_step in selected:
            if not 0 <= decision_step <= len(episode):
                raise ValueError(
                    "decision_step must be between zero and episode length"
                )

            history_start = max(0, decision_step - sequence_length)
            valid_steps = episode.steps[history_start:decision_step]
            padding_length = sequence_length - len(valid_steps)
            padding_mask = [True] * padding_length + [False] * len(valid_steps)

            reference_observation = episode.steps[0].observation
            zero_observation = cls._zero_observation(reference_observation)
            for component_index in range(6):
                values = [
                    zero_observation[component_index]
                    for _ in range(padding_length)
                ]
                values.extend(
                    step.observation[component_index] for step in valid_steps
                )
                observation_rows[component_index].append(
                    torch.stack(values, dim=0)
                )

            actions.append(
                [0] * padding_length + [step.action for step in valid_steps]
            )
            rewards.append(
                [0.0] * padding_length
                + [step.reward for step in valid_steps]
            )
            robot_pose.append(
                torch.stack(
                    [torch.zeros(3, dtype=torch.float32)] * padding_length
                    + [step.robot_pose.float() for step in valid_steps],
                    dim=0,
                )
            )
            delta_coverage.append(
                [0.0] * padding_length
                + [step.delta_coverage for step in valid_steps]
            )
            delta_frontier.append(
                [0] * padding_length
                + [step.delta_frontier for step in valid_steps]
            )
            padding_masks.append(padding_mask)
            history_rows.append((episode.episode_id, decision_step))

        observations = tuple(
            torch.stack(component_rows, dim=0)
            for component_rows in observation_rows
        )
        return HistorySequenceBatch(
            observations=observations,
            actions=torch.tensor(actions, dtype=torch.long),
            rewards=torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
            robot_pose=torch.stack(robot_pose, dim=0),
            delta_coverage=torch.tensor(
                delta_coverage,
                dtype=torch.float32,
            ).unsqueeze(-1),
            delta_frontier=torch.tensor(
                delta_frontier,
                dtype=torch.long,
            ).unsqueeze(-1),
            padding_mask=torch.tensor(padding_masks, dtype=torch.bool),
            episode_ids=torch.tensor(
                [episode_id for episode_id, _ in history_rows],
                dtype=torch.long,
            ),
            decision_steps=torch.tensor(
                [decision_step for _, decision_step in history_rows],
                dtype=torch.long,
            ),
        )

    @classmethod
    def _collate_aligned_transitions(
        cls,
        selected: Sequence[Tuple[EpisodeTrajectory, int]],
        sequence_length: int,
    ) -> AlignedTransitionBatch:
        steps = [
            episode.steps[step_index]
            for episode, step_index in selected
        ]
        current_history = cls._collate_histories(
            selected,
            sequence_length,
        )
        next_history = cls._collate_histories(
            [
                (episode, step_index + 1)
                for episode, step_index in selected
            ],
            sequence_length,
        )

        return AlignedTransitionBatch(
            observations=cls._stack_transition_observations(
                selected,
                use_next=False,
            ),
            actions=torch.tensor(
                [step.action for step in steps],
                dtype=torch.long,
            ).reshape(-1, 1, 1),
            rewards=torch.tensor(
                [step.reward for step in steps],
                dtype=torch.float32,
            ).reshape(-1, 1, 1),
            dones=torch.tensor(
                [step.done for step in steps],
                dtype=torch.float32,
            ).reshape(-1, 1, 1),
            next_observations=cls._stack_transition_observations(
                selected,
                use_next=True,
            ),
            current_history=current_history,
            next_history=next_history,
            episode_ids=torch.tensor(
                [episode.episode_id for episode, _ in selected],
                dtype=torch.long,
            ),
            step_indices=torch.tensor(
                [step_index for _, step_index in selected],
                dtype=torch.long,
            ),
        )

    @classmethod
    def _collate(
        cls,
        selected: Sequence[Tuple[EpisodeTrajectory, int]],
        sequence_length: int,
    ) -> SequenceBatch:
        observations = cls._stack_observations(
            selected, sequence_length, use_next=False
        )
        next_observations = cls._stack_observations(
            selected, sequence_length, use_next=True
        )

        step_windows = [
            episode.steps[start:start + sequence_length]
            for episode, start in selected
        ]

        actions = torch.tensor(
            [[step.action for step in steps] for steps in step_windows],
            dtype=torch.long,
        )
        rewards = torch.tensor(
            [[step.reward for step in steps] for steps in step_windows],
            dtype=torch.float32,
        ).unsqueeze(-1)
        robot_pose = torch.stack(
            [
                torch.stack([step.robot_pose for step in steps], dim=0)
                for steps in step_windows
            ],
            dim=0,
        ).float()
        delta_coverage = torch.tensor(
            [[step.delta_coverage for step in steps] for steps in step_windows],
            dtype=torch.float32,
        ).unsqueeze(-1)
        delta_frontier = torch.tensor(
            [[step.delta_frontier for step in steps] for steps in step_windows],
            dtype=torch.long,
        ).unsqueeze(-1)
        dones = torch.tensor(
            [[step.done for step in steps] for steps in step_windows],
            dtype=torch.bool,
        ).unsqueeze(-1)
        episode_ids = torch.tensor(
            [episode.episode_id for episode, _ in selected],
            dtype=torch.long,
        )
        start_steps = torch.tensor(
            [start for _, start in selected],
            dtype=torch.long,
        )

        return SequenceBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            robot_pose=robot_pose,
            delta_coverage=delta_coverage,
            delta_frontier=delta_frontier,
            dones=dones,
            episode_ids=episode_ids,
            start_steps=start_steps,
        )


def build_history_sequence(
    trajectory: EpisodeTrajectory,
    sequence_length: int,
    decision_step: Optional[int] = None,
) -> Optional[HistorySequenceBatch]:
    """Build one online history; return None when no transition exists yet."""
    TrajectoryReplayBuffer._validate_sequence_length(sequence_length)
    if decision_step is None:
        decision_step = len(trajectory)
    if not 0 <= decision_step <= len(trajectory):
        raise ValueError(
            "decision_step must be between zero and trajectory length"
        )
    if len(trajectory) == 0:
        return None
    return TrajectoryReplayBuffer._collate_histories(
        [(trajectory, decision_step)],
        sequence_length,
    )
