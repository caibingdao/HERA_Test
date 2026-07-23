import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

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
