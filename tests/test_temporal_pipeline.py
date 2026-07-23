import unittest

import torch
import torch.nn as nn

from models.history_encoder import HistoryStateEncoder
from replay_buffer import (
    EpisodeTrajectory,
    TrajectoryReplayBuffer,
    TrajectoryStep,
)


def make_observation(value: float):
    node_inputs = torch.full((4, 4), value, dtype=torch.float32)
    node_padding_mask = torch.zeros((1, 4), dtype=torch.bool)
    edge_mask = torch.zeros((4, 4), dtype=torch.bool)
    current_index = torch.zeros((1, 1), dtype=torch.long)
    current_edge = torch.arange(3, dtype=torch.long).reshape(3, 1)
    edge_padding_mask = torch.zeros((1, 3), dtype=torch.bool)
    return (
        node_inputs,
        node_padding_mask,
        edge_mask,
        current_index,
        current_edge,
        edge_padding_mask,
    )


def make_episode(episode_id: int, offset: int):
    episode = EpisodeTrajectory(episode_id=episode_id)
    for step_index in range(5):
        value = float(offset + step_index)
        episode.append(
            TrajectoryStep(
                observation=make_observation(value),
                action=step_index % 3,
                reward=value / 10,
                next_observation=make_observation(value + 1),
                robot_pose=torch.tensor(
                    [value, -value, step_index * 0.1],
                    dtype=torch.float32,
                ),
                delta_coverage=step_index / 100,
                delta_frontier=step_index - 2,
                done=step_index == 4,
                step_index=step_index,
            )
        )
    return episode


class IdentityGraphModel(nn.Module):
    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        return node_inputs


class TemporalReplayBufferTest(unittest.TestCase):
    def setUp(self):
        self.replay = TrajectoryReplayBuffer(capacity_steps=20)
        self.replay.add_episode(make_episode(episode_id=10, offset=100))
        self.replay.add_episode(make_episode(episode_id=20, offset=200))

    def test_sequence_sampler_preserves_episode_boundaries(self):
        self.assertEqual(self.replay.available_windows(sequence_length=3), 6)

        batch = self.replay.sample_sequences(
            batch_size=6,
            sequence_length=3,
            seed=7,
        )

        self.assertEqual(batch.observations[0].shape, (6, 3, 4, 4))
        self.assertEqual(batch.actions.shape, (6, 3))
        self.assertEqual(batch.rewards.shape, (6, 3, 1))
        self.assertEqual(batch.robot_pose.shape, (6, 3, 3))
        self.assertEqual(batch.delta_coverage.shape, (6, 3, 1))
        self.assertEqual(batch.delta_frontier.shape, (6, 3, 1))

        sampled_windows = set()
        for row in range(batch.batch_size):
            episode_id = int(batch.episode_ids[row])
            start_step = int(batch.start_steps[row])
            sampled_windows.add((episode_id, start_step))

            offset = 100 if episode_id == 10 else 200
            values = batch.observations[0][row, :, 0, 0]
            expected = torch.tensor(
                [
                    offset + start_step,
                    offset + start_step + 1,
                    offset + start_step + 2,
                ],
                dtype=torch.float32,
            )
            torch.testing.assert_close(values, expected)
            expected_delta_frontier = torch.tensor(
                [start_step - 2, start_step - 1, start_step],
                dtype=torch.long,
            )
            torch.testing.assert_close(
                batch.delta_frontier[row, :, 0],
                expected_delta_frontier,
            )

        self.assertEqual(len(sampled_windows), 6)

    def test_history_encoder_builds_fixed_width_tokens(self):
        batch = self.replay.sample_sequences(
            batch_size=2,
            sequence_length=3,
            seed=3,
        )
        encoder = HistoryStateEncoder(
            graph_embedding_dim=4,
            action_embedding_dim=2,
            token_dim=8,
            pose_scale=100,
            reward_scale=20,
            delta_frontier_scale=100,
        )

        tokens = encoder.encode_sequence(IdentityGraphModel(), batch)

        self.assertEqual(tokens.shape, (2, 3, 8))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_graph_pooling_ignores_padded_nodes(self):
        node_features = torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]]
        )
        padding_mask = torch.tensor([[[False, False, True]]])

        pooled = HistoryStateEncoder.pool_graph(node_features, padding_mask)

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 2.0]]))


if __name__ == "__main__":
    unittest.main()
