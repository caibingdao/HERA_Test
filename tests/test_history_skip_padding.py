import copy
import unittest

import torch
import torch.nn as nn

from models.history_encoder import HistoryStateEncoder
from replay_buffer import HistorySequenceBatch


class CountingGraphEncoder(nn.Module):
    def __init__(self, dimension=4):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)
        self.encoded_graphs = 0

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        self.encoded_graphs += node_inputs.size(0)
        return self.projection(node_inputs)


def make_history(padding_mask):
    padding_mask = torch.as_tensor(padding_mask, dtype=torch.bool)
    batch_size, sequence_length = padding_mask.shape
    nodes = 5
    generator = torch.Generator().manual_seed(20260725)
    node_inputs = torch.randn(
        batch_size,
        sequence_length,
        nodes,
        4,
        generator=generator,
    )
    node_padding_mask = torch.zeros(
        batch_size,
        sequence_length,
        1,
        nodes,
        dtype=torch.bool,
    )
    edge_mask = torch.zeros(
        batch_size,
        sequence_length,
        nodes,
        nodes,
        dtype=torch.bool,
    )
    current_index = torch.zeros(
        batch_size,
        sequence_length,
        1,
        1,
        dtype=torch.long,
    )
    current_edge = (
        torch.arange(3)
        .reshape(1, 1, 3, 1)
        .expand(batch_size, sequence_length, -1, -1)
        .clone()
    )
    edge_padding_mask = torch.zeros(
        batch_size,
        sequence_length,
        1,
        3,
        dtype=torch.bool,
    )
    observations = (
        node_inputs,
        node_padding_mask,
        edge_mask,
        current_index,
        current_edge,
        edge_padding_mask,
    )
    return HistorySequenceBatch(
        observations=observations,
        actions=torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.long,
        ),
        rewards=torch.zeros(batch_size, sequence_length, 1),
        robot_pose=torch.zeros(batch_size, sequence_length, 3),
        delta_coverage=torch.zeros(batch_size, sequence_length, 1),
        delta_frontier=torch.zeros(
            batch_size,
            sequence_length,
            1,
            dtype=torch.long,
        ),
        padding_mask=padding_mask,
        episode_ids=torch.arange(batch_size),
        decision_steps=torch.arange(batch_size),
    )


class HistorySkipPaddingTest(unittest.TestCase):
    def build_encoder(self, enabled):
        return HistoryStateEncoder(
            graph_embedding_dim=4,
            action_embedding_dim=2,
            token_dim=8,
            graph_batch_size=3,
            skip_padded_history_graphs=enabled,
        )

    def test_only_valid_graphs_are_encoded_for_padding_patterns(self):
        patterns = {
            "all_valid": [[False, False, False, False]],
            "half_left_padding": [[True, True, False, False]],
            "last_only": [[True, True, True, False]],
            "all_padding": [[True, True, True, True]],
            "mixed_rows": [
                [False, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ],
        }
        for name, pattern in patterns.items():
            with self.subTest(name=name):
                history = make_history(pattern)
                graph = CountingGraphEncoder()
                encoder = self.build_encoder(enabled=True)
                with torch.no_grad():
                    graph_embedding, action_feature = (
                        encoder.graph_features_from_observations(
                            graph,
                            history.observations,
                            history.actions,
                            padding_mask=history.padding_mask,
                        )
                    )
                expected = int((~history.padding_mask).sum())
                self.assertEqual(graph.encoded_graphs, expected)
                self.assertEqual(
                    graph_embedding.shape[:2],
                    history.padding_mask.shape,
                )
                self.assertTrue(
                    (
                        graph_embedding[history.padding_mask] == 0
                    ).all()
                )
                self.assertTrue(
                    (
                        action_feature[history.padding_mask] == 0
                    ).all()
                )

    def test_forward_and_valid_gradients_match_baseline(self):
        history = make_history(
            [
                [True, True, False, False],
                [True, False, False, False],
            ]
        )
        baseline_graph = CountingGraphEncoder()
        optimized_graph = copy.deepcopy(baseline_graph)
        baseline_graph.eval()
        optimized_graph.eval()
        baseline = self.build_encoder(enabled=False)
        optimized = self.build_encoder(enabled=True)
        optimized.load_state_dict(baseline.state_dict())

        baseline_tokens = baseline.encode_sequence(
            baseline_graph,
            history,
        )
        optimized_tokens = optimized.encode_sequence(
            optimized_graph,
            history,
        )
        torch.testing.assert_close(
            baseline_tokens,
            optimized_tokens,
            rtol=1e-6,
            atol=1e-7,
        )

        baseline_tokens.sum().backward()
        optimized_tokens.sum().backward()
        for baseline_parameter, optimized_parameter in zip(
            baseline_graph.parameters(),
            optimized_graph.parameters(),
        ):
            torch.testing.assert_close(
                baseline_parameter.grad,
                optimized_parameter.grad,
                rtol=1e-6,
                atol=1e-7,
            )
        for baseline_parameter, optimized_parameter in zip(
            baseline.parameters(),
            optimized.parameters(),
        ):
            torch.testing.assert_close(
                baseline_parameter.grad,
                optimized_parameter.grad,
                rtol=1e-6,
                atol=1e-7,
            )


if __name__ == "__main__":
    unittest.main()
