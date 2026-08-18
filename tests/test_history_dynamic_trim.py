import copy
import unittest

import torch
import torch.nn as nn

from models.history_encoder import HistoryStateEncoder
from tests.test_history_skip_padding import make_history


class RecordingGraphEncoder(nn.Module):
    def __init__(self, dimension=4):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)
        self.node_counts = []

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        self.node_counts.append(node_inputs.size(1))
        return self.projection(node_inputs)


class HistoryDynamicTrimTest(unittest.TestCase):
    def build_encoder(self, trim):
        return HistoryStateEncoder(
            graph_embedding_dim=4,
            action_embedding_dim=2,
            token_dim=8,
            graph_batch_size=2,
            trim_history_graph_nodes=trim,
        )

    @staticmethod
    def add_node_padding(history):
        node_padding = history.observations[1].clone()
        node_padding[0, :, :, 3:] = True
        if node_padding.size(0) > 1:
            node_padding[1, :, :, 4:] = True
        observations = list(history.observations)
        observations[1] = node_padding
        observations[2] = observations[2].clone()
        observations[2][0, :, 3:, :] = True
        observations[2][0, :, :, 3:] = True
        if observations[2].size(0) > 1:
            observations[2][1, :, 4:, :] = True
            observations[2][1, :, :, 4:] = True
        return history._replace(observations=tuple(observations))

    def test_trimmed_outputs_and_gradients_match_untrimmed(self):
        history = self.add_node_padding(
            make_history(
                [
                    [False, False],
                    [False, False],
                ]
            )
        )
        baseline_graph = RecordingGraphEncoder()
        trimmed_graph = copy.deepcopy(baseline_graph)
        baseline_graph.eval()
        trimmed_graph.eval()
        baseline = self.build_encoder(trim=False)
        trimmed = self.build_encoder(trim=True)
        trimmed.load_state_dict(baseline.state_dict())

        baseline_tokens = baseline.encode_sequence(
            baseline_graph,
            history,
        )
        trimmed_tokens = trimmed.encode_sequence(
            trimmed_graph,
            history,
        )
        self.assertTrue(all(count == 5 for count in baseline_graph.node_counts))
        self.assertEqual(trimmed_graph.node_counts, [3, 4])
        torch.testing.assert_close(
            baseline_tokens,
            trimmed_tokens,
            rtol=1e-6,
            atol=1e-7,
        )

        baseline_tokens.sum().backward()
        trimmed_tokens.sum().backward()
        for baseline_parameter, trimmed_parameter in zip(
            baseline_graph.parameters(),
            trimmed_graph.parameters(),
        ):
            torch.testing.assert_close(
                baseline_parameter.grad,
                trimmed_parameter.grad,
                rtol=1e-6,
                atol=1e-7,
            )
        for baseline_parameter, trimmed_parameter in zip(
            baseline.parameters(),
            trimmed.parameters(),
        ):
            torch.testing.assert_close(
                baseline_parameter.grad,
                trimmed_parameter.grad,
                rtol=1e-6,
                atol=1e-7,
            )

    def test_selected_action_must_survive_trim(self):
        history = self.add_node_padding(make_history([[False, False]]))
        observations = list(history.observations)
        current_edge = observations[4].clone()
        current_edge[:, :, 0, 0] = 4
        observations[4] = current_edge
        history = history._replace(observations=tuple(observations))

        with self.assertRaisesRegex(ValueError, "outside"):
            self.build_encoder(trim=True).graph_features_from_observations(
                RecordingGraphEncoder(),
                history.observations,
                history.actions,
                padding_mask=history.padding_mask,
            )


if __name__ == "__main__":
    unittest.main()
