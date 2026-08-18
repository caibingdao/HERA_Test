import copy
import unittest
from unittest import mock

import torch

import models.history_encoder as history_module
from models.history_encoder import HistoryStateEncoder
from tests.test_history_skip_padding import (
    CountingGraphEncoder,
    make_history,
)


class HistoryCheckpointConfigTest(unittest.TestCase):
    def build_encoder(self, enabled):
        return HistoryStateEncoder(
            graph_embedding_dim=4,
            action_embedding_dim=2,
            token_dim=8,
            graph_batch_size=2,
            use_history_checkpoint=enabled,
        )

    def test_switch_controls_checkpoint_without_changing_gradients(self):
        history = make_history([[False, False], [False, False]])
        checkpoint_graph = CountingGraphEncoder()
        direct_graph = copy.deepcopy(checkpoint_graph)
        checkpoint_encoder = self.build_encoder(enabled=True)
        direct_encoder = self.build_encoder(enabled=False)
        direct_encoder.load_state_dict(checkpoint_encoder.state_dict())

        original_checkpoint = history_module.checkpoint
        with mock.patch.object(
            history_module,
            "checkpoint",
            wraps=original_checkpoint,
        ) as checkpoint_call:
            checkpoint_tokens = checkpoint_encoder.encode_sequence(
                checkpoint_graph,
                history,
            )
        self.assertGreater(checkpoint_call.call_count, 0)

        with mock.patch.object(
            history_module,
            "checkpoint",
            wraps=original_checkpoint,
        ) as checkpoint_call:
            direct_tokens = direct_encoder.encode_sequence(
                direct_graph,
                history,
            )
        self.assertEqual(checkpoint_call.call_count, 0)
        torch.testing.assert_close(
            checkpoint_tokens,
            direct_tokens,
            rtol=1e-6,
            atol=1e-7,
        )

        checkpoint_tokens.sum().backward()
        direct_tokens.sum().backward()
        for checkpoint_parameter, direct_parameter in zip(
            checkpoint_graph.parameters(),
            direct_graph.parameters(),
        ):
            torch.testing.assert_close(
                checkpoint_parameter.grad,
                direct_parameter.grad,
                rtol=1e-6,
                atol=1e-7,
            )
        for checkpoint_parameter, direct_parameter in zip(
            checkpoint_encoder.parameters(),
            direct_encoder.parameters(),
        ):
            torch.testing.assert_close(
                checkpoint_parameter.grad,
                direct_parameter.grad,
                rtol=1e-6,
                atol=1e-7,
            )


if __name__ == "__main__":
    unittest.main()
