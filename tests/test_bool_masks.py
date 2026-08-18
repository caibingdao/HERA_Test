import unittest

import numpy as np
import torch

from agent import Agent
from model import MultiHeadAttention


class BooleanMaskTest(unittest.TestCase):
    def test_attention_integer_and_boolean_masks_are_equivalent(self):
        torch.manual_seed(20260725)
        attention = MultiHeadAttention(embedding_dim=16, n_heads=4)
        query = torch.randn(2, 5, 16, requires_grad=True)
        adjacency = torch.randint(0, 2, (2, 5, 5), dtype=torch.int16)
        padding = torch.tensor(
            [
                [[0, 0, 0, 1, 1]],
                [[0, 0, 0, 0, 1]],
            ],
            dtype=torch.int16,
        )

        integer_output, _ = attention(
            query,
            attn_mask=adjacency,
            key_padding_mask=padding,
        )
        integer_gradient = torch.autograd.grad(
            integer_output.square().mean(),
            query,
        )[0]

        boolean_query = query.detach().clone().requires_grad_(True)
        boolean_output, _ = attention(
            boolean_query,
            attn_mask=adjacency.bool(),
            key_padding_mask=padding.bool(),
        )
        boolean_gradient = torch.autograd.grad(
            boolean_output.square().mean(),
            boolean_query,
        )[0]

        torch.testing.assert_close(
            integer_output,
            boolean_output,
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            integer_gradient,
            boolean_gradient,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_agent_observation_mask_and_index_dtypes(self):
        agent = Agent.__new__(Agent)
        agent.device = torch.device("cpu")
        agent.node_coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        )
        agent.utility = np.array([1.0, 2.0, 3.0])
        agent.guidepost = np.array([0.0, 1.0, 0.0])
        agent.current_index = 0
        agent.adjacent_matrix = np.array(
            [
                [False, False, True],
                [False, False, False],
                [True, False, False],
            ],
            dtype=bool,
        )
        agent.neighbor_indices = np.array([0, 1])

        observation = agent.get_observation()
        self.assertEqual(observation[1].dtype, torch.bool)
        self.assertEqual(observation[2].dtype, torch.bool)
        self.assertEqual(observation[5].dtype, torch.bool)
        self.assertEqual(observation[3].dtype, torch.long)
        self.assertEqual(observation[4].dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
