import unittest

import numpy as np
import torch
import torch.optim as optim

from driver import sac_update
from model import PolicyNet, QNet
from replay_buffer import HistorySequenceBatch


def make_observation(batch_size, nodes=8, neighbors=3, device="cuda"):
    node_inputs = torch.randn(batch_size, nodes, 4, device=device)
    node_padding_mask = torch.zeros(
        batch_size,
        1,
        nodes,
        dtype=torch.bool,
        device=device,
    )
    edge_mask = torch.zeros(
        batch_size,
        nodes,
        nodes,
        dtype=torch.bool,
        device=device,
    )
    current_index = torch.zeros(
        batch_size,
        1,
        1,
        dtype=torch.long,
        device=device,
    )
    current_edge = (
        torch.arange(neighbors, device=device)
        .reshape(1, neighbors, 1)
        .expand(batch_size, -1, -1)
    )
    edge_padding_mask = torch.zeros(
        batch_size,
        1,
        neighbors,
        dtype=torch.bool,
        device=device,
    )
    return (
        node_inputs,
        node_padding_mask,
        edge_mask,
        current_index,
        current_edge,
        edge_padding_mask,
    )


def make_history(batch_size, sequence_length, device="cuda"):
    observations = make_observation(
        batch_size * sequence_length,
        device=device,
    )
    observations = tuple(
        tensor.reshape(
            batch_size,
            sequence_length,
            *tensor.shape[1:],
        )
        for tensor in observations
    )
    return HistorySequenceBatch(
        observations=observations,
        actions=torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.long,
            device=device,
        ),
        rewards=torch.zeros(
            batch_size,
            sequence_length,
            1,
            device=device,
        ),
        robot_pose=torch.zeros(
            batch_size,
            sequence_length,
            3,
            device=device,
        ),
        delta_coverage=torch.zeros(
            batch_size,
            sequence_length,
            1,
            device=device,
        ),
        delta_frontier=torch.zeros(
            batch_size,
            sequence_length,
            1,
            dtype=torch.long,
            device=device,
        ),
        padding_mask=torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        episode_ids=torch.arange(batch_size, device=device),
        decision_steps=torch.arange(batch_size, device=device),
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    "BF16 SAC smoke test requires supported CUDA hardware",
)
class Bf16SacSmokeTest(unittest.TestCase):
    def test_single_update_is_finite_and_keeps_master_weights_fp32(self):
        torch.manual_seed(20260725)
        device = torch.device("cuda")
        kwargs = {
            "use_mamba": True,
            "mamba_seq_len": 2,
            "mamba_d_model": 32,
            "mamba_d_state": 8,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "fusion_hidden_dim": 32,
            "action_embedding_dim": 8,
            "history_graph_batch_size": 4,
            "skip_padded_history_graphs": True,
            "trim_history_graph_nodes": True,
        }
        policy = PolicyNet(4, 32, **kwargs).to(device)
        q1 = QNet(4, 32, **kwargs).to(device)
        q2 = QNet(4, 32, **kwargs).to(device)
        target_q1 = QNet(4, 32, **kwargs).to(device)
        target_q2 = QNet(4, 32, **kwargs).to(device)
        target_q1.load_state_dict(q1.state_dict())
        target_q2.load_state_dict(q2.state_dict())
        policy_optimizer = optim.Adam(policy.parameters(), lr=1e-5)
        q1_optimizer = optim.Adam(q1.parameters(), lr=1e-5)
        q2_optimizer = optim.Adam(q2.parameters(), lr=1e-5)
        log_alpha = torch.tensor(
            [-2.0],
            device=device,
            requires_grad=True,
        )
        alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
        observation = make_observation(2)
        next_observation = make_observation(2)
        history = make_history(2, 2)

        result = sac_update(
            policy_net=policy,
            q_net1=q1,
            q_net2=q2,
            target_q_net1=target_q1,
            target_q_net2=target_q2,
            policy_optimizer=policy_optimizer,
            q_net1_optimizer=q1_optimizer,
            q_net2_optimizer=q2_optimizer,
            log_alpha=log_alpha,
            log_alpha_optimizer=alpha_optimizer,
            entropy_target=0.05 * (-np.log(1 / 3)),
            observation=observation,
            action=torch.zeros(2, 1, 1, dtype=torch.long, device=device),
            reward=torch.zeros(2, 1, 1, device=device),
            done=torch.zeros(2, 1, 1, device=device),
            next_observation=next_observation,
            current_history=history,
            next_history=history,
            use_bf16=True,
            check_finite=True,
        )

        for value in result.values():
            self.assertTrue(torch.isfinite(value).all())
        for network in (policy, q1, q2, target_q1, target_q2):
            self.assertTrue(
                all(
                    parameter.dtype == torch.float32
                    for parameter in network.parameters()
                )
            )


if __name__ == "__main__":
    unittest.main()
