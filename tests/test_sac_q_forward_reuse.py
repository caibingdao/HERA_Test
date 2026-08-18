import copy
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from driver import sac_update


class TinyPolicy(nn.Module):
    def __init__(self, features=4, actions=3):
        super().__init__()
        self.output = nn.Linear(features, actions)
        self.forward_count = 0

    def forward(self, node_inputs, *unused, history_batch=None):
        self.forward_count += 1
        state = node_inputs.mean(dim=1)
        return torch.log_softmax(self.output(state), dim=-1)


class TinyQ(nn.Module):
    def __init__(self, features=4, actions=3):
        super().__init__()
        self.output = nn.Linear(features, actions)
        self.forward_count = 0

    def forward(self, node_inputs, *unused, history_batch=None):
        self.forward_count += 1
        state = node_inputs.mean(dim=1)
        return self.output(state).unsqueeze(-1)


def make_observation(batch_size=5):
    node_inputs = torch.randn(batch_size, 4, 4)
    placeholders = [torch.empty(0) for _ in range(5)]
    return (node_inputs, *placeholders)


def make_bundle(seed):
    torch.manual_seed(seed)
    policy = TinyPolicy()
    q1 = TinyQ()
    q2 = TinyQ()
    target_q1 = copy.deepcopy(q1)
    target_q2 = copy.deepcopy(q2)
    log_alpha = torch.tensor([-2.0], requires_grad=True)
    return {
        "policy_net": policy,
        "q_net1": q1,
        "q_net2": q2,
        "target_q_net1": target_q1,
        "target_q_net2": target_q2,
        "policy_optimizer": optim.Adam(policy.parameters(), lr=1e-4),
        "q_net1_optimizer": optim.Adam(q1.parameters(), lr=1e-4),
        "q_net2_optimizer": optim.Adam(q2.parameters(), lr=1e-4),
        "log_alpha": log_alpha,
        "log_alpha_optimizer": optim.Adam([log_alpha], lr=1e-4),
    }


class SacQForwardReuseTest(unittest.TestCase):
    def test_reuse_matches_original_updates_and_halves_current_q_forwards(self):
        torch.manual_seed(20260725)
        observation = make_observation()
        next_observation = make_observation()
        action = torch.tensor([0, 1, 2, 0, 1]).reshape(-1, 1, 1)
        reward = torch.randn(5, 1, 1)
        done = torch.zeros(5, 1, 1)
        baseline = make_bundle(seed=17)
        reused = make_bundle(seed=17)

        common = {
            "entropy_target": 0.05 * (-np.log(1 / 3)),
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "next_observation": next_observation,
            "use_bf16": False,
            "check_finite": True,
        }
        baseline_result = sac_update(
            **baseline,
            **common,
            reuse_current_q_forward=False,
        )
        reused_result = sac_update(
            **reused,
            **common,
            reuse_current_q_forward=True,
        )

        for name in baseline_result:
            torch.testing.assert_close(
                baseline_result[name],
                reused_result[name],
                rtol=1e-6,
                atol=1e-7,
            )
        for model_name in ("policy_net", "q_net1", "q_net2"):
            baseline_state = baseline[model_name].state_dict()
            reused_state = reused[model_name].state_dict()
            for name in baseline_state:
                torch.testing.assert_close(
                    baseline_state[name],
                    reused_state[name],
                    rtol=1e-6,
                    atol=1e-7,
                )

        self.assertEqual(baseline["q_net1"].forward_count, 2)
        self.assertEqual(baseline["q_net2"].forward_count, 2)
        self.assertEqual(reused["q_net1"].forward_count, 1)
        self.assertEqual(reused["q_net2"].forward_count, 1)


if __name__ == "__main__":
    unittest.main()
