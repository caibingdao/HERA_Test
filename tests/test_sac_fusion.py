import io
import math
import subprocess
import sys
import unittest
from unittest import mock

import torch
import torch.optim as optim

import driver
import worker as worker_module
from model import PolicyNet, QNet
from models.mamba_encoder import OfficialMamba
from parameter import (
    ACTION_EMBEDDING_DIM,
    EMBEDDING_DIM as PROJECT_EMBEDDING_DIM,
    FUSION_HIDDEN_DIM,
    MAMBA_D_CONV,
    MAMBA_D_MODEL,
    MAMBA_D_STATE,
    MAMBA_EXPAND,
    MAMBA_SEQ_LEN,
    NODE_INPUT_DIM,
)
from replay_buffer import EpisodeTrajectory, TrajectoryReplayBuffer, TrajectoryStep
from worker import Worker


NODE_DIM = 4
EMBEDDING_DIM = 128
NODE_COUNT = 6
CANDIDATE_COUNT = 4
SEQUENCE_LENGTH = 8


def make_observation(value, legal_candidates):
    node_inputs = torch.arange(
        NODE_COUNT * NODE_DIM,
        dtype=torch.float32,
    ).reshape(NODE_COUNT, NODE_DIM)
    node_inputs = node_inputs / 100 + value
    node_padding_mask = torch.zeros((1, NODE_COUNT), dtype=torch.bool)
    node_padding_mask[0, -1] = True
    edge_mask = torch.zeros((NODE_COUNT, NODE_COUNT), dtype=torch.bool)
    current_index = torch.zeros((1, 1), dtype=torch.long)
    current_edge = torch.tensor([1, 2, 3, 0], dtype=torch.long).reshape(
        CANDIDATE_COUNT,
        1,
    )
    edge_padding_mask = torch.ones(
        (1, CANDIDATE_COUNT),
        dtype=torch.bool,
    )
    edge_padding_mask[0, :legal_candidates] = False
    return (
        node_inputs,
        node_padding_mask,
        edge_mask,
        current_index,
        current_edge,
        edge_padding_mask,
    )


def make_episode(episode_id, offset, legal_candidates):
    trajectory = EpisodeTrajectory(episode_id=episode_id)
    for step_index in range(6):
        value = offset + step_index / 10
        trajectory.append(
            TrajectoryStep(
                observation=make_observation(value, legal_candidates),
                action=step_index % legal_candidates,
                reward=0.1 * (step_index + 1),
                next_observation=make_observation(
                    value + 0.05,
                    legal_candidates,
                ),
                robot_pose=torch.tensor(
                    [value, -value, step_index / 10],
                    dtype=torch.float32,
                ),
                delta_coverage=0.01 * step_index,
                delta_frontier=step_index - 2,
                done=step_index == 5,
                step_index=step_index,
            )
        )
    return trajectory


def make_aligned_batch():
    first = make_episode(10, 1.0, legal_candidates=2)
    second = make_episode(20, 2.0, legal_candidates=3)
    return TrajectoryReplayBuffer._collate_aligned_transitions(
        [(first, 3), (second, 4)],
        sequence_length=SEQUENCE_LENGTH,
    )


def network_options(use_mamba):
    return {
        "use_mamba": use_mamba,
        "mamba_seq_len": SEQUENCE_LENGTH,
        "mamba_d_model": EMBEDDING_DIM,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "fusion_hidden_dim": EMBEDDING_DIM,
        "action_embedding_dim": 16,
    }


def parameter_snapshot(module):
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def any_parameter_changed(before, module):
    return any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in module.named_parameters()
    )


def assert_module_has_finite_nonzero_gradients(testcase, module):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    testcase.assertTrue(gradients)
    testcase.assertTrue(
        all(torch.isfinite(gradient).all() for gradient in gradients)
    )
    total = sum(float(gradient.abs().sum()) for gradient in gradients)
    testcase.assertGreater(total, 0)


def run_one_update(testcase, use_mamba, device):
    batch = make_aligned_batch().to(device)
    options = network_options(use_mamba)
    policy = PolicyNet(NODE_DIM, EMBEDDING_DIM, **options).to(device)
    q1 = QNet(NODE_DIM, EMBEDDING_DIM, **options).to(device)
    q2 = QNet(NODE_DIM, EMBEDDING_DIM, **options).to(device)
    target_q1 = QNet(NODE_DIM, EMBEDDING_DIM, **options).to(device)
    target_q2 = QNet(NODE_DIM, EMBEDDING_DIM, **options).to(device)
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

    snapshots = {
        "policy": parameter_snapshot(policy),
        "q1": parameter_snapshot(q1),
        "q2": parameter_snapshot(q2),
    }
    if device.type == "cuda":
        update_policy = torch.nn.DataParallel(policy)
        update_q1 = torch.nn.DataParallel(q1)
        update_q2 = torch.nn.DataParallel(q2)
        update_target_q1 = torch.nn.DataParallel(target_q1)
        update_target_q2 = torch.nn.DataParallel(target_q2)
    else:
        update_policy = policy
        update_q1 = q1
        update_q2 = q2
        update_target_q1 = target_q1
        update_target_q2 = target_q2
    current_history = batch.current_history if use_mamba else None
    next_history = batch.next_history if use_mamba else None
    result = driver.sac_update(
        policy_net=update_policy,
        q_net1=update_q1,
        q_net2=update_q2,
        target_q_net1=update_target_q1,
        target_q_net2=update_target_q2,
        policy_optimizer=policy_optimizer,
        q_net1_optimizer=q1_optimizer,
        q_net2_optimizer=q2_optimizer,
        log_alpha=log_alpha,
        log_alpha_optimizer=alpha_optimizer,
        entropy_target=0.05 * (-math.log(1 / CANDIDATE_COUNT)),
        observation=list(batch.observations),
        action=batch.actions,
        reward=batch.rewards,
        done=batch.dones,
        next_observation=list(batch.next_observations),
        current_history=current_history,
        next_history=next_history,
    )

    losses = torch.stack(
        [
            result["policy_loss"].detach(),
            result["q1_loss"].detach(),
            result["q2_loss"].detach(),
            result["alpha_loss"].detach(),
        ]
    )
    testcase.assertTrue(torch.isfinite(losses).all())
    testcase.assertTrue(any_parameter_changed(snapshots["policy"], policy))
    testcase.assertTrue(any_parameter_changed(snapshots["q1"], q1))
    testcase.assertTrue(any_parameter_changed(snapshots["q2"], q2))
    return batch, policy, q1, q2, target_q1, target_q2, result


class BaselineRegressionTest(unittest.TestCase):
    def test_baseline_update_and_variable_neighbor_mask(self):
        torch.manual_seed(3)
        batch = make_aligned_batch()
        policy = PolicyNet(
            NODE_DIM,
            EMBEDDING_DIM,
            **network_options(False),
        )
        logp = policy(*batch.observations)

        self.assertEqual(logp.shape, (2, CANDIDATE_COUNT))
        self.assertFalse(hasattr(policy, "mamba_encoder"))
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue(torch.all(logp[0, 2:].exp() == 0))
        self.assertTrue(torch.all(logp[1, 3:].exp() == 0))
        self.assertAlmostEqual(
            float(logp[0, :2].exp().sum()),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(logp[1, :3].exp().sum()),
            1.0,
            places=5,
        )

        _, policy, q1, q2, _, _, _ = run_one_update(
            self,
            use_mamba=False,
            device=torch.device("cpu"),
        )
        assert_module_has_finite_nonzero_gradients(self, policy.pointer)
        assert_module_has_finite_nonzero_gradients(self, q1.q_values_layer)
        assert_module_has_finite_nonzero_gradients(self, q2.q_values_layer)

    def test_checkpoint_mode_mismatch_is_explicit(self):
        with mock.patch.object(driver, "USE_MAMBA", False):
            driver.validate_checkpoint_mode({})
        with mock.patch.object(driver, "USE_MAMBA", True):
            with self.assertRaisesRegex(
                RuntimeError,
                "Checkpoint/model mode mismatch",
            ):
                driver.validate_checkpoint_mode({})


@unittest.skipUnless(
    OfficialMamba is not None and torch.cuda.is_available(),
    "Official mamba-ssm and CUDA are required",
)
class MambaFusionSACIntegrationTest(unittest.TestCase):
    device = torch.device("cuda")

    def test_fusion_shapes_sac_gradients_target_update_and_entry_smoke(self):
        torch.manual_seed(7)
        batch = make_aligned_batch().to(self.device)
        options = network_options(True)
        policy_for_shapes = PolicyNet(
            NODE_DIM,
            EMBEDDING_DIM,
            **options,
        ).to(self.device)

        enhanced = policy_for_shapes.encode_graph(
            batch.observations[0],
            batch.observations[1],
            batch.observations[2],
        )
        current_node, enhanced_current = policy_for_shapes.decode_state(
            enhanced,
            batch.observations[3],
            batch.observations[1],
        )
        graph_feature = policy_for_shapes.current_embedding(
            torch.cat((enhanced_current, current_node), dim=-1)
        )
        temporal_feature = policy_for_shapes.encode_temporal_history(
            batch.current_history,
            batch_size=batch.batch_size,
            device=self.device,
            dtype=graph_feature.dtype,
        )
        fused_feature = policy_for_shapes.fuse_state_feature(
            graph_feature,
            batch.current_history,
        )
        actor_output = policy_for_shapes(
            *batch.observations,
            history_batch=batch.current_history,
        )
        data_parallel_output = torch.nn.DataParallel(policy_for_shapes)(
            *batch.observations,
            history_batch=batch.current_history,
        )

        self.assertEqual(graph_feature.shape, (2, 1, EMBEDDING_DIM))
        self.assertEqual(temporal_feature.shape, (2, EMBEDDING_DIM))
        self.assertEqual(fused_feature.shape, (2, 1, EMBEDDING_DIM))
        self.assertEqual(actor_output.shape, (2, CANDIDATE_COUNT))
        self.assertEqual(data_parallel_output.shape, actor_output.shape)
        torch.testing.assert_close(data_parallel_output, actor_output)
        self.assertTrue(torch.isfinite(actor_output).all())
        self.assertTrue(torch.all(actor_output[0, 2:].exp() == 0))
        self.assertTrue(torch.all(actor_output[1, 3:].exp() == 0))

        (
            batch,
            policy,
            q1,
            q2,
            target_q1,
            target_q2,
            result,
        ) = run_one_update(self, use_mamba=True, device=self.device)

        q1_output = q1(
            *batch.observations,
            history_batch=batch.current_history,
        )
        q2_output = q2(
            *batch.observations,
            history_batch=batch.current_history,
        )
        target_output = target_q1(
            *batch.next_observations,
            history_batch=batch.next_history,
        )
        self.assertEqual(q1_output.shape, (2, CANDIDATE_COUNT, 1))
        self.assertEqual(q2_output.shape, (2, CANDIDATE_COUNT, 1))
        self.assertEqual(target_output.shape, (2, CANDIDATE_COUNT, 1))
        self.assertTrue(torch.isfinite(q1_output).all())
        self.assertTrue(torch.isfinite(q2_output).all())
        self.assertTrue(torch.isfinite(target_output).all())

        for network in (policy, q1, q2):
            assert_module_has_finite_nonzero_gradients(
                self,
                network.history_encoder,
            )
            assert_module_has_finite_nonzero_gradients(
                self,
                network.mamba_encoder,
            )
            assert_module_has_finite_nonzero_gradients(
                self,
                network.graph_mamba_fusion,
            )
        assert_module_has_finite_nonzero_gradients(self, policy.pointer)
        assert_module_has_finite_nonzero_gradients(self, q1.q_values_layer)
        assert_module_has_finite_nonzero_gradients(self, q2.q_values_layer)

        target_q1.load_state_dict(q1.state_dict())
        target_q2.load_state_dict(q2.state_dict())
        for online, target in ((q1, target_q1), (q2, target_q2)):
            for prefix in (
                "history_encoder.",
                "mamba_encoder.",
                "graph_mamba_fusion.",
            ):
                online_values = {
                    name: value
                    for name, value in online.state_dict().items()
                    if name.startswith(prefix)
                }
                target_values = {
                    name: value
                    for name, value in target.state_dict().items()
                    if name.startswith(prefix)
                }
                self.assertTrue(online_values)
                self.assertEqual(online_values.keys(), target_values.keys())
                for name in online_values:
                    torch.testing.assert_close(
                        online_values[name],
                        target_values[name],
                    )

        checkpoint_log_alpha = torch.tensor(
            [-2.0],
            device=self.device,
            requires_grad=True,
        )
        checkpoint_policy_optimizer = optim.Adam(
            policy.parameters(),
            lr=1e-5,
        )
        checkpoint_q1_optimizer = optim.Adam(q1.parameters(), lr=1e-5)
        checkpoint_q2_optimizer = optim.Adam(q2.parameters(), lr=1e-5)
        checkpoint_alpha_optimizer = optim.Adam(
            [checkpoint_log_alpha],
            lr=1e-4,
        )
        with mock.patch.object(driver, "USE_MAMBA", True):
            checkpoint = driver.build_checkpoint(
                policy_net=policy,
                q_net1=q1,
                q_net2=q2,
                target_q_net1=target_q1,
                target_q_net2=target_q2,
                log_alpha=checkpoint_log_alpha,
                policy_optimizer=checkpoint_policy_optimizer,
                q_net1_optimizer=checkpoint_q1_optimizer,
                q_net2_optimizer=checkpoint_q2_optimizer,
                log_alpha_optimizer=checkpoint_alpha_optimizer,
                episode=123,
                target_q_update_counter=17,
            )
            driver.validate_checkpoint_mode(checkpoint)

        for key in (
            "policy_model",
            "q_net1_model",
            "q_net2_model",
            "target_q_net1_model",
            "target_q_net2_model",
            "policy_optimizer",
            "q_net1_optimizer",
            "q_net2_optimizer",
            "log_alpha_optimizer",
            "log_alpha",
            "mamba_config",
        ):
            self.assertIn(key, checkpoint)

        checkpoint_bytes = io.BytesIO()
        torch.save(checkpoint, checkpoint_bytes)
        checkpoint_bytes.seek(0)
        restored = torch.load(
            checkpoint_bytes,
            map_location=self.device,
            weights_only=False,
        )
        restored_policy = PolicyNet(
            NODE_DIM,
            EMBEDDING_DIM,
            **network_options(True),
        ).to(self.device)
        restored_q1 = QNet(
            NODE_DIM,
            EMBEDDING_DIM,
            **network_options(True),
        ).to(self.device)
        restored_target_q1 = QNet(
            NODE_DIM,
            EMBEDDING_DIM,
            **network_options(True),
        ).to(self.device)
        restored_policy.load_state_dict(restored["policy_model"])
        restored_q1.load_state_dict(restored["q_net1_model"])
        restored_target_q1.load_state_dict(
            restored["target_q_net1_model"]
        )

        self.assertTrue(torch.isfinite(result["value_prime"]).all())

    def test_ray_training_entry_initializes_mamba_runner(self):
        script = """
import parameter
parameter.USE_MAMBA = True

import ray
from runner import RLRunner

ray.init(
    num_cpus=1,
    num_gpus=1,
    include_dashboard=False,
    log_to_driver=False,
)
try:
    runner = RLRunner.remote(0, use_mamba=True)
    weights = ray.get(runner.get_weights.remote(), timeout=60)
    assert any(name.startswith("mamba_encoder.") for name in weights)
    assert any(name.startswith("history_encoder.") for name in weights)
    assert any(name.startswith("graph_mamba_fusion.") for name in weights)
finally:
    ray.shutdown()
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_short_online_worker_rollout_uses_mamba_history(self):
        policy = PolicyNet(
            NODE_INPUT_DIM,
            PROJECT_EMBEDDING_DIM,
            use_mamba=True,
            mamba_seq_len=MAMBA_SEQ_LEN,
            mamba_d_model=MAMBA_D_MODEL,
            mamba_d_state=MAMBA_D_STATE,
            mamba_d_conv=MAMBA_D_CONV,
            mamba_expand=MAMBA_EXPAND,
            fusion_hidden_dim=FUSION_HIDDEN_DIM,
            action_embedding_dim=ACTION_EMBEDDING_DIM,
        ).to(self.device)
        rollout_worker = Worker(
            meta_agent_id=0,
            policy_net=policy,
            global_step=77,
            device=self.device,
            save_image=False,
        )

        with mock.patch.object(
            worker_module,
            "MAX_EPISODE_STEP",
            3,
        ):
            rollout_worker.run_episode()

        self.assertEqual(len(rollout_worker.trajectory), 3)
        self.assertEqual(len(rollout_worker.episode_buffer[0]), 0)
        self.assertTrue(
            all(
                step.observation[0].device.type == "cpu"
                for step in rollout_worker.trajectory.steps
            )
        )


if __name__ == "__main__":
    unittest.main()
