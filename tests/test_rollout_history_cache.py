import unittest
from unittest import mock

import torch

import driver
import parameter
import worker as worker_module
from model import PolicyNet
from models.rollout_history_cache import RolloutHistoryCache
from replay_buffer import (
    EpisodeTrajectory,
    TrajectoryStep,
    build_history_sequence,
)
from runner import Runner
from worker import Worker


NODE_DIM = 4
EMBEDDING_DIM = 32
SEQUENCE_LENGTH = 8
TOKEN_DIM = 32
CANDIDATE_COUNT = 5
NODE_COUNT = 9


def make_observation(step_index, valid_candidates):
    generator = torch.Generator().manual_seed(3100 + step_index)
    node_inputs = torch.randn(
        NODE_COUNT,
        NODE_DIM,
        generator=generator,
    )
    node_padding_mask = torch.zeros(
        1,
        NODE_COUNT,
        dtype=torch.bool,
    )
    edge_mask = torch.zeros(
        NODE_COUNT,
        NODE_COUNT,
        dtype=torch.bool,
    )
    current_index = torch.tensor([[0]], dtype=torch.long)
    candidate_indices = list(range(1, valid_candidates + 1))
    candidate_indices += [0] * (
        CANDIDATE_COUNT - valid_candidates
    )
    current_edge = torch.tensor(
        candidate_indices,
        dtype=torch.long,
    ).reshape(CANDIDATE_COUNT, 1)
    edge_padding_mask = torch.tensor(
        [False] * valid_candidates
        + [True] * (CANDIDATE_COUNT - valid_candidates),
        dtype=torch.bool,
    ).reshape(1, CANDIDATE_COUNT)
    return (
        node_inputs,
        node_padding_mask,
        edge_mask,
        current_index,
        current_edge,
        edge_padding_mask,
    )


def make_episode(episode_id, length):
    observations = [
        make_observation(
            step_index,
            valid_candidates=2 + step_index % 3,
        )
        for step_index in range(length + 1)
    ]
    trajectory = EpisodeTrajectory(episode_id=episode_id)
    for step_index in range(length):
        valid_candidates = 2 + step_index % 3
        trajectory.append(
            TrajectoryStep(
                observation=observations[step_index],
                action=step_index % valid_candidates,
                reward=0.25 * (step_index + 1),
                next_observation=observations[step_index + 1],
                robot_pose=torch.tensor(
                    [
                        float(step_index),
                        float(-step_index) / 2,
                        float(step_index) / 10,
                    ],
                    dtype=torch.float32,
                ),
                delta_coverage=0.01 * step_index,
                delta_frontier=step_index - 3,
                done=step_index == length - 1,
                step_index=step_index,
            )
        )
    return trajectory


def add_batch_dimension(observation, device):
    return tuple(
        component.unsqueeze(0).to(device)
        for component in observation
    )


def build_policy(device):
    return PolicyNet(
        NODE_DIM,
        EMBEDDING_DIM,
        use_mamba=True,
        mamba_seq_len=SEQUENCE_LENGTH,
        mamba_d_model=TOKEN_DIM,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        fusion_hidden_dim=EMBEDDING_DIM,
        action_embedding_dim=8,
    ).to(device).eval()


def build_cache():
    return RolloutHistoryCache(
        sequence_length=SEQUENCE_LENGTH,
        graph_embedding_dim=EMBEDDING_DIM,
        token_dim=TOKEN_DIM,
    )


def append_fake_entry(
    cache,
    episode_id,
    policy_version,
    transition_id,
    device="cpu",
):
    graph = torch.full(
        (EMBEDDING_DIM,),
        float(transition_id),
        device=device,
    )
    action = graph + 0.5
    token = torch.full(
        (TOKEN_DIM,),
        float(transition_id) + 1,
        device=device,
    )
    cache.append(
        episode_id=episode_id,
        policy_version=policy_version,
        transition_id=transition_id,
        graph_embedding=graph,
        action_feature=action,
        token=token,
    )


class RolloutHistoryCacheLifecycleTest(unittest.TestCase):
    def test_legacy_defaults_and_unavailable_learner_switch(self):
        self.assertIsInstance(parameter.USE_ROLLOUT_HISTORY_CACHE, bool)
        self.assertFalse(parameter.USE_LEARNER_HISTORY_DEDUP)
        driver.validate_history_optimization_config(
            use_mamba=True,
            use_rollout_history_cache=False,
            use_learner_history_dedup=False,
        )
        baseline_runner = Runner(
            meta_agent_id=0,
            use_mamba=False,
            use_rollout_history_cache=False,
        )
        self.assertIsNone(baseline_runner.rollout_history_cache)

        with self.assertRaisesRegex(
            ValueError,
            "requires USE_MAMBA=True",
        ):
            driver.validate_history_optimization_config(
                use_mamba=False,
                use_rollout_history_cache=True,
                use_learner_history_dedup=False,
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "same network parameters",
        ):
            driver.validate_history_optimization_config(
                use_mamba=True,
                use_rollout_history_cache=False,
                use_learner_history_dedup=True,
            )

    def test_padding_window_bound_and_detached_storage(self):
        cache = build_cache()
        cache.start_episode(
            episode_id=11,
            policy_version=4,
            device="cpu",
            dtype=torch.float32,
            use_mamba=True,
        )

        checked_lengths = {0, 1, 2, SEQUENCE_LENGTH - 1, SEQUENCE_LENGTH}
        for transition_id in range(10 * SEQUENCE_LENGTH + 1):
            sequence = cache.get_sequence(
                episode_id=11,
                policy_version=4,
                decision_step=transition_id,
                device="cpu",
                dtype=torch.float32,
                use_mamba=True,
            )
            if min(transition_id, SEQUENCE_LENGTH) in checked_lengths:
                valid_length = int((~sequence.padding_mask).sum().item())
                self.assertEqual(
                    valid_length,
                    min(transition_id, SEQUENCE_LENGTH),
                )
                self.assertEqual(
                    sequence.transition_ids[-valid_length:]
                    if valid_length
                    else (),
                    tuple(
                        range(
                            max(0, transition_id - SEQUENCE_LENGTH),
                            transition_id,
                        )
                    ),
                )
            append_fake_entry(
                cache,
                episode_id=11,
                policy_version=4,
                transition_id=transition_id,
            )
            self.assertLessEqual(len(cache), SEQUENCE_LENGTH)
            for entry in cache.entries:
                for tensor in (
                    entry.graph_embedding,
                    entry.action_feature,
                    entry.token,
                ):
                    self.assertFalse(tensor.requires_grad)
                    self.assertIsNone(tensor.grad_fn)

        cache.end_episode(11)
        self.assertFalse(cache.active)
        self.assertEqual(len(cache), 0)

    def test_episode_boundary_policy_update_and_discontinuity_invalidate(self):
        cache = build_cache()
        cache.start_episode(
            episode_id=21,
            policy_version=0,
            device="cpu",
            dtype=torch.float32,
            use_mamba=True,
        )
        append_fake_entry(cache, 21, 0, 0)
        cache.end_episode(21)
        cache.start_episode(
            episode_id=22,
            policy_version=0,
            device="cpu",
            dtype=torch.float32,
            use_mamba=True,
        )
        sequence = cache.get_sequence(
            episode_id=22,
            policy_version=0,
            decision_step=0,
            device="cpu",
            dtype=torch.float32,
            use_mamba=True,
        )
        self.assertTrue(sequence.padding_mask.all())
        self.assertNotIn(0, sequence.transition_ids)

        append_fake_entry(cache, 22, 0, 0)
        cache.on_policy_update(1)
        self.assertFalse(cache.active)
        self.assertEqual(cache.policy_version, 1)
        self.assertEqual(len(cache), 0)

        cache.start_episode(
            episode_id=22,
            policy_version=1,
            device="cpu",
            dtype=torch.float32,
            use_mamba=True,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "continuous transition ids",
        ):
            append_fake_entry(cache, 22, 1, 2)
        self.assertFalse(cache.active)
        self.assertEqual(len(cache), 0)

    def test_runner_weight_load_invalidates_cache(self):
        cache = build_cache()
        cache.start_episode(
            episode_id=31,
            policy_version=0,
            device="cpu",
            dtype=torch.float32,
            use_mamba=True,
        )
        append_fake_entry(cache, 31, 0, 0)

        runner = Runner.__new__(Runner)
        runner.local_network = torch.nn.Linear(3, 2)
        runner.policy_version = 0
        runner.rollout_history_cache = cache
        weights = {
            name: value.detach().clone()
            for name, value in runner.local_network.state_dict().items()
        }
        Runner.set_policy_net_weights(runner, weights)
        self.assertEqual(runner.policy_version, 1)
        self.assertFalse(cache.active)
        self.assertEqual(len(cache), 0)

    def test_configuration_device_and_dtype_changes_invalidate(self):
        cache = build_cache()
        with self.assertRaisesRegex(
            RuntimeError,
            "dimensions changed",
        ):
            cache.start_episode(
                episode_id=41,
                policy_version=0,
                device="cpu",
                dtype=torch.float32,
                use_mamba=True,
                sequence_length=SEQUENCE_LENGTH + 1,
                graph_embedding_dim=EMBEDDING_DIM,
                token_dim=TOKEN_DIM,
            )
        self.assertFalse(cache.active)

        cache.start_episode(
            episode_id=41,
            policy_version=0,
            device="cpu",
            dtype=torch.float32,
            use_mamba=True,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "metadata",
        ):
            cache.get_sequence(
                episode_id=41,
                policy_version=0,
                decision_step=0,
                device="cpu",
                dtype=torch.float64,
                use_mamba=True,
            )
        self.assertFalse(cache.active)


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA is required for Official Mamba rollout-cache equivalence",
)
class RolloutHistoryCacheEquivalenceTest(unittest.TestCase):
    device = torch.device("cuda")

    def test_cached_policy_forward_encodes_only_the_current_graph(self):
        policy = build_policy(self.device)
        policy_device = next(policy.parameters()).device
        episode = make_episode(episode_id=61, length=4)
        history = build_history_sequence(
            episode,
            sequence_length=SEQUENCE_LENGTH,
        ).to(policy_device)
        observation = add_batch_dimension(
            episode.steps[-1].next_observation,
            policy_device,
        )
        cache = build_cache()
        cache.start_episode(
            episode_id=61,
            policy_version=0,
            device=policy_device,
            dtype=torch.float32,
            use_mamba=True,
        )
        for transition_id in range(len(episode)):
            append_fake_entry(
                cache,
                episode_id=61,
                policy_version=0,
                transition_id=transition_id,
                device=policy_device,
            )
        cached_history = cache.get_sequence(
            episode_id=61,
            policy_version=0,
            decision_step=len(episode),
            device=policy_device,
            dtype=torch.float32,
            use_mamba=True,
        )

        original_encode_graph = policy.encode_graph
        with mock.patch.object(
            policy,
            "encode_graph",
            wraps=original_encode_graph,
        ) as cached_encode:
            policy(*observation, cached_history=cached_history)
        self.assertEqual(cached_encode.call_count, 1)

        with mock.patch.object(
            policy,
            "encode_graph",
            wraps=original_encode_graph,
        ) as legacy_encode:
            policy(*observation, history_batch=history)
        self.assertGreater(legacy_encode.call_count, 1)

    def test_stepwise_legacy_and_cached_outputs_are_equivalent(self):
        torch.manual_seed(20260724)
        policy = build_policy(self.device)
        # Rollout finalization encodes one graph at a time. Matching that
        # batch shape removes harmless GEMM batching round-off from this
        # cache-equivalence assertion.
        policy.history_encoder.graph_batch_size = 1
        policy_device = next(policy.parameters()).device
        source_episode = make_episode(
            episode_id=71,
            length=SEQUENCE_LENGTH + 1,
        )
        online_prefix = EpisodeTrajectory(episode_id=71)
        cache = build_cache()
        cache.start_episode(
            episode_id=71,
            policy_version=0,
            device=policy_device,
            dtype=torch.float32,
            use_mamba=True,
        )
        lengths_seen = set()

        for decision_step in range(len(source_episode) + 1):
            observation = (
                source_episode.steps[decision_step].observation
                if decision_step < len(source_episode)
                else source_episode.steps[-1].next_observation
            )
            observation = add_batch_dimension(
                observation,
                self.device,
            )
            legacy_history = build_history_sequence(
                online_prefix,
                sequence_length=SEQUENCE_LENGTH,
            )
            if legacy_history is not None:
                legacy_history = legacy_history.to(self.device)
            cached_history = cache.get_sequence(
                episode_id=71,
                policy_version=0,
                decision_step=decision_step,
                device=policy_device,
                dtype=torch.float32,
                use_mamba=True,
            )
            valid_length = min(decision_step, SEQUENCE_LENGTH)
            lengths_seen.add(decision_step)

            if legacy_history is None:
                self.assertEqual(valid_length, 0)
                self.assertTrue(cached_history.padding_mask.all())
                legacy_tokens = cached_history.tokens.new_zeros(
                    cached_history.tokens.shape
                )
            else:
                (
                    legacy_graph,
                    legacy_action,
                ) = (
                    policy.history_encoder
                    .graph_features_from_observations(
                        policy,
                        legacy_history.observations,
                        legacy_history.actions,
                    )
                )
                legacy_tokens = policy.history_encoder.encode_sequence(
                    policy,
                    legacy_history,
                )
                cached_entries = cache.entries
                torch.testing.assert_close(
                    torch.stack(
                        [
                            entry.graph_embedding
                            for entry in cached_entries
                        ],
                        dim=0,
                    ),
                    legacy_graph[0, -valid_length:, :],
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    torch.stack(
                        [
                            entry.action_feature
                            for entry in cached_entries
                        ],
                        dim=0,
                    ),
                    legacy_action[0, -valid_length:, :],
                    rtol=1e-5,
                    atol=1e-6,
                )
            torch.testing.assert_close(
                cached_history.tokens,
                legacy_tokens,
                rtol=1e-5,
                atol=1e-6,
            )
            if legacy_history is not None:
                self.assertTrue(
                    torch.equal(
                        cached_history.padding_mask,
                        legacy_history.padding_mask,
                    )
                )

            legacy_temporal = policy.encode_temporal_history(
                history_batch=legacy_history,
                batch_size=1,
                device=policy_device,
                dtype=torch.float32,
            )
            cached_temporal = policy.encode_temporal_history(
                history_batch=None,
                batch_size=1,
                device=policy_device,
                dtype=torch.float32,
                cached_history=cached_history,
            )
            torch.testing.assert_close(
                cached_temporal,
                legacy_temporal,
                rtol=2e-5,
                atol=2e-6,
            )

            enhanced = policy.encode_graph(
                observation[0],
                observation[1],
                observation[2],
            )
            current, decoded = policy.decode_state(
                enhanced,
                observation[3],
                observation[1],
            )
            graph_feature = policy.current_embedding(
                torch.cat((decoded, current), dim=-1)
            )
            legacy_fused = policy.fuse_state_feature(
                graph_feature,
                legacy_history,
            )
            cached_fused = policy.fuse_state_feature(
                graph_feature,
                None,
                cached_history=cached_history,
            )
            torch.testing.assert_close(
                cached_fused,
                legacy_fused,
                rtol=2e-5,
                atol=2e-6,
            )

            legacy_logp = policy(
                *observation,
                history_batch=legacy_history,
            )
            cached_logp, rollout_context = policy(
                *observation,
                cached_history=cached_history,
                return_rollout_context=True,
            )
            torch.testing.assert_close(
                cached_logp,
                legacy_logp,
                rtol=2e-5,
                atol=2e-6,
            )
            torch.testing.assert_close(
                cached_logp.exp(),
                legacy_logp.exp(),
                rtol=2e-5,
                atol=2e-6,
            )
            action_mask = observation[5].bool().squeeze(1)
            self.assertTrue(
                torch.all(cached_logp.exp()[action_mask] == 0)
            )
            legal_actions = torch.nonzero(
                ~action_mask[0],
                as_tuple=False,
            ).squeeze(1).tolist()
            self.assertIn(
                int(cached_logp.argmax(dim=-1).item()),
                legal_actions,
            )
            self.assertTrue(
                torch.equal(
                    torch.isfinite(cached_logp),
                    torch.isfinite(legacy_logp),
                )
            )
            self.assertEqual(
                int(cached_logp.argmax(dim=-1).item()),
                int(legacy_logp.argmax(dim=-1).item()),
            )

            if decision_step == len(source_episode):
                break
            step = source_episode.steps[decision_step]
            (
                graph_embedding,
                action_feature,
                token,
            ) = policy.finalize_rollout_history_token(
                rollout_context=rollout_context,
                action_index=step.action,
                robot_pose=step.robot_pose,
                reward=step.reward,
                delta_coverage=step.delta_coverage,
                delta_frontier=step.delta_frontier,
            )
            cache.append(
                episode_id=71,
                policy_version=0,
                transition_id=decision_step,
                graph_embedding=graph_embedding,
                action_feature=action_feature,
                token=token,
            )
            online_prefix.append(step)

        for required_length in (
            0,
            1,
            2,
            SEQUENCE_LENGTH - 1,
            SEQUENCE_LENGTH,
            SEQUENCE_LENGTH + 1,
        ):
            self.assertIn(required_length, lengths_seen)

    def test_cached_worker_completes_short_online_rollout_and_resets(self):
        policy = PolicyNet(
            parameter.NODE_INPUT_DIM,
            parameter.EMBEDDING_DIM,
            use_mamba=True,
            mamba_seq_len=parameter.MAMBA_SEQ_LEN,
            mamba_d_model=parameter.MAMBA_D_MODEL,
            mamba_d_state=parameter.MAMBA_D_STATE,
            mamba_d_conv=parameter.MAMBA_D_CONV,
            mamba_expand=parameter.MAMBA_EXPAND,
            fusion_hidden_dim=parameter.FUSION_HIDDEN_DIM,
            action_embedding_dim=parameter.ACTION_EMBEDDING_DIM,
        ).to(self.device)
        cache = RolloutHistoryCache(
            sequence_length=parameter.MAMBA_SEQ_LEN,
            graph_embedding_dim=parameter.EMBEDDING_DIM,
            token_dim=parameter.MAMBA_D_MODEL,
        )
        rollout_worker = Worker(
            meta_agent_id=0,
            policy_net=policy,
            global_step=88,
            device=self.device,
            save_image=False,
            rollout_history_cache=cache,
            policy_version=3,
        )

        with mock.patch.object(
            worker_module,
            "MAX_EPISODE_STEP",
            3,
        ):
            rollout_worker.run_episode()

        self.assertEqual(len(rollout_worker.trajectory), 3)
        self.assertFalse(cache.active)
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
