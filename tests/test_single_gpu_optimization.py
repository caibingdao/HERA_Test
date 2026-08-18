import subprocess
import sys
import unittest
from collections import OrderedDict
from unittest import mock

import torch
import torch.nn as nn

import driver
import parameter
from model import PolicyNet
from parameter import (
    NUM_META_AGENT,
    RUNNER_GPU_FRACTION,
)
from runner import RLRunner, validate_runner_resource_config


class FakeRunnerClass:
    calls = []

    @classmethod
    def remote(
        cls,
        meta_agent_id,
        use_mamba,
        use_rollout_history_cache,
        map_list=None,
        training_seed=None,
    ):
        cls.calls.append(
            (
                meta_agent_id,
                use_mamba,
                use_rollout_history_cache,
                map_list,
                training_seed,
            )
        )
        return (
            meta_agent_id,
            use_mamba,
            use_rollout_history_cache,
            map_list,
            training_seed,
        )


class RunnerConfigurationTest(unittest.TestCase):
    def setUp(self):
        FakeRunnerClass.calls = []

    def test_default_and_supported_runner_counts(self):
        self.assertGreater(NUM_META_AGENT, 0)
        self.assertGreater(RUNNER_GPU_FRACTION, 0)
        self.assertLessEqual(
            NUM_META_AGENT * RUNNER_GPU_FRACTION,
            1.0,
        )
        self.assertEqual(
            RLRunner._default_options["num_gpus"],
            RUNNER_GPU_FRACTION,
        )

        for count in (1, 2, 4):
            with self.subTest(count=count):
                total_gpu = validate_runner_resource_config(
                    count,
                    RUNNER_GPU_FRACTION,
                )
                self.assertEqual(
                    total_gpu,
                    count * RUNNER_GPU_FRACTION,
                )
                FakeRunnerClass.calls = []
                actors = driver.create_meta_agents(
                    num_meta_agents=count,
                    use_mamba=True,
                    runner_class=FakeRunnerClass,
                )
                self.assertEqual(len(actors), count)
                self.assertEqual(
                    FakeRunnerClass.calls,
                    [
                        (index, True, False, None, None)
                        if not parameter.USE_ROLLOUT_HISTORY_CACHE
                        else (index, True, True, None, None)
                        for index in range(count)
                    ],
                )

    def test_invalid_runner_gpu_total_is_explicit(self):
        with self.assertRaisesRegex(
            ValueError,
            r"4.*0\.3.*1\.2.*<= 1\.0",
        ):
            validate_runner_resource_config(4, 0.30)

        for count, fraction in ((0, 0.25), (2, 0), (True, 0.25)):
            with self.subTest(count=count, fraction=fraction):
                with self.assertRaises(ValueError):
                    validate_runner_resource_config(count, fraction)

    def test_baseline_runner_declares_cpu_and_zero_active_gpu(self):
        script = """
import parameter
parameter.USE_MAMBA = False
parameter.USE_GPU = False

from runner import ACTIVE_RUNNER_GPU_FRACTION, RLRunner, Runner

assert ACTIVE_RUNNER_GPU_FRACTION == 0.0
assert RLRunner._default_options["num_gpus"] == 0.0
runner = Runner(meta_agent_id=0, use_mamba=False)
assert runner.device.type == "cpu"
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_data_parallel_requires_opt_in_and_multiple_gpus(self):
        networks = tuple(nn.Linear(2, 2) for _ in range(5))
        cuda_device = torch.device("cuda")

        with mock.patch.object(
            driver.torch.cuda,
            "device_count",
            return_value=1,
        ):
            prepared = driver.prepare_training_networks(
                networks,
                device=cuda_device,
                enable_data_parallel=True,
            )
        self.assertEqual(prepared, networks)

        with mock.patch.object(
            driver.torch.cuda,
            "device_count",
            return_value=4,
        ):
            prepared = driver.prepare_training_networks(
                networks,
                device=cuda_device,
                enable_data_parallel=False,
            )
        self.assertEqual(prepared, networks)

        with (
            mock.patch.object(
                driver.torch.cuda,
                "device_count",
                return_value=2,
            ),
            mock.patch.object(
                driver.nn,
                "DataParallel",
                side_effect=lambda network: ("wrapped", network),
            ) as data_parallel,
        ):
            prepared = driver.prepare_training_networks(
                networks,
                device=cuda_device,
                enable_data_parallel=True,
            )
        self.assertEqual(data_parallel.call_count, 5)
        self.assertTrue(
            all(item[0] == "wrapped" for item in prepared)
        )

    @unittest.skipUnless(
        torch.cuda.device_count() == 1,
        "This regression specifically verifies the single-GPU path",
    )
    def test_actual_single_gpu_models_remain_unwrapped(self):
        networks = tuple(nn.Linear(2, 2).cuda() for _ in range(5))
        prepared = driver.prepare_training_networks(
            networks,
            device=torch.device("cuda"),
            enable_data_parallel=True,
        )
        for original, training_network in zip(networks, prepared):
            self.assertIs(training_network, original)
            self.assertNotIsInstance(
                training_network,
                nn.DataParallel,
            )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required to verify that weight export does not move the model",
    )
    def test_policy_weight_export_does_not_move_model_to_cpu(self):
        policy = nn.Linear(3, 2).cuda()
        weights = driver.policy_weights_for_workers(policy)
        self.assertEqual(policy.weight.device.type, "cuda")
        self.assertTrue(weights)
        self.assertTrue(
            all(value.device.type == "cpu" for value in weights.values())
        )


class CheckpointCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.options = {
            "use_mamba": True,
            "mamba_seq_len": 8,
            "mamba_d_model": 128,
            "mamba_d_state": 16,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "fusion_hidden_dim": 128,
            "action_embedding_dim": 16,
        }

    def build_policy(self):
        return PolicyNet(4, 128, **self.options)

    def assert_temporal_parameters_match(self, source, restored):
        source_state = source.state_dict()
        restored_state = restored.state_dict()
        for prefix in (
            "history_encoder.",
            "mamba_encoder.",
            "graph_mamba_fusion.",
        ):
            names = [
                name for name in source_state
                if name.startswith(prefix)
            ]
            self.assertTrue(names)
            for name in names:
                torch.testing.assert_close(
                    source_state[name],
                    restored_state[name],
                )

    def test_standard_and_module_prefixed_checkpoints_load_strictly(self):
        torch.manual_seed(19)
        source = self.build_policy()
        standard_state = source.state_dict()

        restored_standard = self.build_policy()
        driver.load_model_state_dict(
            restored_standard,
            standard_state,
            "policy",
        )
        self.assert_temporal_parameters_match(
            source,
            restored_standard,
        )

        prefixed_state = OrderedDict(
            (f"module.{name}", value.clone())
            for name, value in standard_state.items()
        )
        restored_prefixed = self.build_policy()
        driver.load_model_state_dict(
            restored_prefixed,
            prefixed_state,
            "policy",
        )
        self.assert_temporal_parameters_match(
            source,
            restored_prefixed,
        )

    def test_checkpoint_key_mismatches_are_not_silent(self):
        source = self.build_policy()
        state = OrderedDict(source.state_dict())
        missing_name = next(
            name for name in state
            if name.startswith("mamba_encoder.")
        )
        state.pop(missing_name)
        with self.assertRaisesRegex(
            RuntimeError,
            rf"missing_keys=.*{missing_name}",
        ):
            driver.load_model_state_dict(
                self.build_policy(),
                state,
                "policy",
            )

        state = OrderedDict(source.state_dict())
        state["unexpected.parameter"] = torch.zeros(1)
        with self.assertRaisesRegex(
            RuntimeError,
            r"unexpected_keys=.*unexpected\.parameter",
        ):
            driver.load_model_state_dict(
                self.build_policy(),
                state,
                "policy",
            )

    def test_saved_state_dict_uses_standard_keys(self):
        wrapped = nn.DataParallel(nn.Linear(3, 2))
        saved_state = driver.standard_model_state_dict(wrapped)
        self.assertTrue(saved_state)
        self.assertTrue(
            all(not name.startswith("module.") for name in saved_state)
        )


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA is required for the Ray Mamba resource smoke test",
)
class RayRunnerResourceSmokeTest(unittest.TestCase):
    def test_two_mamba_runners_reserve_half_gpu(self):
        script = """
import os
os.environ["PYTHONPATH"] = os.getcwd()
import ray

from parameter import RUNNER_GPU_FRACTION
from runner import RLRunner

ray.init(
    num_cpus=2,
    num_gpus=1,
    object_store_memory=512 * 1024 ** 2,
    include_dashboard=False,
    log_to_driver=False,
)
actors = []
try:
    actors = [
        RLRunner.remote(index, use_mamba=True)
        for index in range(2)
    ]
    runtime = ray.get(
        [actor.get_runtime_resources.remote() for actor in actors],
        timeout=60,
    )
    assert all(item["device_type"] == "cuda" for item in runtime), runtime
    assert all(item["ray_gpu_ids"] == [0] for item in runtime), runtime
    available_gpu = ray.available_resources().get("GPU", 0.0)
    expected_gpu = 1.0 - 2 * RUNNER_GPU_FRACTION
    assert abs(available_gpu - expected_gpu) < 1e-6, (
        available_gpu,
        expected_gpu,
    )
finally:
    for actor in actors:
        ray.kill(actor)
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


if __name__ == "__main__":
    unittest.main()
