import random
import sys
import types

import numpy as np
import pytest
import torch

import parameter
from experiment_utils import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from model import PolicyNet
from scripts import train_on_split


def _untrained_policy_state(seed):
    torch.manual_seed(seed)
    policy = PolicyNet(
        parameter.NODE_INPUT_DIM,
        parameter.EMBEDDING_DIM,
        use_mamba=False,
    )
    return {
        name: value.detach().clone()
        for name, value in policy.state_dict().items()
    }


def test_pure_ariadne_defaults_to_fresh_initialization():
    assert parameter.MODEL_SOURCE_REF == "ariadne_mamba_1.1_trained"
    assert (
        parameter.MODEL_SOURCE_COMMIT
        == "1e8316b9739e2b46b8aaaa4d2be69f175206f207"
    )
    assert parameter.USE_MAMBA is False
    assert parameter.RUN_FOLDER_NAME == "ariadne_no_mamba"
    assert parameter.LOAD_MODEL is False
    assert parameter.COLLECT_TRAJECTORY is False
    assert parameter.USE_ROLLOUT_HISTORY_CACHE is False


def test_random_initialization_is_seeded_but_not_pretrained():
    first = _untrained_policy_state(123)
    repeated = _untrained_policy_state(123)
    different = _untrained_policy_state(124)

    assert all(
        torch.equal(first[name], repeated[name])
        for name in first
    )
    assert any(
        not torch.equal(first[name], different[name])
        for name in first
        if first[name].is_floating_point()
    )


def test_split_training_wrapper_forces_fresh_run(tmp_path, monkeypatch):
    calls = []
    fake_driver = types.ModuleType("driver")

    def fake_train(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "final_checkpoint.pth")

    fake_driver.main = fake_train
    monkeypatch.setitem(sys.modules, "driver", fake_driver)

    exit_code = train_on_split.main(
        [
            "--split-file",
            str(tmp_path / "split.json"),
            "--training-seed",
            "7",
            "--total-env-steps",
            "64",
            "--output-dir",
            str(tmp_path / "fresh_output"),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["load_model"] is False
    assert calls[0]["resume_checkpoint"] is None


def test_split_training_wrapper_resumes_explicit_checkpoint(
    tmp_path,
    monkeypatch,
):
    calls = []
    fake_driver = types.ModuleType("driver")

    def fake_train(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "run" / "final_checkpoint.pth")

    fake_driver.main = fake_train
    monkeypatch.setitem(sys.modules, "driver", fake_driver)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.pth"
    checkpoint.write_bytes(b"full training checkpoint placeholder")

    exit_code = train_on_split.main(
        [
            "--split-file",
            str(tmp_path / "split.json"),
            "--training-seed",
            "7",
            "--total-env-steps",
            "64",
            "--resume-checkpoint",
            str(checkpoint),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["load_model"] is True
    assert calls[0]["resume_checkpoint"] == checkpoint.resolve()
    assert calls[0]["output_dir"] == run_dir.resolve()


def test_resume_requires_checkpoint_parent_as_output_dir(
    tmp_path,
    monkeypatch,
):
    fake_driver = types.ModuleType("driver")
    fake_driver.main = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "driver", fake_driver)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    other_output = tmp_path / "other"
    other_output.mkdir()

    with pytest.raises(ValueError, match="checkpoint's parent"):
        train_on_split.main(
            [
                "--split-file",
                str(tmp_path / "split.json"),
                "--total-env-steps",
                "64",
                "--output-dir",
                str(other_output),
                "--resume-checkpoint",
                str(checkpoint),
            ]
        )


def test_checkpoint_rng_state_round_trip():
    seed_everything(123, deterministic=False)
    state = capture_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    seed_everything(999, deterministic=False)
    restore_rng_state(state)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
