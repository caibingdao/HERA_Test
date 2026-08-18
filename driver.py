import torch
import torch.optim as optim
import torch.nn as nn
import ray
import os
import numpy as np
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

from experiment_utils import (
    capture_rng_state,
    git_commit_hash,
    load_split_manifest,
    restore_rng_state,
    seed_everything,
    sha256_file,
    write_json,
)
from model import PolicyNet, QNet
from replay_buffer import TrajectoryReplayBuffer
from runner import RLRunner, validate_runner_resource_config
from parameter import *
from training_profiler import TrainingProfiler, profile_section


def network_kwargs():
    return {
        "use_mamba": USE_MAMBA,
        "mamba_seq_len": MAMBA_SEQ_LEN,
        "mamba_d_model": MAMBA_D_MODEL,
        "mamba_d_state": MAMBA_D_STATE,
        "mamba_d_conv": MAMBA_D_CONV,
        "mamba_expand": MAMBA_EXPAND,
        "fusion_hidden_dim": FUSION_HIDDEN_DIM,
        "action_embedding_dim": ACTION_EMBEDDING_DIM,
        "history_graph_batch_size": HISTORY_GRAPH_BATCH_SIZE,
        "skip_padded_history_graphs": SKIP_PADDED_HISTORY_GRAPHS,
        "trim_history_graph_nodes": TRIM_HISTORY_GRAPH_NODES,
        "use_history_checkpoint": USE_HISTORY_CHECKPOINT,
    }


def build_policy_network(device):
    return PolicyNet(
        NODE_INPUT_DIM,
        EMBEDDING_DIM,
        **network_kwargs(),
    ).to(device)


def build_q_network(device):
    return QNet(
        NODE_INPUT_DIM,
        EMBEDDING_DIM,
        **network_kwargs(),
    ).to(device)


def learner_autocast(reference_tensor, use_bf16):
    if use_bf16 and reference_tensor.device.type != "cuda":
        raise RuntimeError("BF16 learner autocast requires CUDA")
    return torch.autocast(
        device_type=reference_tensor.device.type,
        dtype=torch.bfloat16,
        enabled=bool(use_bf16),
    )


def validate_finite_training_values(values):
    invalid = [
        name
        for name, value in values.items()
        if not torch.isfinite(value).all()
    ]
    if invalid:
        raise FloatingPointError(
            f"Non-finite learner values detected: {invalid}"
        )


def validate_checkpoint_mode(checkpoint):
    checkpoint_uses_mamba = bool(checkpoint.get("use_mamba", False))
    if checkpoint_uses_mamba != USE_MAMBA:
        raise RuntimeError(
            "Checkpoint/model mode mismatch: checkpoint USE_MAMBA="
            f"{checkpoint_uses_mamba}, current USE_MAMBA={USE_MAMBA}. "
            "Use a checkpoint created by the same model variant."
        )
    if USE_MAMBA:
        missing_target_keys = [
            key
            for key in (
                "target_q_net1_model",
                "target_q_net2_model",
            )
            if key not in checkpoint
        ]
        if missing_target_keys:
            raise RuntimeError(
                "Mamba checkpoint is missing target critic state: "
                f"{missing_target_keys}"
            )
        expected_config = {
            "sequence_length": MAMBA_SEQ_LEN,
            "d_model": MAMBA_D_MODEL,
            "d_state": MAMBA_D_STATE,
            "d_conv": MAMBA_D_CONV,
            "expand": MAMBA_EXPAND,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
            "action_embedding_dim": ACTION_EMBEDDING_DIM,
        }
        checkpoint_config = checkpoint.get("mamba_config")
        if checkpoint_config != expected_config:
            raise RuntimeError(
                "Mamba checkpoint configuration mismatch: "
                f"checkpoint={checkpoint_config}, expected={expected_config}"
            )


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def standard_model_state_dict(model):
    return unwrap_model(model).state_dict()


def normalize_checkpoint_state_dict(state_dict):
    keys = list(state_dict.keys())
    if not keys:
        return state_dict

    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError(
            "Checkpoint state_dict mixes 'module.'-prefixed and standard keys"
        )
    if not all(prefixed):
        return state_dict

    normalized = state_dict.__class__()
    for key, value in state_dict.items():
        normalized_key = key[len("module."):]
        if normalized_key in normalized:
            raise RuntimeError(
                "Checkpoint contains duplicate key after removing "
                f"'module.' prefix: {normalized_key}"
            )
        normalized[normalized_key] = value
    return normalized


def load_model_state_dict(model, state_dict, model_name):
    model = unwrap_model(model)
    normalized_state = normalize_checkpoint_state_dict(state_dict)
    expected_keys = set(model.state_dict().keys())
    checkpoint_keys = set(normalized_state.keys())
    missing_keys = sorted(expected_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"{model_name} checkpoint state mismatch: "
            f"missing_keys={missing_keys}, "
            f"unexpected_keys={unexpected_keys}"
        )
    model.load_state_dict(normalized_state, strict=True)


def policy_weights_for_workers(policy_net):
    return {
        name: value.detach().cpu()
        for name, value in standard_model_state_dict(policy_net).items()
    }


def prepare_training_networks(
    networks,
    device,
    enable_data_parallel=ENABLE_MULTI_GPU_DATA_PARALLEL,
):
    use_data_parallel = (
        enable_data_parallel
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    )
    if use_data_parallel:
        return tuple(nn.DataParallel(network) for network in networks)
    return tuple(networks)


def create_meta_agents(
    num_meta_agents,
    use_mamba,
    use_rollout_history_cache=None,
    runner_class=RLRunner,
    map_list=None,
    training_seed=None,
):
    if use_rollout_history_cache is None:
        use_rollout_history_cache = (
            USE_ROLLOUT_HISTORY_CACHE and use_mamba
        )
    return [
        runner_class.remote(
            i,
            use_mamba=use_mamba,
            use_rollout_history_cache=use_rollout_history_cache,
            map_list=map_list,
            training_seed=training_seed,
        )
        for i in range(num_meta_agents)
    ]


def validate_history_optimization_config(
    use_mamba=USE_MAMBA,
    use_rollout_history_cache=None,
    use_learner_history_dedup=USE_LEARNER_HISTORY_DEDUP,
):
    if use_rollout_history_cache is None:
        use_rollout_history_cache = (
            USE_ROLLOUT_HISTORY_CACHE and use_mamba
        )
    for name, value in (
        ("USE_MAMBA", use_mamba),
        ("USE_ROLLOUT_HISTORY_CACHE", use_rollout_history_cache),
        ("USE_LEARNER_HISTORY_DEDUP", use_learner_history_dedup),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be True or False")
    if use_rollout_history_cache and not use_mamba:
        raise ValueError(
            "USE_ROLLOUT_HISTORY_CACHE=True requires USE_MAMBA=True"
        )
    if use_learner_history_dedup:
        raise RuntimeError(
            "USE_LEARNER_HISTORY_DEDUP=True is unavailable: the current "
            "SAC update does not encode current and next history with the "
            "same network parameters in one computation graph"
        )


def build_checkpoint(
    policy_net,
    q_net1,
    q_net2,
    target_q_net1,
    target_q_net2,
    log_alpha,
    policy_optimizer,
    q_net1_optimizer,
    q_net2_optimizer,
    log_alpha_optimizer,
    episode,
    target_q_update_counter,
    experiment_metadata=None,
    completed_env_steps=0,
    completed_episodes=None,
):
    if completed_episodes is None:
        completed_episodes = episode
    checkpoint = {
        "policy_model": standard_model_state_dict(policy_net),
        "q_net1_model": standard_model_state_dict(q_net1),
        "q_net2_model": standard_model_state_dict(q_net2),
        "target_q_net1_model": standard_model_state_dict(target_q_net1),
        "target_q_net2_model": standard_model_state_dict(target_q_net2),
        "log_alpha": log_alpha,
        "policy_optimizer": policy_optimizer.state_dict(),
        "q_net1_optimizer": q_net1_optimizer.state_dict(),
        "q_net2_optimizer": q_net2_optimizer.state_dict(),
        "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
        "episode": episode,
        "completed_episodes": int(completed_episodes),
        "completed_env_steps": int(completed_env_steps),
        "target_q_update_counter": target_q_update_counter,
        "use_mamba": USE_MAMBA,
        "mamba_config": {
            "sequence_length": MAMBA_SEQ_LEN,
            "d_model": MAMBA_D_MODEL,
            "d_state": MAMBA_D_STATE,
            "d_conv": MAMBA_D_CONV,
            "expand": MAMBA_EXPAND,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
            "action_embedding_dim": ACTION_EMBEDDING_DIM,
        },
    }
    if experiment_metadata is not None:
        checkpoint["experiment_metadata"] = dict(experiment_metadata)
    checkpoint["rng_state"] = capture_rng_state()
    return checkpoint


def save_checkpoint(checkpoint, path):
    """Atomically persist a training checkpoint."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)


def sac_update(
    policy_net,
    q_net1,
    q_net2,
    target_q_net1,
    target_q_net2,
    policy_optimizer,
    q_net1_optimizer,
    q_net2_optimizer,
    log_alpha,
    log_alpha_optimizer,
    entropy_target,
    observation,
    action,
    reward,
    done,
    next_observation,
    current_history=None,
    next_history=None,
    use_bf16=None,
    check_finite=None,
    reuse_current_q_forward=None,
):
    """Run the original discrete SAC losses for exactly one sampled batch."""
    if use_bf16 is None:
        use_bf16 = (
            USE_BF16 and observation[0].device.type == "cuda"
        )
    else:
        use_bf16 = bool(use_bf16)
    check_finite = (
        CHECK_TRAIN_FINITE if check_finite is None else bool(check_finite)
    )
    reuse_current_q_forward = (
        REUSE_CURRENT_Q_FORWARD
        if reuse_current_q_forward is None
        else bool(reuse_current_q_forward)
    )
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this GPU")

    with profile_section("current_q_for_policy_ms"):
        if reuse_current_q_forward:
            q_net1_optimizer.zero_grad(set_to_none=True)
            q_net2_optimizer.zero_grad(set_to_none=True)
            with learner_autocast(observation[0], use_bf16):
                q_values1 = q_net1(
                    *observation,
                    history_batch=current_history,
                )
                q_values2 = q_net2(
                    *observation,
                    history_batch=current_history,
                )
        else:
            with torch.no_grad():
                with learner_autocast(observation[0], use_bf16):
                    q_values1 = q_net1(
                        *observation,
                        history_batch=current_history,
                    )
                    q_values2 = q_net2(
                        *observation,
                        history_batch=current_history,
                    )
        q_values1 = q_values1.float()
        q_values2 = q_values2.float()
        q_values = torch.min(q_values1, q_values2)

    with profile_section("policy_current_forward_ms"):
        with learner_autocast(observation[0], use_bf16):
            logp = policy_net(
                *observation,
                history_batch=current_history,
            )
        logp = logp.float()
    policy_loss = torch.sum(
        (
            logp.exp().unsqueeze(2)
            * (
                log_alpha.exp().detach() * logp.unsqueeze(2)
                - q_values.detach()
            )
        ),
        dim=1,
    ).mean()

    with profile_section("policy_backward_ms"):
        policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_net.parameters(),
            max_norm=100,
            norm_type=2,
        )
        policy_optimizer.step()

    with profile_section("target_forward_ms"):
        with torch.no_grad():
            with learner_autocast(next_observation[0], use_bf16):
                next_logp = policy_net(
                    *next_observation,
                    history_batch=next_history,
                )
                next_q_values1 = target_q_net1(
                    *next_observation,
                    history_batch=next_history,
                )
                next_q_values2 = target_q_net2(
                    *next_observation,
                    history_batch=next_history,
                )
            next_logp = next_logp.float()
            next_q_values1 = next_q_values1.float()
            next_q_values2 = next_q_values2.float()
            next_q_values = torch.min(next_q_values1, next_q_values2)
            value_prime = torch.sum(
                next_logp.unsqueeze(2).exp()
                * (
                    next_q_values
                    - log_alpha.exp() * next_logp.unsqueeze(2)
                ),
                dim=1,
            ).unsqueeze(1)
            target_q_batch = reward + GAMMA * (1 - done) * value_prime

    mse_loss = nn.MSELoss()

    with profile_section("q1_forward_backward_ms"):
        if not reuse_current_q_forward:
            with learner_autocast(observation[0], use_bf16):
                q_values1 = q_net1(
                    *observation,
                    history_batch=current_history,
                )
            q_values1 = q_values1.float()
        q1 = torch.gather(q_values1, 1, action)
        q1_loss = mse_loss(q1, target_q_batch.detach()).mean()

        if not reuse_current_q_forward:
            q_net1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        q1_grad_norm = torch.nn.utils.clip_grad_norm_(
            q_net1.parameters(),
            max_norm=20000,
            norm_type=2,
        )
        q_net1_optimizer.step()

    with profile_section("q2_forward_backward_ms"):
        if not reuse_current_q_forward:
            with learner_autocast(observation[0], use_bf16):
                q_values2 = q_net2(
                    *observation,
                    history_batch=current_history,
                )
            q_values2 = q_values2.float()
        q2 = torch.gather(q_values2, 1, action)
        q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

        if not reuse_current_q_forward:
            q_net2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        q2_grad_norm = torch.nn.utils.clip_grad_norm_(
            q_net2.parameters(),
            max_norm=20000,
            norm_type=2,
        )
        q_net2_optimizer.step()

    entropy = (logp * logp.exp()).sum(dim=-1)
    alpha_loss = -(
        log_alpha * (entropy.detach() + entropy_target)
    ).mean()

    if check_finite:
        validate_finite_training_values(
            {
                "policy_loss": policy_loss,
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "alpha_loss": alpha_loss,
                "entropy": entropy,
                "value_prime": value_prime,
            }
        )

    with profile_section("alpha_backward_ms"):
        log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        log_alpha_optimizer.step()

    return {
        "policy_loss": policy_loss,
        "q1_loss": q1_loss,
        "q2_loss": q2_loss,
        "alpha_loss": alpha_loss,
        "entropy": entropy,
        "value_prime": value_prime,
        "policy_grad_norm": policy_grad_norm,
        "q1_grad_norm": q1_grad_norm,
        "q2_grad_norm": q2_grad_norm,
    }


def main(
    *,
    split_file=None,
    split_name="train",
    training_seed=None,
    total_env_steps=None,
    output_dir=None,
    resume_checkpoint=None,
    load_model=None,
):
    from torch.utils.tensorboard import SummaryWriter

    if total_env_steps is not None and total_env_steps <= 0:
        raise ValueError("total_env_steps must be a positive integer")
    if split_file is not None and split_name != "train":
        raise ValueError(
            "the training entry point may only use split_name='train'; "
            "test maps are reserved for independent evaluation"
        )

    selected_maps = None
    split_manifest = None
    if split_file is not None:
        split_manifest, selected_paths = load_split_manifest(
            split_file,
            split_name,
        )
        selected_maps = [str(path) for path in selected_paths]
        if training_seed is not None:
            random.Random(training_seed).shuffle(selected_maps)
        print(
            f"Loaded {len(selected_maps)} {split_name} maps from "
            f"{Path(split_file).resolve()}"
        )

    if training_seed is not None:
        seed_everything(int(training_seed), deterministic=True)
        warnings.warn(
            "Deterministic random streams are enabled. Ray rollout workers "
            "finish asynchronously, so complete training runs may still not "
            "be bitwise identical across machines or scheduling orders.",
            RuntimeWarning,
        )

    history_cache_enabled = USE_ROLLOUT_HISTORY_CACHE and USE_MAMBA
    validate_history_optimization_config(
        use_mamba=USE_MAMBA,
        use_rollout_history_cache=history_cache_enabled,
    )
    validate_runner_resource_config(
        NUM_META_AGENT,
        RUNNER_GPU_FRACTION,
    )
    if USE_MAMBA and not torch.cuda.is_available():
        raise RuntimeError(
            "USE_MAMBA=True requires an available NVIDIA CUDA GPU"
        )
    if USE_MAMBA and not USE_GPU_GLOBAL:
        raise RuntimeError(
            "USE_MAMBA=True requires USE_GPU_GLOBAL=True"
        )
    if (
        USE_BF16
        and USE_GPU_GLOBAL
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("BF16 is not supported on this GPU")

    run_model_path = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else Path(model_path)
    )
    run_train_path = (
        run_model_path / "tensorboard"
        if output_dir is not None
        else Path(train_path)
    )
    run_model_path.mkdir(parents=True, exist_ok=True)
    run_train_path.mkdir(parents=True, exist_ok=True)
    Path(gifs_path).mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent
    python_path_entries = [
        entry
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    if str(repo_root) not in python_path_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(repo_root), *python_path_entries]
        )
    experiment_metadata = {
        "git_commit": git_commit_hash(repo_root),
        "split_manifest": (
            str(Path(split_file).expanduser().resolve())
            if split_file is not None
            else None
        ),
        "split_manifest_sha256": (
            sha256_file(split_file) if split_file is not None else None
        ),
        "split_seed": (
            split_manifest.get("split_seed")
            if split_manifest is not None
            else None
        ),
        "split_name": split_name if split_file is not None else None,
        "training_seed": training_seed,
        "total_environment_steps": total_env_steps,
        "requested_total_env_steps": total_env_steps,
        "model_name": RUN_FOLDER_NAME,
        "model_source_ref": MODEL_SOURCE_REF,
        "model_source_commit": MODEL_SOURCE_COMMIT,
        "use_mamba": USE_MAMBA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "cell_size": CELL_SIZE,
            "node_resolution": NODE_RESOLUTION,
            "sensor_range": SENSOR_RANGE,
            "max_episode_step": MAX_EPISODE_STEP,
            "replay_size": REPLAY_SIZE,
            "minimum_buffer_size": MINIMUM_BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "gamma": GAMMA,
            "num_meta_agents": NUM_META_AGENT,
            "node_input_dim": NODE_INPUT_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "k_size": K_SIZE,
            "node_padding_size": NODE_PADDING_SIZE,
            "mamba_sequence_length": MAMBA_SEQ_LEN,
            "mamba_d_model": MAMBA_D_MODEL,
            "mamba_d_state": MAMBA_D_STATE,
            "mamba_d_conv": MAMBA_D_CONV,
            "mamba_expand": MAMBA_EXPAND,
            "rollout_history_cache": history_cache_enabled,
            "history_graph_batch_size": HISTORY_GRAPH_BATCH_SIZE,
            "skip_padded_history_graphs": SKIP_PADDED_HISTORY_GRAPHS,
            "trim_history_graph_nodes": TRIM_HISTORY_GRAPH_NODES,
            "use_history_checkpoint": USE_HISTORY_CHECKPOINT,
            "use_union_history_collate": USE_UNION_HISTORY_COLLATE,
            "use_bf16": USE_BF16,
            "check_train_finite": CHECK_TRAIN_FINITE,
            "reuse_current_q_forward": REUSE_CURRENT_Q_FORWARD,
            "use_non_blocking_transfer": USE_NON_BLOCKING_TRANSFER,
        },
    }
    write_json(run_model_path / "training_config.json", experiment_metadata)

    ray.init(
        num_cpus=NUM_META_AGENT,
        object_store_memory=RAY_OBJECT_STORE_MEMORY,
    )
    print("Welcome to RL autonomous exploration!")
    print(
        "History optimizations: "
        f"rollout_cache={history_cache_enabled}, "
        f"learner_dedup={USE_LEARNER_HISTORY_DEDUP}"
    )

    writer = SummaryWriter(str(run_train_path))

    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    # initialize neural networks
    global_policy_net = build_policy_network(device)
    global_q_net1 = build_q_network(device)
    global_q_net2 = build_q_network(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = build_q_network(device)
    global_target_q_net2 = build_q_network(device)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    curr_episode = 0
    completed_episodes = 0
    completed_env_steps = 0
    target_q_update_counter = 1
    target_networks_loaded = False

    # load model and optimizer trained before
    should_load_model = LOAD_MODEL if load_model is None else bool(load_model)
    checkpoint_to_load = (
        Path(resume_checkpoint).expanduser()
        if resume_checkpoint is not None
        else run_model_path / "checkpoint.pth"
    )
    if should_load_model:
        if not checkpoint_to_load.is_file():
            raise FileNotFoundError(
                f"training checkpoint does not exist: {checkpoint_to_load}. "
                "Use the split training wrapper without --resume-checkpoint "
                "to start a fresh run."
            )
        print('Loading Model...')
        try:
            checkpoint = torch.load(
                checkpoint_to_load,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_to_load,
                map_location="cpu",
            )
        validate_checkpoint_mode(checkpoint)
        if split_file is not None:
            resume_metadata = checkpoint.get("experiment_metadata")
            if not isinstance(resume_metadata, dict):
                raise RuntimeError(
                    "Cannot resume split training from a checkpoint without "
                    "experiment_metadata; its train/test provenance is unknown."
                )
            provenance_checks = {
                "split_manifest_sha256": experiment_metadata[
                    "split_manifest_sha256"
                ],
                "split_name": "train",
                "training_seed": training_seed,
            }
            mismatches = {
                key: {
                    "checkpoint": resume_metadata.get(key),
                    "requested": expected,
                }
                for key, expected in provenance_checks.items()
                if resume_metadata.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    "Resume checkpoint provenance does not match this "
                    f"training run: {mismatches}"
                )
            experiment_metadata["resumed_from"] = str(
                checkpoint_to_load.resolve()
            )
            experiment_metadata["parent_timestamp_utc"] = (
                resume_metadata.get("timestamp_utc")
            )
        load_model_state_dict(
            global_policy_net,
            checkpoint['policy_model'],
            "policy",
        )
        load_model_state_dict(
            global_q_net1,
            checkpoint['q_net1_model'],
            "q_net1",
        )
        load_model_state_dict(
            global_q_net2,
            checkpoint['q_net2_model'],
            "q_net2",
        )
        if (
            'target_q_net1_model' in checkpoint
            and 'target_q_net2_model' in checkpoint
        ):
            load_model_state_dict(
                global_target_q_net1,
                checkpoint['target_q_net1_model'],
                "target_q_net1",
            )
            load_model_state_dict(
                global_target_q_net2,
                checkpoint['target_q_net2_model'],
                "target_q_net2",
            )
            target_networks_loaded = True
        log_alpha = (
            checkpoint['log_alpha']
            .to(device)
            .detach()
            .requires_grad_(True)
        )
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
        
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        curr_episode = checkpoint['episode']
        completed_episodes = int(
            checkpoint.get("completed_episodes", curr_episode)
        )
        completed_env_steps = int(
            checkpoint.get("completed_env_steps", 0)
        )
        target_q_update_counter = checkpoint.get(
            'target_q_update_counter',
            1,
        )
        checkpoint_rng_state = checkpoint.get("rng_state")
        if checkpoint_rng_state is None:
            warnings.warn(
                "Resume checkpoint has no rng_state; random streams will "
                "restart from the requested training seed.",
                RuntimeWarning,
            )
        else:
            restore_rng_state(checkpoint_rng_state)

        experiment_metadata.update(
            {
                "resume_checkpoint_sha256": sha256_file(
                    checkpoint_to_load
                ),
                "resumed_completed_env_steps": completed_env_steps,
                "resumed_completed_episodes": completed_episodes,
            }
        )
        write_json(
            run_model_path / "training_config.json",
            experiment_metadata,
        )

        print("curr_episode set to ", curr_episode)
        print("completed_env_steps set to ", completed_env_steps)
        print(log_alpha, log_alpha.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    if not target_networks_loaded:
        global_target_q_net1.load_state_dict(global_q_net1.state_dict())
        global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    def persist_final_checkpoint():
        final_checkpoint = build_checkpoint(
            policy_net=global_policy_net,
            q_net1=global_q_net1,
            q_net2=global_q_net2,
            target_q_net1=global_target_q_net1,
            target_q_net2=global_target_q_net2,
            log_alpha=log_alpha,
            policy_optimizer=global_policy_optimizer,
            q_net1_optimizer=global_q_net1_optimizer,
            q_net2_optimizer=global_q_net2_optimizer,
            log_alpha_optimizer=log_alpha_optimizer,
            episode=curr_episode,
            target_q_update_counter=target_q_update_counter,
            experiment_metadata=experiment_metadata,
            completed_env_steps=completed_env_steps,
            completed_episodes=completed_episodes,
        )
        save_checkpoint(
            final_checkpoint,
            run_model_path / "checkpoint.pth",
        )
        final_path = run_model_path / "final_checkpoint.pth"
        save_checkpoint(final_checkpoint, final_path)
        completed_metadata = dict(experiment_metadata)
        completed_metadata.update(
            {
                "completed_env_steps": completed_env_steps,
                "completed_episodes": completed_episodes,
                "final_scheduled_episode_id": curr_episode,
                "final_checkpoint": str(final_path.resolve()),
            }
        )
        write_json(
            run_model_path / "training_config.json",
            completed_metadata,
        )
        return str(final_path.resolve())

    if (
        total_env_steps is not None
        and completed_env_steps >= total_env_steps
    ):
        print(
            "Checkpoint already satisfies the requested environment-step "
            f"budget: {completed_env_steps} >= {total_env_steps}"
        )
        final_checkpoint_path = persist_final_checkpoint()
        writer.close()
        ray.shutdown()
        return final_checkpoint_path

    # launch meta agents
    meta_agents = create_meta_agents(
        num_meta_agents=NUM_META_AGENT,
        use_mamba=USE_MAMBA,
        use_rollout_history_cache=history_cache_enabled,
        map_list=selected_maps,
        training_seed=training_seed,
    )

    # get global networks weights
    weights_set = []
    policy_weights = policy_weights_for_workers(global_policy_net)
    weights_set.append(policy_weights)

    # DataParallel is opt-in and only useful when multiple GPUs are present.
    (
        dp_policy,
        dp_q_net1,
        dp_q_net2,
        dp_target_q_net1,
        dp_target_q_net2,
    ) = prepare_training_networks(
        (
            global_policy_net,
            global_q_net1,
            global_q_net2,
            global_target_q_net1,
            global_target_q_net2,
        ),
        device=device,
    )

    # launch the first job on each runner
    job_list = []
    job_step_limits = {}
    reserved_env_steps = 0
    for i, meta_agent in enumerate(meta_agents):
        max_episode_steps = None
        if total_env_steps is not None:
            available_steps = (
                total_env_steps
                - completed_env_steps
                - reserved_env_steps
            )
            if available_steps <= 0:
                break
            max_episode_steps = min(
                MAX_EPISODE_STEP,
                available_steps,
            )
        curr_episode += 1
        job_ref = meta_agent.job.remote(
            weights_set,
            curr_episode,
            max_episode_steps,
        )
        job_list.append(job_ref)
        if max_episode_steps is not None:
            job_step_limits[job_ref] = max_episode_steps
            reserved_env_steps += max_episode_steps

    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(15):
        experience_buffer.append([])
    trajectory_replay_buffer = TrajectoryReplayBuffer(
        REPLAY_SIZE,
        use_union_history_collate=USE_UNION_HISTORY_COLLATE,
    )
    train_profiler = TrainingProfiler(
        enabled=ENABLE_TRAIN_PROFILING,
        warmup_updates=PROFILE_WARMUP_UPDATES,
        active_updates=PROFILE_ACTIVE_UPDATES,
        log_interval=PROFILE_LOG_INTERVAL,
        device=device,
    )
    final_checkpoint_path = None

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            if total_env_steps is not None:
                for job_ref in done_id:
                    reserved_env_steps -= job_step_limits.pop(job_ref)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                job_results, metrics, trajectory, info = job
                completed_episodes += 1
                completed_env_steps += int(metrics.get("steps", 0))
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                if trajectory is not None:
                    trajectory_replay_buffer.add_episode(trajectory)
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            reached_step_limit = (
                total_env_steps is not None
                and completed_env_steps >= total_env_steps
            )

            # Keep the legacy asynchronous scheduling unless a requested
            # environment-step budget has just been reached.
            if not reached_step_limit:
                max_episode_steps = None
                if total_env_steps is not None:
                    available_steps = (
                        total_env_steps
                        - completed_env_steps
                        - reserved_env_steps
                    )
                    if available_steps > 0:
                        max_episode_steps = min(
                            MAX_EPISODE_STEP,
                            available_steps,
                        )
                if total_env_steps is None or max_episode_steps is not None:
                    curr_episode += 1
                    job_ref = meta_agents[info['id']].job.remote(
                        weights_set,
                        curr_episode,
                        max_episode_steps,
                    )
                    job_list.append(job_ref)
                    if max_episode_steps is not None:
                        job_step_limits[job_ref] = max_episode_steps
                        reserved_env_steps += max_episode_steps

            # start training
            replay_steps = (
                trajectory_replay_buffer.available_transitions
                if USE_MAMBA
                else len(experience_buffer[0])
            )
            if curr_episode % 1 == 0 and replay_steps >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if not USE_MAMBA and len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(8):
                    profile_active = train_profiler.begin_update()
                    if USE_MAMBA:
                        aligned_batch = (
                            trajectory_replay_buffer
                            .sample_aligned_transitions(
                                batch_size=BATCH_SIZE,
                                sequence_length=MAMBA_SEQ_LEN,
                                seed=random.randrange(2 ** 63),
                            )
                        )
                        with profile_section("batch_to_device_ms"):
                            aligned_batch = aligned_batch.to(
                                device,
                                non_blocking=USE_NON_BLOCKING_TRANSFER,
                            )
                        observation = list(aligned_batch.observations)
                        action = aligned_batch.actions
                        reward = aligned_batch.rewards
                        done = aligned_batch.dones
                        next_observation = list(
                            aligned_batch.next_observations
                        )
                        current_history = aligned_batch.current_history
                        next_history = aligned_batch.next_history
                    else:
                        # Original single-transition replay path.
                        sample_indices = random.sample(
                            indices,
                            BATCH_SIZE,
                        )
                        rollouts = []
                        for i in range(len(experience_buffer)):
                            rollouts.append(
                                [
                                    experience_buffer[i][index]
                                    for index in sample_indices
                                ]
                            )

                        with profile_section("batch_to_device_ms"):
                            stacked = [
                                torch.stack(rollout).to(device)
                                for rollout in rollouts
                            ]
                        observation = stacked[0:6]
                        action, reward, done = stacked[6:9]
                        next_observation = stacked[9:15]
                        current_history = None
                        next_history = None

                    with profile_section("total_sac_update_ms"):
                        update_result = sac_update(
                            policy_net=dp_policy,
                            q_net1=dp_q_net1,
                            q_net2=dp_q_net2,
                            target_q_net1=dp_target_q_net1,
                            target_q_net2=dp_target_q_net2,
                            policy_optimizer=global_policy_optimizer,
                            q_net1_optimizer=global_q_net1_optimizer,
                            q_net2_optimizer=global_q_net2_optimizer,
                            log_alpha=log_alpha,
                            log_alpha_optimizer=log_alpha_optimizer,
                            entropy_target=entropy_target,
                            observation=observation,
                            action=action,
                            reward=reward,
                            done=done,
                            next_observation=next_observation,
                            current_history=current_history,
                            next_history=next_history,
                        )
                    if profile_active:
                        train_profiler.end_update()
                        if train_profiler.should_log:
                            print(
                                "learner profile (ms): "
                                + train_profiler.format_summary()
                            )

                    policy_loss = update_result["policy_loss"]
                    q1_loss = update_result["q1_loss"]
                    q2_loss = update_result["q2_loss"]
                    alpha_loss = update_result["alpha_loss"]
                    entropy = update_result["entropy"]
                    value_prime = update_result["value_prime"]
                    policy_grad_norm = update_result[
                        "policy_grad_norm"
                    ]
                    q_grad_norm = update_result["q2_grad_norm"]

                    target_q_update_counter += 1
                    # print("target q update counter", target_q_update_counter % 1024)

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            policy_weights = policy_weights_for_workers(global_policy_net)
            weights_set.append(policy_weights)

            # update the target q net
            if target_q_update_counter > 64:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = build_checkpoint(
                    policy_net=global_policy_net,
                    q_net1=global_q_net1,
                    q_net2=global_q_net2,
                    target_q_net1=global_target_q_net1,
                    target_q_net2=global_target_q_net2,
                    log_alpha=log_alpha,
                    policy_optimizer=global_policy_optimizer,
                    q_net1_optimizer=global_q_net1_optimizer,
                    q_net2_optimizer=global_q_net2_optimizer,
                    log_alpha_optimizer=log_alpha_optimizer,
                    episode=curr_episode,
                    target_q_update_counter=target_q_update_counter,
                    experiment_metadata=experiment_metadata,
                    completed_env_steps=completed_env_steps,
                    completed_episodes=completed_episodes,
                )
                path_checkpoint = run_model_path / "checkpoint.pth"
                save_checkpoint(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

            if reached_step_limit:
                print(
                    "Reached requested environment-step budget: "
                    f"{completed_env_steps} >= {total_env_steps}"
                )
                final_checkpoint_path = persist_final_checkpoint()
                break

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
    finally:
        writer.close()
        ray.shutdown()

    return final_checkpoint_path


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, travel_dist, success_rate, explored_rate = tensorboard_data

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)


if __name__ == "__main__":
    main()
