import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import ray
import os
import numpy as np
import random
import json

from model import PolicyNet, QNet
from temporal_pose import pose_config, validate_checkpoint_pose
from replay_buffer import TrajectoryReplayBuffer
from runner import RLRunner, validate_runner_resource_config
from parameter import *
from env import configured_training_map_paths
from training_profiler import TrainingProfiler, profile_section


def save_checkpoint_atomic(checkpoint, path):
    """Write a checkpoint atomically and retain the last complete version."""
    temporary_path = path + ".tmp"
    previous_path = path + ".previous"
    previous_temporary_path = previous_path + ".tmp"
    torch.save(checkpoint, temporary_path)
    with open(temporary_path, "rb") as checkpoint_file:
        os.fsync(checkpoint_file.fileno())
    if os.path.exists(path):
        if os.path.exists(previous_temporary_path):
            os.unlink(previous_temporary_path)
        os.link(path, previous_temporary_path)
        os.replace(previous_temporary_path, previous_path)
    os.replace(temporary_path, path)


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
        "use_action_features_in_token": USE_HISTORY_ACTION_FEATURES_IN_TOKEN,
        "use_features_in_token": USE_HISTORY_FEATURES_IN_TOKEN,
        "use_behavior_reward_in_token": (
            USE_HISTORY_BEHAVIOR_REWARD_IN_TOKEN
        ),
        "use_frontier_delta_in_token": USE_HISTORY_FRONTIER_DELTA_IN_TOKEN,
        "use_map_delta_in_token": USE_HISTORY_MAP_DELTA_IN_TOKEN,
        "use_motion_delta_in_token": USE_HISTORY_MOTION_DELTA_IN_TOKEN,
        "use_reward_in_token": USE_HISTORY_REWARD_IN_TOKEN,
        "temporal_pose_encoding": TEMPORAL_POSE_ENCODING,
        "temporal_delta_position_scale": TEMPORAL_DELTA_POSITION_SCALE,
    }


def temporal_token_config():
    return {
        "layout": "local_motion_target_reward_new_free_cells_v2",
        "raw_input_dim": ACTION_EMBEDDING_DIM + 5,
        **({"action_features": USE_HISTORY_ACTION_FEATURES_IN_TOKEN}
           if USE_HISTORY_ACTION_FEATURES_IN_TOKEN != USE_HISTORY_FEATURES_IN_TOKEN else {}),
        "features": USE_HISTORY_FEATURES_IN_TOKEN,
        "behavior_reward": USE_HISTORY_BEHAVIOR_REWARD_IN_TOKEN,
        "frontier_delta": False,
        "map_delta": USE_HISTORY_MAP_DELTA_IN_TOKEN,
        "motion_delta": USE_HISTORY_MOTION_DELTA_IN_TOKEN,
        "reward": USE_HISTORY_REWARD_IN_TOKEN,
    }


def training_map_config():
    selected_maps = configured_training_map_paths()
    if selected_maps is None:
        return None
    return {
        "split_file": os.path.abspath(MAP_SPLIT_FILE),
        "split_name": MAP_SPLIT_NAME,
        "map_count": len(selected_maps),
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
    if (USE_MAMBA or USE_LOCAL_REVISIT_RECOVERY) and checkpoint.get(
            "exploration_progress_signal") != "new_free_cell_count_v1":
        raise RuntimeError("Legacy checkpoint uses coverage feedback; raw new-free-cell "
                           "feedback requires a newly trained checkpoint")
    if USE_MAMBA:
        validate_checkpoint_pose(checkpoint, TEMPORAL_POSE_ENCODING,
                                 TEMPORAL_DELTA_POSITION_SCALE,
                                 ALLOW_TEMPORAL_POSE_OVERRIDE)
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
            "raw_token_dim": ACTION_EMBEDDING_DIM + 5,
        }
        checkpoint_config = checkpoint.get("mamba_config")
        if checkpoint_config != expected_config:
            raise RuntimeError(
                "Mamba checkpoint configuration mismatch: "
                f"checkpoint={checkpoint_config}, expected={expected_config}"
            )
        expected_token_config = temporal_token_config()
        checkpoint_token_config = checkpoint.get("temporal_token_config")
        if checkpoint_token_config is None:
            full_token_config = {
                name: True for name in expected_token_config
            }
            if expected_token_config != full_token_config:
                raise RuntimeError(
                    "Legacy checkpoint has no temporal_token_config and may "
                    "only be loaded with the full history token; current="
                    f"{expected_token_config}"
                )
        else:
            checkpoint_token_config = dict(checkpoint_token_config)
            # Checkpoints created before the fine-grained switch always used
            # the three motion values, so their implicit value is True.
            checkpoint_token_config.setdefault("motion_delta", True)
            checkpoint_token_config.setdefault("reward", True)
        if checkpoint_token_config is not None and checkpoint_token_config != expected_token_config:
            raise RuntimeError(
                "History-token checkpoint configuration mismatch: "
                f"checkpoint={checkpoint_token_config}, "
                f"expected={expected_token_config}"
            )


def unwrap_model(model):
    if isinstance(model, (nn.DataParallel, DistributedDataParallel)):
        return model.module
    return model


def initialize_distributed_learner():
    if not ENABLE_DISTRIBUTED_LEARNER:
        return 0, 1, None
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Distributed learner must be launched with torchrun; missing "
            f"environment variables: {missing}"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != RAY_ROLLOUT_GPUS:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} must equal "
            f"RAY_ROLLOUT_GPUS={RAY_ROLLOUT_GPUS}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=DISTRIBUTED_BACKEND,
        init_method="env://",
    )
    control_group = dist.new_group(backend="gloo")
    return rank, world_size, control_group


def broadcast_control(payload, rank, control_group):
    objects = [payload if rank == 0 else None]
    dist.broadcast_object_list(objects, src=0, group=control_group)
    return objects[0]


def wrap_distributed_training_networks(networks, local_rank):
    return tuple(
        DistributedDataParallel(
            network,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        for network in networks
    )


def broadcast_model_state(model):
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, src=0)


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
    use_rollout_history_cache=USE_ROLLOUT_HISTORY_CACHE,
    runner_class=RLRunner,
):
    return [
        runner_class.remote(
            i,
            use_mamba=use_mamba,
            use_rollout_history_cache=use_rollout_history_cache,
        )
        for i in range(num_meta_agents)
    ]


def validate_history_optimization_config(
    use_mamba=USE_MAMBA,
    use_rollout_history_cache=USE_ROLLOUT_HISTORY_CACHE,
    use_learner_history_dedup=USE_LEARNER_HISTORY_DEDUP,
):
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
):
    return {
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
        "target_q_update_counter": target_q_update_counter,
        "exploration_progress_signal": "new_free_cell_count_v1",
        "use_mamba": USE_MAMBA,
        "mamba_config": {
            "sequence_length": MAMBA_SEQ_LEN,
            "d_model": MAMBA_D_MODEL,
            "d_state": MAMBA_D_STATE,
            "d_conv": MAMBA_D_CONV,
            "expand": MAMBA_EXPAND,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
            "action_embedding_dim": ACTION_EMBEDDING_DIM,
            "raw_token_dim": ACTION_EMBEDDING_DIM + 5,
        },
        **pose_config(TEMPORAL_POSE_ENCODING, TEMPORAL_DELTA_POSITION_SCALE),
        "temporal_pose_override_allowed": ALLOW_TEMPORAL_POSE_OVERRIDE,
        "temporal_token_config": temporal_token_config(),
        "recovery_trigger_config": {
            "regional_trigger_steps": LOCAL_REVISIT_TRIGGER_STEPS,
            "point_repeat_trigger_steps": POINT_REPEAT_TRIGGER_STEPS,
            "new_free_cell_threshold": LOCAL_REVISIT_NEW_FREE_CELL_THRESHOLD,
            "progress_window": LOCAL_REVISIT_PROGRESS_WINDOW,
        },
        "use_local_revisit_recovery": USE_LOCAL_REVISIT_RECOVERY,
        "training_map_config": training_map_config(),
    }


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
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(log_alpha.grad, op=dist.ReduceOp.SUM)
            log_alpha.grad.div_(dist.get_world_size())
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


def main():
    import csv
    import time
    from training_budget import budget_reached as reached_budget, can_launch, restore_progress
    from torch.utils.tensorboard import SummaryWriter
    env_budget = int(os.environ.get('ARIADNE_TOTAL_ENV_STEPS', '0'))
    episode_budget = int(os.environ.get('ARIADNE_TOTAL_EPISODES', '0'))
    completed_env_steps = 0
    completed_episodes = 0
    learner_updates = 0
    progress_stream = None
    progress_writer = None
    run_started = time.monotonic()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    active_map_config = training_map_config()

    rank, world_size, control_group = initialize_distributed_learner()
    is_primary = rank == 0
    local_rank = int(os.environ.get("LOCAL_RANK", LEARNER_GPU_INDEX))
    validate_history_optimization_config()
    validate_runner_resource_config(
        NUM_META_AGENT,
        RUNNER_GPU_FRACTION,
        RAY_ROLLOUT_GPUS,
    )
    if USE_MAMBA and not torch.cuda.is_available():
        raise RuntimeError(
            "USE_MAMBA=True requires an available NVIDIA CUDA GPU"
        )
    if USE_MAMBA and not USE_GPU_GLOBAL:
        raise RuntimeError(
            "USE_MAMBA=True requires USE_GPU_GLOBAL=True"
        )
    if USE_BF16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this GPU")
    required_gpu_count = max(RAY_ROLLOUT_GPUS, LEARNER_GPU_INDEX + 1)
    available_gpu_count = torch.cuda.device_count()
    if LEARNER_GPU_INDEX < 0:
        raise RuntimeError("LEARNER_GPU_INDEX must be non-negative")
    if available_gpu_count < required_gpu_count:
        raise RuntimeError(
            "Configured multi-GPU layout requires at least "
            f"{required_gpu_count} CUDA GPUs, but found "
            f"{available_gpu_count}"
        )

    if is_primary:
        ray.init(
            address='local',
            _temp_dir=os.environ.get('ARIADNE_RAY_TEMP_DIR', '/tmp/ray'),
            num_cpus=NUM_META_AGENT,
            num_gpus=RAY_ROLLOUT_GPUS,
            object_store_memory=RAY_OBJECT_STORE_MEMORY,
            include_dashboard=False,
        )
        print("Welcome to RL autonomous exploration!")
        print(
            "History optimizations: "
            f"rollout_cache={USE_ROLLOUT_HISTORY_CACHE}, "
            f"learner_dedup={USE_LEARNER_HISTORY_DEDUP}"
        )
        print(
            "Ablation configuration: "
            f"run={RUN_FOLDER_NAME}, seed={RANDOM_SEED}, "
            f"token={temporal_token_config()}, "
            f"recovery={USE_LOCAL_REVISIT_RECOVERY}, "
            f"maps={active_map_config}"
        )
        writer = SummaryWriter(train_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(gifs_path):
            os.makedirs(gifs_path)
    else:
        writer = None

    # use GPU/CPU for driver/worker
    device = (
        torch.device(f'cuda:{local_rank}')
        if USE_GPU_GLOBAL
        else torch.device('cpu')
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if is_primary and ENABLE_DISTRIBUTED_LEARNER:
            print(
                "GPU layout: learner=DDP/NCCL on "
                f"{world_size} GPUs, rollout_gpus="
                f"0-{RAY_ROLLOUT_GPUS - 1}, "
                f"rollout_workers={NUM_META_AGENT}"
            )
        elif is_primary:
            print(
                f"GPU layout: learner={device} exclusive, "
                f"rollout_gpus=0-{RAY_ROLLOUT_GPUS - 1}, "
                f"rollout_workers={NUM_META_AGENT}"
            )
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
    target_q_update_counter = 1
    target_networks_loaded = False

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        validate_checkpoint_mode(checkpoint)
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
        completed_episodes, completed_env_steps, learner_updates, elapsed = restore_progress(
            checkpoint, os.path.join(os.environ['ARIADNE_OUTPUT_ROOT'], 'episodes.csv'))
        run_started -= elapsed
        print(f'Resumed: completed_episodes={completed_episodes} env_steps={completed_env_steps} '
              f'updates={learner_updates} episode_budget={episode_budget}', flush=True)
        if reached_budget(completed_episodes, completed_env_steps, episode_budget, env_budget):
            print('Requested training budget already reached; no rollouts launched.', flush=True)
            ray.shutdown()
            return
        target_q_update_counter = checkpoint.get(
            'target_q_update_counter',
            1,
        )

        if is_primary:
            print("curr_episode set to ", curr_episode)
            print(log_alpha, log_alpha.requires_grad)
            print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    if not target_networks_loaded:
        global_target_q_net1.load_state_dict(global_q_net1.state_dict())
        global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    if ENABLE_DISTRIBUTED_LEARNER:
        dp_policy, dp_q_net1, dp_q_net2 = (
            wrap_distributed_training_networks(
                (global_policy_net, global_q_net1, global_q_net2),
                local_rank,
            )
        )
        dp_target_q_net1 = global_target_q_net1
        dp_target_q_net2 = global_target_q_net2
        broadcast_model_state(global_target_q_net1)
        broadcast_model_state(global_target_q_net2)
        global_target_q_net1.eval()
        global_target_q_net2.eval()
    else:
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

    if is_primary:
        meta_agents = create_meta_agents(
            num_meta_agents=NUM_META_AGENT,
            use_mamba=USE_MAMBA,
            use_rollout_history_cache=USE_ROLLOUT_HISTORY_CACHE,
        )
        weights_set = [policy_weights_for_workers(global_policy_net)]
    else:
        meta_agents = []
        weights_set = None

    # launch the first job on each runner
    job_list = []
    if is_primary:
        for i, meta_agent in enumerate(meta_agents):
            if not can_launch(completed_episodes, len(job_list), episode_budget):
                break
            curr_episode += 1
            job_list.append(
                meta_agent.job.remote(weights_set, curr_episode)
            )

    # initialize metric collector
    metric_name = [
        'travel_dist',
        'success_rate',
        'explored_rate',
        'total_recovery_triggers',
        'total_recovery_steps',
        'recovery_step_ratio',
        'recovery_target_reached',
        'recovery_target_reach_rate',
        'recovery_mean_steps_per_trigger',
        'recovery_retrigger_count',
        'recovery_frontier_target_count',
        'recovery_filter_fallback_count',
        'recovery_graph_fallback_count',
        'recovery_nearest_node_count',
        'recovery_regional_trigger_count',
        'recovery_point_repeat_trigger_count',
        'recovery_combined_trigger_count',
        'recovery_retrigger_within_20_count',
        'recovery_retrigger_within_30_count',
        'recovery_mean_candidate_num',
        'recovery_mean_frontier_count',
        'recovery_mean_navigation_distance',
        'recovery_mean_loop_distance',
    ]
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(16):
        experience_buffer.append([])
    trajectory_replay_buffer = TrajectoryReplayBuffer(
        REPLAY_SIZE,
        use_union_history_collate=USE_UNION_HISTORY_COLLATE,
    )
    if ENABLE_DISTRIBUTED_LEARNER:
        if not USE_MAMBA:
            raise RuntimeError(
                "Distributed learner currently requires USE_MAMBA=True"
            )
        if BATCH_SIZE % world_size != 0:
            raise RuntimeError(
                f"BATCH_SIZE={BATCH_SIZE} must be divisible by "
                f"WORLD_SIZE={world_size}"
            )
    local_batch_size = BATCH_SIZE // world_size
    train_profiler = TrainingProfiler(
        enabled=ENABLE_TRAIN_PROFILING,
        warmup_updates=PROFILE_WARMUP_UPDATES,
        active_updates=PROFILE_ACTIVE_UPDATES,
        log_interval=PROFILE_LOG_INTERVAL,
        device=device,
    )

    # collect data from worker and do training
    if is_primary and os.environ.get('ARIADNE_OUTPUT_ROOT'):
        progress_path = os.path.join(os.environ['ARIADNE_OUTPUT_ROOT'], 'episodes.csv')
        has_progress = LOAD_MODEL and os.path.exists(progress_path)
        progress_stream = open(progress_path, 'a' if LOAD_MODEL else 'x', newline='')
        progress_writer = csv.DictWriter(progress_stream, fieldnames=[
            'episode', 'completed_episodes', 'env_steps', 'learner_updates',
            'elapsed_seconds', 'episode_steps', 'success', 'coverage', 'path_length',
            'map_path', 'decision_total_ms', 'decision_count', 'recovery_steps', 'recovery_triggers'])
        if not has_progress:
            progress_writer.writeheader()
        progress_stream.flush()
    try:
        while True:
            if is_primary:
                done_id, job_list = ray.wait(job_list)
                done_jobs = ray.get(done_id)
                new_trajectories = []
                for job in done_jobs:
                    job_results, metrics, trajectory, info = job
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] += job_results[i]
                    if trajectory is not None:
                        trajectory_replay_buffer.add_episode(trajectory)
                        new_trajectories.append(trajectory)
                        completed_env_steps += len(trajectory)
                        completed_episodes += 1
                    for n in metric_name:
                        perf_metrics[n].append(metrics[n])

                if can_launch(completed_episodes, len(job_list), episode_budget):
                    curr_episode += 1
                    job_list.append(
                        meta_agents[info['id']].job.remote(
                            weights_set,
                            curr_episode,
                        )
                    )
                replay_steps = (
                    trajectory_replay_buffer.available_transitions
                )
                should_train = replay_steps >= MINIMUM_BUFFER_SIZE
                control = {
                    "stop": False,
                    "trajectories": new_trajectories,
                    "curr_episode": curr_episode,
                    "should_train": should_train,
                    "update_seeds": [
                        random.randrange(2 ** 63)
                        for _ in range(LEARNER_UPDATES_PER_TRAIN)
                    ] if should_train else [],
                }
            else:
                control = None

            if ENABLE_DISTRIBUTED_LEARNER:
                control = broadcast_control(
                    control,
                    rank,
                    control_group,
                )
                if control["stop"]:
                    break
                curr_episode = control["curr_episode"]
                if not is_primary:
                    for trajectory in control["trajectories"]:
                        trajectory_replay_buffer.add_episode(trajectory)

            # start training
            replay_steps = (
                trajectory_replay_buffer.available_transitions
                if USE_MAMBA
                else len(experience_buffer[0])
            )
            if control["should_train"]:
                if is_primary:
                    print("training")

                # keep the replay buffer size
                if not USE_MAMBA and len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # Keep GPUs occupied for multiple learner updates after each
                # completed rollout. This is configurable for multi-GPU runs.
                for j in range(LEARNER_UPDATES_PER_TRAIN):
                    profile_active = train_profiler.begin_update()
                    if USE_MAMBA:
                        aligned_batch = (
                            trajectory_replay_buffer
                            .sample_aligned_transitions(
                                batch_size=local_batch_size,
                                sequence_length=MAMBA_SEQ_LEN,
                                seed=control["update_seeds"][j] + rank,
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
                        if train_profiler.should_log and is_primary:
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
                    learner_updates += 1
                    # print("target q update counter", target_q_update_counter % 1024)

                # data record to be written in tensorboard
                if is_primary:
                    perf_data = []
                    for n in metric_name:
                        perf_data.append(np.nanmean(perf_metrics[n]))
                    data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                            entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                            alpha_loss.item(), *perf_data]
                    training_data.append(data)

            # write record to tensorboard
            if is_primary and len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            if is_primary:
                weights_set = [
                    policy_weights_for_workers(global_policy_net)
                ]

            # update the target q net
            if target_q_update_counter > 64:
                if is_primary:
                    print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            if is_primary and progress_writer is not None:
                progress_writer.writerow(dict(
                    episode=info['episode_number'], completed_episodes=completed_episodes,
                    env_steps=completed_env_steps, learner_updates=learner_updates,
                    elapsed_seconds=time.monotonic()-run_started,
                    episode_steps=len(trajectory), success=metrics['success_rate'],
                    coverage=metrics['explored_rate'], path_length=metrics['travel_dist'],
                    map_path=metrics.get('map_path', ''),
                    decision_total_ms=metrics.get('decision_total_ms', 0),
                    decision_count=metrics.get('decision_count', 0),
                    recovery_steps=metrics['total_recovery_steps'],
                    recovery_triggers=metrics['total_recovery_triggers']))
                progress_stream.flush()
                if os.environ.get('ARIADNE_WRITE_RECOVERY_DIAGNOSTICS') == '1':
                    with open(os.path.join(os.environ['ARIADNE_OUTPUT_ROOT'],
                                           'recovery_diagnostics.jsonl'), 'a') as diagnostic_stream:
                        diagnostic_stream.write(json.dumps({
                            'episode': info['episode_number'],
                            'completed_episodes': completed_episodes,
                            'map_path': metrics.get('map_path', ''),
                            'steps': metrics.get('recovery_diagnostics', []),
                            'events': metrics.get('recovery_trigger_events', []),
                            'target_reached': metrics.get('recovery_target_reached', 0),
                        }, default=lambda value: value.tolist()) + '\n')
                print(f'Progress: env_steps={completed_env_steps} updates={learner_updates} '
                      f'completed_episodes={completed_episodes}', flush=True)
            budget_reached = reached_budget(completed_episodes, completed_env_steps, episode_budget, env_budget)
            # Preserve original rollout-boundary scheduling; budget may overshoot by <=127.
            if is_primary and (curr_episode % 32 == 0 or budget_reached):
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
                )
                checkpoint.update(env_steps=completed_env_steps, learner_updates=learner_updates,
                                  completed_episodes=completed_episodes, requested_total_episodes=episode_budget,
                                  training_seed=RANDOM_SEED, model_seed=RANDOM_SEED,
                                  requested_total_env_steps=env_budget, is_final=budget_reached)
                path_checkpoint = os.path.join(model_path, 'checkpoint.pth')
                save_checkpoint_atomic(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
            if budget_reached:
                break

    except KeyboardInterrupt:
        if is_primary:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)
    finally:
        if progress_stream is not None:
            progress_stream.close()
        if writer is not None:
            writer.close()
        if is_primary:
            ray.shutdown()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    (
        reward,
        value,
        policy_loss,
        q_value_loss,
        entropy,
        policy_grad_norm,
        q_value_grad_norm,
        log_alpha,
        alpha_loss,
        travel_dist,
        success_rate,
        explored_rate,
        recovery_triggers,
        recovery_steps,
        recovery_step_ratio,
        recovery_target_reached,
        recovery_target_reach_rate,
        recovery_mean_steps_per_trigger,
        recovery_retrigger_count,
        recovery_frontier_target_count,
        recovery_filter_fallback_count,
        recovery_graph_fallback_count,
        recovery_nearest_node_count,
        recovery_regional_trigger_count,
        recovery_point_repeat_trigger_count,
        recovery_combined_trigger_count,
        recovery_retrigger_within_20_count,
        recovery_retrigger_within_30_count,
        recovery_mean_candidate_num,
        recovery_mean_frontier_count,
        recovery_mean_navigation_distance,
        recovery_mean_loop_distance,
    ) = tensorboard_data

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
    writer.add_scalar(tag='Recovery/Triggers', scalar_value=recovery_triggers, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/Steps', scalar_value=recovery_steps, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/Step Ratio', scalar_value=recovery_step_ratio, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/TargetReached', scalar_value=recovery_target_reached, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/TargetReachRate', scalar_value=recovery_target_reach_rate, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/MeanStepsPerTrigger', scalar_value=recovery_mean_steps_per_trigger, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/ReTriggerCount', scalar_value=recovery_retrigger_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/FrontierTargets', scalar_value=recovery_frontier_target_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/FilterFallbacks', scalar_value=recovery_filter_fallback_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/GraphFallbacks', scalar_value=recovery_graph_fallback_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/NearestNodeTargets', scalar_value=recovery_nearest_node_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/RegionalTriggers', scalar_value=recovery_regional_trigger_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/PointRepeatTriggers', scalar_value=recovery_point_repeat_trigger_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/CombinedEvidenceTriggers', scalar_value=recovery_combined_trigger_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/ReTriggerWithin20', scalar_value=recovery_retrigger_within_20_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/ReTriggerWithin30', scalar_value=recovery_retrigger_within_30_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/MeanCandidateNum', scalar_value=recovery_mean_candidate_num, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/MeanFrontierCount', scalar_value=recovery_mean_frontier_count, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/MeanNavigationDistance', scalar_value=recovery_mean_navigation_distance, global_step=curr_episode)
    writer.add_scalar(tag='Recovery/MeanLoopDistance', scalar_value=recovery_mean_loop_distance, global_step=curr_episode)


if __name__ == "__main__":
    main()
