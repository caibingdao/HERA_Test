"""ARIADNE and ARIADNE+Mamba evaluation adapter.

This module deliberately constructs only the policy network.  It never creates
optimizers, critic networks, replay buffers used for learning, or target
networks.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agent import Agent, NoValidActionError
from env import Env
from experiment_utils import (
    checkpoint_model_seed,
    coverage_success,
    seed_everything,
    trajectory_path_length,
)
from local_revisit_recovery import LocalRevisitDetector
from model import PolicyNet
from parameter import (
    ACTION_EMBEDDING_DIM,
    EMBEDDING_DIM,
    FUSION_HIDDEN_DIM,
    HISTORY_GRAPH_BATCH_SIZE,
    LOCAL_REVISIT_COVERAGE_THRESHOLD,
    LOCAL_REVISIT_EXIT_THRESHOLD,
    LOCAL_REVISIT_GRAPH_RADIUS,
    LOCAL_REVISIT_HISTORY_LENGTH,
    LOCAL_REVISIT_MIN_COUNT,
    LOCAL_REVISIT_PROGRESS_WINDOW,
    LOCAL_REVISIT_SCORE_THRESHOLD,
    LOCAL_REVISIT_TRIGGER_STEPS,
    MAMBA_D_CONV,
    MAMBA_D_MODEL,
    MAMBA_D_STATE,
    MAMBA_EXPAND,
    MAMBA_SEQ_LEN,
    NODE_INPUT_DIM,
    POINT_REPEAT_MAX_COUNT,
    POINT_REPEAT_TOP_K,
    POINT_REPEAT_TOP_K_COUNT,
    POINT_REPEAT_TRIGGER_STEPS,
    RECOVERY_GOAL_TOLERANCE,
    RECOVERY_MAX_STEPS,
    RECOVERY_PATH_FAILURE_LIMIT,
    SKIP_PADDED_HISTORY_GRAPHS,
    TRIM_HISTORY_GRAPH_NODES,
    USE_FRONTIER_ESCAPE,
    USE_HISTORY_CHECKPOINT,
    USE_POINT_REPEAT_DETECTION,
    ESCAPE_DISTANCE_EPS,
    ESCAPE_MIN_DISTANCE,
    ESCAPE_MIN_LOOP_DISTANCE,
    FRONTIER_CELL_SIZE,
)
from replay_buffer import (
    EpisodeTrajectory,
    TrajectoryStep,
    build_history_sequence,
    detach_observation,
)
from utils import get_frontier_in_map


_MAMBA_CONFIG_DEFAULTS = {
    "sequence_length": MAMBA_SEQ_LEN,
    "d_model": MAMBA_D_MODEL,
    "d_state": MAMBA_D_STATE,
    "d_conv": MAMBA_D_CONV,
    "expand": MAMBA_EXPAND,
    "fusion_hidden_dim": FUSION_HIDDEN_DIM,
    "action_embedding_dim": ACTION_EMBEDDING_DIM,
}
_MAMBA_CONFIG_KEYS = tuple(_MAMBA_CONFIG_DEFAULTS)
_ADAPTER_ARGUMENTS = {
    "device",
    "enable_recovery",
    "model_name",
    "model_seed",
}


def _recovery_configuration() -> Dict[str, Any]:
    """Return the exact Loop Recovery V4 configuration used for training."""

    return {
        "history_length": LOCAL_REVISIT_HISTORY_LENGTH,
        "graph_radius": LOCAL_REVISIT_GRAPH_RADIUS,
        "score_threshold": LOCAL_REVISIT_SCORE_THRESHOLD,
        "exit_threshold": LOCAL_REVISIT_EXIT_THRESHOLD,
        "progress_window": LOCAL_REVISIT_PROGRESS_WINDOW,
        "coverage_threshold": LOCAL_REVISIT_COVERAGE_THRESHOLD,
        "trigger_steps": LOCAL_REVISIT_TRIGGER_STEPS,
        "min_count": LOCAL_REVISIT_MIN_COUNT,
        "goal_tolerance": RECOVERY_GOAL_TOLERANCE,
        "max_recovery_steps": RECOVERY_MAX_STEPS,
        "path_failure_limit": RECOVERY_PATH_FAILURE_LIMIT,
        "use_frontier_escape": USE_FRONTIER_ESCAPE,
        "escape_min_distance": ESCAPE_MIN_DISTANCE,
        "escape_min_loop_distance": ESCAPE_MIN_LOOP_DISTANCE,
        "escape_distance_eps": ESCAPE_DISTANCE_EPS,
        "frontier_cell_size": FRONTIER_CELL_SIZE,
        "point_repeat_enabled": USE_POINT_REPEAT_DETECTION,
        "point_repeat_max_count": POINT_REPEAT_MAX_COUNT,
        "point_repeat_top_k": POINT_REPEAT_TOP_K,
        "point_repeat_top_k_count": POINT_REPEAT_TOP_K_COUNT,
        "point_repeat_trigger_steps": POINT_REPEAT_TRIGGER_STEPS,
    }


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load a full training checkpoint on CPU across PyTorch versions."""

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"checkpoint root must be a mapping, got {type(checkpoint).__name__}"
        )
    if "policy_model" not in checkpoint:
        raise KeyError("checkpoint is missing required 'policy_model' state")
    if not isinstance(checkpoint["policy_model"], Mapping):
        raise TypeError("checkpoint field 'policy_model' must be a state mapping")
    return checkpoint


def _normalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Accept both regular and DataParallel-prefixed policy checkpoints."""

    keys = list(state_dict)
    if not keys:
        raise ValueError("checkpoint policy state is empty")
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError(
            "checkpoint policy state mixes DataParallel-prefixed and "
            "unprefixed keys"
        )
    if not all(prefixed):
        return dict(state_dict)
    return {
        key[len("module.") :]: value
        for key, value in state_dict.items()
    }


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _network_dimensions(checkpoint: Mapping[str, Any]) -> tuple[int, int]:
    metadata = checkpoint.get("experiment_metadata", {})
    hyperparameters = (
        metadata.get("hyperparameters", {})
        if isinstance(metadata, Mapping)
        else {}
    )
    if not isinstance(hyperparameters, Mapping):
        hyperparameters = {}
    node_input_dim = int(
        hyperparameters.get("node_input_dim", NODE_INPUT_DIM)
    )
    embedding_dim = int(
        hyperparameters.get("embedding_dim", EMBEDDING_DIM)
    )
    if node_input_dim <= 0 or embedding_dim <= 0:
        raise ValueError(
            "checkpoint network dimensions must be positive: "
            f"node_input_dim={node_input_dim}, "
            f"embedding_dim={embedding_dim}"
        )
    return node_input_dim, embedding_dim


def _mamba_configuration(
    checkpoint: Mapping[str, Any],
    use_mamba: bool,
) -> Dict[str, int]:
    raw_configuration = checkpoint.get("mamba_config")
    if use_mamba:
        if not isinstance(raw_configuration, Mapping):
            raise ValueError(
                "Mamba checkpoint is missing the required 'mamba_config'"
            )
        missing = [
            key for key in _MAMBA_CONFIG_KEYS if key not in raw_configuration
        ]
        if missing:
            raise ValueError(
                f"Mamba checkpoint configuration is missing keys: {missing}"
            )
    elif not isinstance(raw_configuration, Mapping):
        raw_configuration = {}

    configuration = {
        key: int(raw_configuration.get(key, default))
        for key, default in _MAMBA_CONFIG_DEFAULTS.items()
    }
    non_positive = [
        key for key, value in configuration.items() if value <= 0
    ]
    if non_positive:
        raise ValueError(
            "Mamba checkpoint configuration contains non-positive values: "
            f"{non_positive}"
        )
    return configuration


def _resolve_device(requested: Any, use_mamba: bool) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(str(requested))
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid adapter device {requested!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA evaluation was requested but CUDA is unavailable: {device}"
        )
    if use_mamba and device.type != "cuda":
        raise RuntimeError(
            "ARIADNE+Mamba evaluation requires a CUDA device in this project; "
            "pass --adapter-arg device='\"cuda\"' on a CUDA-enabled host"
        )
    return device


def _checkpoint_model_name(
    checkpoint: Mapping[str, Any],
    *,
    use_mamba: bool,
    override: Any,
) -> str:
    if override is not None:
        model_name = str(override).strip()
    else:
        metadata = checkpoint.get("experiment_metadata", {})
        metadata_name = (
            metadata.get("model_name")
            if isinstance(metadata, Mapping)
            else None
        )
        checkpoint_name = checkpoint.get("model_name")
        model_name = str(
            metadata_name
            or checkpoint_name
            or ("ARIADNE+Mamba" if use_mamba else "ARIADNE")
        ).strip()
    if not model_name:
        raise ValueError("model_name cannot be empty")
    return model_name


class ARIADNEAdapter:
    """Read-only policy adapter implementing one complete exploration episode."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        deterministic: bool,
        adapter_args: Mapping[str, Any],
    ):
        unknown_arguments = sorted(
            set(adapter_args).difference(_ADAPTER_ARGUMENTS)
        )
        if unknown_arguments:
            raise ValueError(
                f"unsupported ARIADNE adapter arguments: {unknown_arguments}"
            )
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a boolean")
        # Loop Recovery V4 is part of the deployed 3.1 controller, rather than
        # an optional training-only aid.  Pure-policy evaluation remains
        # available as an explicit ablation with enable_recovery=false.
        enable_recovery = adapter_args.get("enable_recovery", True)
        if not isinstance(enable_recovery, bool):
            raise TypeError("enable_recovery must be a boolean")

        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"checkpoint does not exist: {self.checkpoint_path}"
            )
        self.deterministic = deterministic
        self.recovery_enabled = enable_recovery
        checkpoint = _load_checkpoint(self.checkpoint_path)
        self.use_mamba = bool(checkpoint.get("use_mamba", False))
        self.device = _resolve_device(
            adapter_args.get("device"),
            self.use_mamba,
        )

        node_input_dim, embedding_dim = _network_dimensions(checkpoint)
        mamba = _mamba_configuration(checkpoint, self.use_mamba)
        self.policy_net = PolicyNet(
            node_input_dim,
            embedding_dim,
            use_mamba=self.use_mamba,
            mamba_seq_len=mamba["sequence_length"],
            mamba_d_model=mamba["d_model"],
            mamba_d_state=mamba["d_state"],
            mamba_d_conv=mamba["d_conv"],
            mamba_expand=mamba["expand"],
            fusion_hidden_dim=mamba["fusion_hidden_dim"],
            action_embedding_dim=mamba["action_embedding_dim"],
            history_graph_batch_size=HISTORY_GRAPH_BATCH_SIZE,
            skip_padded_history_graphs=SKIP_PADDED_HISTORY_GRAPHS,
            trim_history_graph_nodes=TRIM_HISTORY_GRAPH_NODES,
            use_history_checkpoint=USE_HISTORY_CHECKPOINT,
        )
        policy_state = _normalize_state_dict(
            checkpoint["policy_model"]
        )
        self.policy_net.load_state_dict(policy_state, strict=True)
        self.policy_net.to(self.device)
        self.policy_net.eval()

        self.model_name = _checkpoint_model_name(
            checkpoint,
            use_mamba=self.use_mamba,
            override=adapter_args.get("model_name"),
        )
        if "model_seed" in adapter_args:
            self.model_seed = _non_negative_integer(
                adapter_args["model_seed"],
                "model_seed",
            )
        else:
            self.model_seed = checkpoint_model_seed(checkpoint)
            if self.model_seed < 0:
                raise ValueError(
                    "checkpoint does not record a non-negative training seed; "
                    "pass --adapter-arg model_seed=<integer>"
                )

        # A full training checkpoint also contains critics and optimizer state.
        # They are neither constructed nor retained by the read-only adapter.
        del policy_state
        del checkpoint

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_seed": self.model_seed,
            "checkpoint": str(self.checkpoint_path),
            "use_mamba": self.use_mamba,
            "device": str(self.device),
            "deterministic": self.deterministic,
            "recovery_enabled": self.recovery_enabled,
            "recovery_controller": (
                "local_revisit_recovery_v4"
                if self.recovery_enabled
                else None
            ),
            "recovery_configuration": (
                _recovery_configuration()
                if self.recovery_enabled
                else None
            ),
        }

    def _create_recovery_detector(self) -> Optional[LocalRevisitDetector]:
        if not self.recovery_enabled:
            return None
        return LocalRevisitDetector(
            enabled=True,
            **_recovery_configuration(),
        )

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _select_next_waypoint(
        self,
        robot: Agent,
        observation,
        history_batch,
        *,
        deterministic: bool,
    ):
        self._synchronize_device()
        started = time.perf_counter()
        try:
            return robot.select_next_waypoint(
                observation,
                history_batch=history_batch,
                deterministic=deterministic,
            )
        finally:
            self._synchronize_device()
            self._latest_inference_seconds = time.perf_counter() - started

    @staticmethod
    def _validate_config(
        config: Mapping[str, Any],
        default_deterministic: bool,
    ) -> tuple[float, int, Optional[float], bool]:
        if not isinstance(config, Mapping):
            raise TypeError("evaluation config must be a mapping")
        try:
            target_coverage = float(config["target_coverage"])
            max_steps = _positive_integer(config["max_steps"], "max_steps")
        except KeyError as exc:
            raise KeyError(
                f"evaluation config is missing {exc.args[0]!r}"
            ) from exc
        if (
            not math.isfinite(target_coverage)
            or not 0 < target_coverage <= 1
        ):
            raise ValueError("target_coverage must be finite and in (0, 1]")

        max_path_value = config.get("max_path_length")
        max_path_length = (
            None if max_path_value is None else float(max_path_value)
        )
        if max_path_length is not None and (
            not math.isfinite(max_path_length) or max_path_length <= 0
        ):
            raise ValueError(
                "max_path_length must be a positive finite value or None"
            )

        deterministic = config.get(
            "deterministic",
            default_deterministic,
        )
        if not isinstance(deterministic, bool):
            raise TypeError("config deterministic must be a boolean")
        return (
            target_coverage,
            max_steps,
            max_path_length,
            deterministic,
        )

    @torch.no_grad()
    def evaluate_episode(
        self,
        *,
        map_path,
        map_id,
        eval_seed,
        start_seed,
        episode_id,
        config: dict,
    ) -> Dict[str, Any]:
        """Evaluate one map/start pair without creating any training state."""

        episode_started = time.perf_counter()
        eval_seed = _non_negative_integer(eval_seed, "eval_seed")
        start_seed = _non_negative_integer(start_seed, "start_seed")
        episode_id = _non_negative_integer(episode_id, "episode_id")
        (
            target_coverage,
            max_steps,
            max_path_length,
            deterministic,
        ) = self._validate_config(config, self.deterministic)
        seed_everything(eval_seed, deterministic=deterministic)
        self.policy_net.eval()

        environment = Env(
            episode_id,
            plot=False,
            map_path=Path(map_path),
            start_seed=start_seed,
        )
        start_location = environment.robot_location.copy()
        start_cell = environment.start_cell.copy()
        robot = Agent(self.policy_net, self.device, plot=False)
        robot.update_planning_state(
            environment.belief_info,
            environment.robot_location,
        )
        observation = robot.get_observation()

        initial_coverage = float(environment.get_exploration_rate())
        coverage_curve: List[Dict[str, Any]] = [
            {
                "step": 0,
                "path_length": 0.0,
                "coverage": initial_coverage,
            }
        ]
        path_length_to_target = math.nan
        steps_to_target = math.nan
        environment_success = bool(robot.utility.sum() == 0)
        termination_reason: Optional[str] = None
        completed_steps = 0
        inference_times: List[float] = []
        robot_heading = 0.0
        trajectory = (
            EpisodeTrajectory(episode_id=episode_id)
            if self.use_mamba
            else None
        )
        recovery_detector = self._create_recovery_detector()
        recovery_trigger_sources: Counter[str] = Counter()
        recovery_target_types: Counter[str] = Counter()
        recovery_terminal_exit_reason: Optional[str] = None
        recovery_active_at_termination = False

        if coverage_success(initial_coverage, target_coverage):
            path_length_to_target = 0.0
            steps_to_target = 0
            termination_reason = "target_reached"
        elif environment_success:
            termination_reason = "environment_complete"

        while termination_reason is None and completed_steps < max_steps:
            current_observation = observation
            coverage_before = float(environment.get_exploration_rate())
            current_frontiers = get_frontier_in_map(
                environment.belief_info
            )
            frontier_before = len(current_frontiers)
            robot_pose = torch.tensor(
                [
                    environment.robot_location[0],
                    environment.robot_location[1],
                    robot_heading,
                ],
                dtype=torch.float32,
            )

            is_recovery_action = False
            recovery = None
            if recovery_detector is not None:
                trigger_count_before = (
                    recovery_detector.recovery_trigger_count
                )
                recovery = recovery_detector.begin_step(
                    robot.location,
                    robot.node_manager,
                    frontier_points=current_frontiers,
                    exploration_complete=environment_success,
                )
                is_recovery_action = recovery.is_recovery_action
                if (
                    recovery_detector.recovery_trigger_count
                    > trigger_count_before
                ):
                    if recovery.trigger_source is not None:
                        recovery_trigger_sources[
                            str(recovery.trigger_source)
                        ] += 1
                    if recovery.target_type is not None:
                        recovery_target_types[
                            str(recovery.target_type)
                        ] += 1

            history_batch = None
            if trajectory is not None and not is_recovery_action:
                history_batch = build_history_sequence(
                    trajectory,
                    sequence_length=self.policy_net.mamba_seq_len,
                )
                if history_batch is not None:
                    history_batch = history_batch.to(self.device)

            if is_recovery_action:
                next_location = recovery.next_waypoint
                action_index = robot.action_index_for_waypoint(
                    observation,
                    next_location,
                )
            else:
                self._latest_inference_seconds = math.nan
                try:
                    next_location, action_index = self._select_next_waypoint(
                        robot,
                        observation,
                        history_batch,
                        deterministic=deterministic,
                    )
                except NoValidActionError:
                    if math.isfinite(self._latest_inference_seconds):
                        inference_times.append(
                            self._latest_inference_seconds
                        )
                    termination_reason = "no_valid_action"
                    break
                inference_times.append(self._latest_inference_seconds)

            movement = next_location - environment.robot_location
            next_heading = float(np.arctan2(movement[1], movement[0]))
            reward = environment.step(next_location)
            completed_steps += 1

            robot.update_planning_state(
                environment.belief_info,
                environment.robot_location,
            )
            environment_success = bool(robot.utility.sum() == 0)
            if environment_success:
                reward += 20
            observation = robot.get_observation()

            coverage_after = float(environment.explored_rate)
            frontier_after = len(
                get_frontier_in_map(environment.belief_info)
            )
            delta_coverage = coverage_after - coverage_before
            delta_frontier = frontier_after - frontier_before
            if recovery_detector is not None:
                recovery_detector.record_coverage_gain(delta_coverage)

            if trajectory is not None:
                trajectory.append(
                    TrajectoryStep(
                        observation=detach_observation(
                            current_observation
                        ),
                        action=int(action_index.item()),
                        reward=float(reward),
                        next_observation=detach_observation(observation),
                        robot_pose=robot_pose,
                        delta_coverage=float(delta_coverage),
                        delta_frontier=int(delta_frontier),
                        done=environment_success,
                        step_index=completed_steps - 1,
                        is_recovery_action=is_recovery_action,
                    )
                )
            robot_heading = next_heading

            path_length = float(environment.travel_dist)
            coverage_curve.append(
                {
                    "step": completed_steps,
                    "path_length": path_length,
                    "coverage": coverage_after,
                }
            )
            reached_target = coverage_success(
                coverage_after,
                target_coverage,
            )
            if reached_target and math.isnan(path_length_to_target):
                path_length_to_target = path_length
                steps_to_target = completed_steps

            if reached_target:
                termination_reason = "target_reached"
            elif environment_success:
                termination_reason = "environment_complete"
            elif (
                max_path_length is not None
                and path_length >= max_path_length
            ):
                termination_reason = "max_path_length"
            elif completed_steps >= max_steps:
                termination_reason = "max_steps"

        if termination_reason is None:
            termination_reason = "max_steps"

        if recovery_detector is not None:
            recovery_active_at_termination = recovery_detector.recovery_mode
            recovery_terminal_exit_reason = recovery_detector.end_episode(
                environment.robot_location
            )

        path_length = float(environment.travel_dist)
        trajectory_length = trajectory_path_length(
            environment.robot_trajectory
        )
        path_error = abs(path_length - trajectory_length)
        path_tolerance = 1e-9 * max(1.0, path_length)
        if path_error > path_tolerance:
            raise RuntimeError(
                "environment travel distance disagrees with the recorded "
                f"robot trajectory: accumulated={path_length}, "
                f"trajectory={trajectory_length}, error={path_error}"
            )
        if len(environment.robot_trajectory) != completed_steps + 1:
            raise RuntimeError(
                "robot trajectory length does not match completed steps: "
                f"positions={len(environment.robot_trajectory)}, "
                f"steps={completed_steps}"
            )

        final_coverage = float(environment.get_exploration_rate())
        primary_success = coverage_success(
            final_coverage,
            target_coverage,
        )
        if not primary_success:
            path_length_to_target = math.nan
            steps_to_target = math.nan
        mean_inference_ms = (
            float(np.mean(inference_times) * 1000)
            if inference_times
            else math.nan
        )
        max_inference_ms = (
            float(np.max(inference_times) * 1000)
            if inference_times
            else math.nan
        )

        episode = {
            "start_x": float(start_location[0]),
            "start_y": float(start_location[1]),
            "start_cell_x": int(start_cell[0]),
            "start_cell_y": int(start_cell[1]),
            "start_resample_count": int(
                environment.start_resample_count
            ),
            "success": primary_success,
            "coverage_success": primary_success,
            "environment_success": environment_success,
            "termination_reason": termination_reason,
            "path_length": path_length,
            "steps": completed_steps,
            "initial_coverage": initial_coverage,
            "final_coverage": final_coverage,
            "target_coverage": target_coverage,
            "path_length_to_target": path_length_to_target,
            "steps_to_target": steps_to_target,
            "elapsed_time_seconds": time.perf_counter() - episode_started,
            "mean_inference_time_ms": mean_inference_ms,
            "max_inference_time_ms": max_inference_ms,
            "trajectory_path_length": trajectory_length,
            "path_length_validation_error": path_error,
            "recovery_enabled": self.recovery_enabled,
            "recovery_trigger_count": (
                recovery_detector.recovery_trigger_count
                if recovery_detector is not None
                else 0
            ),
            "recovery_steps": (
                recovery_detector.recovery_steps
                if recovery_detector is not None
                else 0
            ),
            "recovery_step_ratio": (
                recovery_detector.recovery_steps / completed_steps
                if recovery_detector is not None and completed_steps
                else 0.0
            ),
            "recovery_target_reached_count": (
                recovery_detector.target_reached_count
                if recovery_detector is not None
                else 0
            ),
            "recovery_active_at_termination": (
                recovery_active_at_termination
            ),
            "recovery_terminal_exit_reason": (
                recovery_terminal_exit_reason or ""
            ),
            "recovery_regional_trigger_count": recovery_trigger_sources[
                "regional"
            ],
            "recovery_point_repeat_trigger_count": (
                recovery_trigger_sources["point_repeat"]
            ),
            "recovery_combined_trigger_count": recovery_trigger_sources[
                "regional+point_repeat"
            ],
            "recovery_frontier_target_count": recovery_target_types[
                "frontier"
            ],
            "recovery_filter_fallback_count": recovery_target_types[
                "frontier_filter_fallback"
            ],
            "recovery_graph_fallback_count": recovery_target_types[
                "graph_fallback"
            ],
            "error": "",
        }
        return {
            "episode": episode,
            "coverage_curve": coverage_curve,
        }


def create_adapter(
    checkpoint: Path,
    deterministic: bool,
    adapter_args: dict,
) -> ARIADNEAdapter:
    """Stable factory loaded by ``scripts/evaluate_on_split.py``."""

    if not isinstance(adapter_args, Mapping):
        raise TypeError("adapter_args must be a mapping")
    return ARIADNEAdapter(
        checkpoint_path=Path(checkpoint),
        deterministic=deterministic,
        adapter_args=adapter_args,
    )
