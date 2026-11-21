"""Run Nearest Frontier in the workspace's shared grid evaluator."""
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from env import Env
from experiment_utils import coverage_success, sha256_file, trajectory_path_length
from parameter import FREE, UNKNOWN

from .planner import nearest_frontier_path, next_waypoint


class NearestFrontierAdapter:
    def __init__(self, checkpoint, deterministic, adapter_args):
        if not deterministic or adapter_args:
            raise ValueError("Nearest Frontier is deterministic and uses its specification JSON")
        self.specification = json.loads(Path(checkpoint).read_text())
        if (self.specification.get("artifact_type") != "untrained_planner_specification"
                or self.specification.get("planner") != "nearest_frontier"
                or self.specification.get("connectivity") != 4):
            raise ValueError("Expected the four-connected Nearest Frontier specification")
        self.max_segment = float(self.specification["max_segment_length_m"])
        if not math.isfinite(self.max_segment) or self.max_segment <= 0:
            raise ValueError("max_segment_length_m must be positive and finite")

    def metadata(self):
        return {
            **self.specification,
            "model_seed": 0,
            "trained": False,
            "deterministic": True,
            "action_selection": "deterministic explicit planner",
            "distance_metric": "shortest four-connected path through observed free cells",
            "recovery_enabled": False,
            "source_sha256": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("planner.py", "adapter.py")
            },
        }

    def evaluate_episode(self, *, map_path, map_id, eval_seed, start_seed, episode_id, config):
        started = perf_counter()
        env = Env(episode_id, plot=False, map_path=map_path, start_seed=start_seed)
        target = float(config["target_coverage"])
        max_path = config["max_path_length"]
        max_cells = math.floor(self.max_segment / env.cell_size + 1e-9)
        if max_cells < 1:
            raise ValueError("The segment cap must allow at least one grid edge")
        origin = np.array([env.belief_origin_x, env.belief_origin_y])
        initial = coverage = float(env.get_exploration_rate())
        curve = [{"step": 0, "path_length": 0.0, "coverage": initial}]
        steps = 0
        timings = []
        reason = "max_steps"
        while True:
            if coverage_success(coverage, target):
                reason = "target_reached"
                break
            if max_path is not None and env.travel_dist >= max_path:
                reason = "max_path_length"
                break
            if steps >= config["max_steps"]:
                break

            decision_started = perf_counter()
            x, y = map(int, env.robot_cell)
            path = nearest_frontier_path(env.robot_belief, (y, x), free=FREE, unknown=UNKNOWN)
            waypoint = next_waypoint(path, max_cells)
            timings.append((perf_counter() - decision_started) * 1000)
            if waypoint is None:
                reason = "environment_complete"
                break
            row, col = waypoint
            env.step(origin + env.cell_size * np.array([col, row]))
            steps += 1
            coverage = float(env.get_exploration_rate())
            curve.append({"step": steps, "path_length": float(env.travel_dist), "coverage": coverage})

        success = coverage_success(coverage, target)
        distance = float(env.travel_dist)
        trajectory_distance = trajectory_path_length(env.robot_trajectory)
        if len(env.robot_trajectory) != steps + 1 or not math.isclose(distance, trajectory_distance, abs_tol=1e-8):
            raise RuntimeError("Recorded trajectory disagrees with step count or path length")
        return {
            "episode": {
                "start_x": 0.0, "start_y": 0.0,
                "start_cell_x": int(env.start_cell[0]),
                "start_cell_y": int(env.start_cell[1]),
                "start_resample_count": int(env.start_resample_count),
                "success": success, "coverage_success": success,
                "environment_success": reason == "environment_complete",
                "termination_reason": reason, "steps": steps, "path_length": distance,
                "initial_coverage": initial, "final_coverage": coverage,
                "target_coverage": target,
                "path_length_to_target": distance if success else math.nan,
                "steps_to_target": steps if success else math.nan,
                "elapsed_time_seconds": perf_counter() - started,
                "mean_inference_time_ms": float(np.mean(timings)) if timings else math.nan,
                "max_inference_time_ms": max(timings) if timings else math.nan,
                "trajectory_path_length": trajectory_distance,
                "path_length_validation_error": abs(distance - trajectory_distance),
                "recovery_enabled": False, "recovery_steps": 0,
                "recovery_trigger_count": 0, "error": "",
            },
            "coverage_curve": curve,
        }


def create_adapter(checkpoint, deterministic, adapter_args):
    return NearestFrontierAdapter(checkpoint, deterministic, adapter_args)
