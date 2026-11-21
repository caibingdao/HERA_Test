"""Run RH-NBVP 2D in the workspace's shared grid evaluator."""
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from env import Env
from experiment_utils import coverage_success, sha256_file, trajectory_path_length


from .planner import RecedingHorizonNBVP


class NBVPAdapter:
    def __init__(self, checkpoint, deterministic, adapter_args):
        if not deterministic or adapter_args:
            raise ValueError("RH-NBVP 2D is deterministic and uses its specification JSON")
        self.specification = json.loads(Path(checkpoint).read_text())
        if self.specification.get("planner") != "eth_asl_rh_nbvp_2d":
            raise ValueError("Expected the RH-NBVP planar specification")
        self.rng = None
        self.diagnostics = []

    def metadata(self):
        return {
            **self.specification,
            "model_seed": 0,
            "trained": False,
            "deterministic": True,
            "action_selection": "seeded_rh_nbvp_2d",
            "gain": "visible unknown area, discounted by cumulative root-to-node distance",
            "native_diagnostics": self.diagnostics,
            "recovery_enabled": False,
            "source_sha256": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("planner.py", "adapter.py", "visibility.cpp")
            },
        }

    def evaluate_episode(self, *, map_path, map_id, eval_seed, start_seed, episode_id, config):
        started = perf_counter()
        env = Env(episode_id, plot=False, map_path=map_path, start_seed=start_seed)
        target = float(config["target_coverage"])
        max_path = config["max_path_length"]
        if self.rng is None:
            self.rng = np.random.default_rng(eval_seed)
        planner = RecedingHorizonNBVP(self.specification, env.cell_size, env.sensor_range, self.rng)
        backtracks = 0
        origin = np.array([env.belief_origin_x, env.belief_origin_y])
        initial = coverage = float(env.get_exploration_rate())
        curve = [{"step": 0, "path_length": 0.0, "coverage": initial}]
        steps = 0
        timings = []
        reason = "max_steps"
        planner_stop = None
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
            position = (env.robot_location - origin) / env.cell_size
            waypoint, status = planner.plan(env.robot_belief, position)
            timings.append((perf_counter() - decision_started) * 1000)
            if waypoint is None:
                planner_stop = status
                reason = "no_valid_action"
                break
            backtracks += int(status == "upstream_backtrack")
            env.step(origin + env.cell_size * waypoint)
            steps += 1
            coverage = float(env.get_exploration_rate())
            curve.append({"step": steps, "path_length": float(env.travel_dist), "coverage": coverage})

        success = coverage_success(coverage, target)
        distance = float(env.travel_dist)
        trajectory_distance = trajectory_path_length(env.robot_trajectory)
        if len(env.robot_trajectory) != steps + 1 or not math.isclose(distance, trajectory_distance, abs_tol=1e-8):
            raise RuntimeError("Recorded trajectory disagrees with step count or path length")
        self.diagnostics.append({"map_id": map_id, "native_backtracks": backtracks,
                                 "termination_reason": reason, "planner_stop": planner_stop})
        return {
            "episode": {
                "start_x": 0.0, "start_y": 0.0,
                "start_cell_x": int(env.start_cell[0]),
                "start_cell_y": int(env.start_cell[1]),
                "start_resample_count": int(env.start_resample_count),
                "success": success, "coverage_success": success,
                "environment_success": False,
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
    return NBVPAdapter(checkpoint, deterministic, adapter_args)
