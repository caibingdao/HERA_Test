"""Native, untrained planners evaluated in the exact Mamba test environment.

The evaluator's checkpoint argument points to a JSON planner specification,
not neural weights. Each planner keeps its native graph and decision rule.
Repository imports are isolated so their generic module names cannot replace
the shared evaluator's environment, parameters, or Mamba classes.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from env import Env
import parameter as shared_parameters
from experiment_utils import coverage_success, sha256_file, trajectory_path_length
from utils import check_collision

WORKSPACE = Path(__file__).resolve().parents[3]
GEOMETRY_FIELDS = (
    "CELL_SIZE", "NODE_RESOLUTION", "FRONTIER_CELL_SIZE", "FREE", "OCCUPIED",
    "UNKNOWN", "SENSOR_RANGE", "UTILITY_RANGE", "MIN_UTILITY", "UPDATING_MAP_SIZE",
)


def load_native_agent(root: Path):
    """Load a checkout's modules, restoring all shared module bindings."""
    names = {path.stem for path in root.glob("*.py")}
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    old_path = sys.path[:]
    sys.path.insert(0, str(root))
    try:
        parameters = importlib.import_module("parameter")
        agent = importlib.import_module("agent")
        return agent.Agent, parameters
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


class ClassicalAdapter:
    def __init__(self, checkpoint, deterministic, adapter_args):
        if not deterministic:
            raise ValueError("classical baseline protocol is deterministic")
        if adapter_args:
            raise ValueError("configure the planner in its JSON specification")
        self.specification = json.loads(Path(checkpoint).read_text())
        if self.specification["artifact_type"] != "untrained_planner_specification":
            raise ValueError("expected a planner specification, not a model checkpoint")
        self.kind = self.specification["planner"]
        self.tare_episode_diagnostics = []
        if self.kind not in ("utility", "nbvp", "tare_local"):
            raise ValueError(f"unknown planner: {self.kind}")
        root = (WORKSPACE / self.specification["repository"]).resolve()
        if root.parent != WORKSPACE:
            raise ValueError("planner checkout must be a sibling in the workspace")
        for name, digest in self.specification["source_sha256"].items():
            if sha256_file(root / name) != digest:
                raise ValueError(f"planner source changed after specification: {name}")
        self.agent_class, parameters = load_native_agent(root)
        for field in GEOMETRY_FIELDS:
            if getattr(parameters, field) != getattr(shared_parameters, field):
                raise ValueError(f"native and Mamba geometry differ: {field}")
        self.native_parameters = {
            key: value for key, value in vars(parameters).items()
            if key.isupper() and isinstance(value, (str, bool, int, float))
        }
        if self.kind == "tare_local" and parameters.PLANNER_TYPE != "tare_local":
            raise ValueError("TARE checkout is not configured for the classical planner")

    def metadata(self):
        return {
            "model_name": self.specification["model_name"],
            # Required by the shared CSV schema; this is not a training seed.
            "model_seed": 0,
            "trained": False,
            "artifact_type": "untrained_planner_specification",
            "planner": self.kind,
            "lambda_value": self.specification.get("lambda_value"),
            "source_sha256": self.specification["source_sha256"],
            "source_commit": self.specification["source_commit"],
            "native_parameters": self.native_parameters,
            "environment": "shared Mamba split-aware Env",
            "geometry": {key: getattr(shared_parameters, key) for key in GEOMETRY_FIELDS},
            "recovery_enabled": False,
            "recovery_policy": "native planner behavior; no Mamba recovery overlay",
            "utility_comparison": "log(A) - lambda * graph_path_length",
            "action_selection": ("official_tare_seeded_randomized_coverage_and_tsp"
                                 if self.kind == "tare_local" else "deterministic explicit planner"),
            "official_source": self.specification.get("official_source"),
            "adaptation": self.specification.get("adaptation"),
            "tare_episode_diagnostics": self.tare_episode_diagnostics if self.kind == "tare_local" else None,
        }

    def evaluate_episode(self, *, map_path, map_id, eval_seed, start_seed, episode_id, config):
        started = time.perf_counter()
        target = float(config["target_coverage"])
        max_steps = int(config["max_steps"])
        max_path = config["max_path_length"]
        environment = Env(episode_id, plot=False, map_path=map_path, start_seed=start_seed)
        if self.kind == "utility":
            robot = self.agent_class(lambda_value=self.specification["lambda_value"], plot=False)
        elif self.kind == "nbvp":
            robot = self.agent_class(plot=False)
        else:
            robot = self.agent_class(None, device="cpu", plot=False)
        robot.update_planning_state(environment.belief_info, environment.robot_location)
        initial = float(environment.get_exploration_rate())
        curve = [{"step": 0, "path_length": 0.0, "coverage": initial}]
        steps = 0
        timings = []
        reason = None
        error = ""
        environment_success = self.kind != "tare_local" and bool(robot.utility.sum() == 0)
        if coverage_success(initial, target):
            reason = "target_reached"
        elif environment_success:
            reason = "environment_complete"

        while reason is None and steps < max_steps:
            decision_started = time.perf_counter()
            if self.kind == "tare_local":
                # Native classical selection does not consume neural observations.
                plan = robot.local_planner.plan_next_action(
                    node_coords=robot.node_coords, current_index=robot.current_index,
                    neighbor_indices=robot.neighbor_indices, node_manager=robot.node_manager,
                    current_frontiers=robot.frontier, map_info=robot.map_info,
                )
                next_location = (
                    None if plan.next_node_index is None
                    else robot.node_coords[plan.next_node_index]
                )
            else:
                next_location, _ = robot.select_next_waypoint()
            timings.append(time.perf_counter() - decision_started)
            if next_location is None:
                reason = "no_valid_action"
                break
            next_location = np.asarray(next_location, dtype=float)
            if next_location.shape != (2,) or not np.isfinite(next_location).all():
                raise ValueError("planner returned a non-finite or malformed waypoint")
            current = environment.robot_location
            node = robot.node_manager.nodes_dict.find(tuple(np.round(current, 1)))
            neighbors = {tuple(np.round(point, 1)) for point in node.data.neighbor_set}
            if tuple(np.round(next_location, 1)) not in neighbors:
                raise ValueError("planner returned a waypoint outside the current graph neighbors")
            if np.allclose(current, next_location, rtol=0, atol=1e-9):
                reason = "no_valid_action"
                break
            if check_collision(current, next_location, environment.belief_info):
                reason = "collision"
                break
            environment.step(next_location)
            steps += 1
            robot.update_planning_state(environment.belief_info, environment.robot_location)
            environment_success = self.kind != "tare_local" and bool(robot.utility.sum() == 0)
            coverage = float(environment.get_exploration_rate())
            curve.append({"step": steps, "path_length": float(environment.travel_dist), "coverage": coverage})
            if coverage_success(coverage, target):
                reason = "target_reached"
            elif environment_success:
                reason = "environment_complete"
            elif max_path is not None and environment.travel_dist >= max_path:
                reason = "max_path_length"
        reason = reason or "max_steps"
        coverage = float(environment.get_exploration_rate())
        success = coverage_success(coverage, target)
        distance = float(environment.travel_dist)
        trajectory_distance = trajectory_path_length(environment.robot_trajectory)
        if len(environment.robot_trajectory) != steps + 1 or not math.isclose(distance, trajectory_distance, abs_tol=1e-8):
            raise RuntimeError("path length or step count disagrees with the recorded trajectory")
        if self.kind == "tare_local":
            diagnostic = robot.local_planner.last_diagnostics
            reachable = diagnostic.get("reachable_nonrobot_frontier_viewpoints", diagnostic.get("reachable_frontier_viewpoints", 0))
            if reason == "no_valid_action" and reachable:
                raise RuntimeError("TARE stopped despite reachable informative viewpoints")
            self.tare_episode_diagnostics.append({
                "map_id": map_id, "steps": steps, "termination_reason": reason,
                "reachable_frontier_viewpoints_at_end": reachable,
                "counters": dict(robot.local_planner.counters),
                "ordinary_point_count_at_end": diagnostic.get("ordinary_point_count", 0),
            })
        return {
            "episode": {
                "start_x": 0.0, "start_y": 0.0,
                "start_cell_x": int(environment.start_cell[0]),
                "start_cell_y": int(environment.start_cell[1]),
                "start_resample_count": int(environment.start_resample_count),
                "success": success, "coverage_success": success,
                "environment_success": environment_success, "termination_reason": reason,
                "path_length": distance, "steps": steps,
                "initial_coverage": initial, "final_coverage": coverage,
                "target_coverage": target,
                "path_length_to_target": distance if success else math.nan,
                "steps_to_target": steps if success else math.nan,
                "elapsed_time_seconds": time.perf_counter() - started,
                "mean_inference_time_ms": float(np.mean(timings) * 1000) if timings else math.nan,
                "max_inference_time_ms": max(timings) * 1000 if timings else math.nan,
                "trajectory_path_length": trajectory_distance,
                "path_length_validation_error": abs(distance - trajectory_distance),
                "recovery_enabled": False, "error": error,
            },
            "coverage_curve": curve,
        }


def create_adapter(checkpoint, deterministic, adapter_args):
    return ClassicalAdapter(checkpoint, deterministic, adapter_args)
