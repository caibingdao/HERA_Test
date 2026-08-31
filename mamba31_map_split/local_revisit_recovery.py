import heapq
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


def _coord_key(coords):
    coords = np.asarray(coords, dtype=float).reshape(2)
    return float(coords[0]), float(coords[1])


@dataclass(frozen=True)
class RecoveryDecision:
    next_waypoint: object = None
    is_recovery_action: bool = False
    revisit_count: int = 0
    revisit_score: float = 0.0
    recent_mean_coverage_gain: float = 0.0
    revisit_trigger_counter: int = 0
    point_repeat_max_count: int = 0
    point_repeat_top_k_count: int = 0
    point_repeat_unique_count: int = 0
    point_repeat_trigger_counter: int = 0
    regional_repeat_detected: bool = False
    point_repeat_detected: bool = False
    low_progress_detected: bool = False
    trigger_source: object = None
    recovery_mode: bool = False
    escape_target: object = None
    escape_path_length: int = 0
    exit_reason: object = None
    target_reached: bool = False
    candidate_num: int = 0
    selected_group_id: object = None
    frontier_count: int = 0
    navigation_distance: object = None
    loop_distance: object = None
    utility: object = None
    target_type: object = None
    escape_candidates: tuple = ()


@dataclass(frozen=True)
class FrontierGroup:
    group_id: int
    frontier_points: tuple
    frontier_count: int
    centroid: tuple


@dataclass(frozen=True)
class EscapeCandidate:
    group_id: int
    frontier_count: int
    centroid: tuple
    target_node: tuple
    navigation_distance: float
    loop_distance: float
    utility: float
    passed_min_distance: bool
    passed_min_loop_distance: bool

    def as_log_dict(self):
        return {
            "group_id": self.group_id,
            "frontier_count": self.frontier_count,
            "centroid": list(self.centroid),
            "D": self.navigation_distance,
            "L": self.loop_distance,
            "utility": self.utility,
            "target_node": list(self.target_node),
            "passed_min_distance": self.passed_min_distance,
            "passed_min_loop_distance": self.passed_min_loop_distance,
        }


class LocalRevisitDetector:
    """Episode-local revisit detector and committed escape controller."""

    def __init__(
        self,
        enabled,
        history_length,
        graph_radius,
        score_threshold,
        exit_threshold,
        progress_window,
        coverage_threshold,
        trigger_steps,
        min_count=None,
        goal_tolerance=0.0,
        max_recovery_steps=32,
        path_failure_limit=3,
        use_frontier_escape=False,
        escape_min_distance=12.0,
        escape_min_loop_distance=8.0,
        escape_distance_eps=1.0,
        frontier_cell_size=0.8,
        point_repeat_enabled=False,
        point_repeat_max_count=3,
        point_repeat_top_k=3,
        point_repeat_top_k_count=None,
        point_repeat_trigger_steps=None,
    ):
        if history_length <= 0 or progress_window <= 0 or trigger_steps <= 0:
            raise ValueError("History, progress, and trigger windows must be positive")
        if graph_radius < 0 or goal_tolerance < 0:
            raise ValueError("Radii and goal tolerance must be non-negative")
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        if max_recovery_steps <= 0 or path_failure_limit <= 0:
            raise ValueError("Recovery limits must be positive")
        if escape_min_distance < 0 or escape_min_loop_distance < 0:
            raise ValueError("Escape distance limits must be non-negative")
        if escape_distance_eps <= 0 or frontier_cell_size <= 0:
            raise ValueError("Escape epsilon and frontier cell size must be positive")
        if min_count is None:
            min_count = int(np.ceil(score_threshold * history_length))
        if not 1 <= min_count <= history_length:
            raise ValueError("min_count must be within the history window")
        if point_repeat_top_k_count is None:
            point_repeat_top_k_count = min(history_length, 8)
        if point_repeat_trigger_steps is None:
            point_repeat_trigger_steps = trigger_steps
        if point_repeat_trigger_steps <= 0:
            raise ValueError(
                "point_repeat_trigger_steps must be positive"
            )
        if point_repeat_enabled:
            if not 1 <= point_repeat_max_count <= history_length:
                raise ValueError(
                    "point_repeat_max_count must be within the history window"
                )
            if not 1 <= point_repeat_top_k <= history_length:
                raise ValueError(
                    "point_repeat_top_k must be within the history window"
                )
            if not 1 <= point_repeat_top_k_count <= history_length:
                raise ValueError(
                    "point_repeat_top_k_count must be within the history window"
                )

        self.enabled = bool(enabled)
        self.history_length = int(history_length)
        self.graph_radius = float(graph_radius)
        self.min_count = int(min_count)
        self.score_threshold = self.min_count / self.history_length
        # Kept as a public compatibility field; V2.1 never exits on this value.
        self.exit_threshold = float(exit_threshold)
        self.progress_window = int(progress_window)
        self.coverage_threshold = float(coverage_threshold)
        self.trigger_steps = int(trigger_steps)
        self.goal_tolerance = float(goal_tolerance)
        self.max_recovery_steps = int(max_recovery_steps)
        self.path_failure_limit = int(path_failure_limit)
        self.use_frontier_escape = bool(use_frontier_escape)
        self.escape_min_distance = float(escape_min_distance)
        self.escape_min_loop_distance = float(escape_min_loop_distance)
        self.escape_distance_eps = float(escape_distance_eps)
        self.frontier_cell_size = float(frontier_cell_size)
        self.point_repeat_enabled = bool(point_repeat_enabled)
        self.point_repeat_max_threshold = int(point_repeat_max_count)
        self.point_repeat_top_k = int(point_repeat_top_k)
        self.point_repeat_top_k_threshold = int(
            point_repeat_top_k_count
        )
        self.point_repeat_trigger_steps = int(
            point_repeat_trigger_steps
        )
        self.reset()

    def reset(self):
        self.position_history = deque(maxlen=self.history_length)
        self.coverage_gains = deque(maxlen=self.progress_window)
        self.visit_counts = Counter()
        self.trigger_counter = 0
        self.point_repeat_trigger_counter = 0
        self.recovery_mode = False
        self.recovery_trigger_count = 0
        self.recovery_steps = 0
        self.current_recovery_steps = 0
        self.target_reached_count = 0
        self.locked_escape_target = None
        self.escape_path = []
        self.path_failure_count = 0
        self.revisit_count = 0
        self.revisit_score = 0.0
        self.recent_mean_coverage_gain = 0.0
        self.point_repeat_max_count = 0
        self.point_repeat_top_k_count = 0
        self.point_repeat_unique_count = 0
        self.regional_repeat_detected = False
        self.point_repeat_detected = False
        self.low_progress_detected = False
        self.trigger_source = None
        self.last_neighborhood = set()
        self.last_exit_reason = None
        self.selected_escape_candidate = None
        self.selected_escape_metrics = {}
        self.escape_candidate_logs = ()
        self.escape_target_type = None

    @property
    def escape_target(self):
        """Compatibility alias for the V2.1 locked world-coordinate target."""
        return self.locked_escape_target

    def record_coverage_gain(self, delta_coverage):
        if self.enabled:
            self.coverage_gains.append(float(delta_coverage))

    def _reset_trigger_counters(self):
        self.trigger_counter = 0
        self.point_repeat_trigger_counter = 0

    def _update_trigger_counters(
        self,
        regional_abnormal,
        point_repeat_abnormal,
    ):
        """Advance the two evidence streaks without allowing OR chaining."""
        self.trigger_counter = (
            self.trigger_counter + 1 if regional_abnormal else 0
        )
        self.point_repeat_trigger_counter = (
            self.point_repeat_trigger_counter + 1
            if point_repeat_abnormal else 0
        )
        regional_ready = self.trigger_counter >= self.trigger_steps
        point_repeat_ready = (
            self.point_repeat_trigger_counter
            >= self.point_repeat_trigger_steps
        )
        if regional_ready and point_repeat_ready:
            return "regional+point_repeat"
        if regional_ready:
            return "regional"
        if point_repeat_ready:
            return "point_repeat"
        return None

    @staticmethod
    def _all_node_keys(node_manager):
        return [
            _coord_key(node.data.coords)
            for node in node_manager.nodes_dict.__iter__()
        ]

    def _nearest_current_node(self, coords, node_manager):
        key = _coord_key(coords)
        if node_manager.check_node_exist_in_dict(key) is not None:
            return key
        nodes = self._all_node_keys(node_manager)
        if not nodes:
            return None
        return min(nodes, key=lambda node: np.linalg.norm(np.subtract(node, key)))

    def _map_history_to_current_graph(self, node_manager):
        return [
            mapped
            for mapped in (
                self._nearest_current_node(coords, node_manager)
                for coords in self.position_history
            )
            if mapped is not None
        ]

    def cluster_frontiers(self, frontier_points):
        """Cluster downsampled frontier points with physical 8-connectivity."""
        points = sorted({_coord_key(point) for point in frontier_points})
        if not points:
            return []

        points_array = np.asarray(points, dtype=float)
        unvisited = set(range(len(points)))
        groups = []
        tolerance = 1e-6
        while unvisited:
            seed = min(unvisited)
            unvisited.remove(seed)
            component = [seed]
            queue = deque([seed])
            while queue:
                current_index = queue.popleft()
                if not unvisited:
                    continue
                remaining = np.fromiter(sorted(unvisited), dtype=int)
                delta = np.abs(
                    points_array[remaining] - points_array[current_index]
                )
                adjacent_mask = np.logical_and(
                    delta[:, 0] <= self.frontier_cell_size + tolerance,
                    delta[:, 1] <= self.frontier_cell_size + tolerance,
                )
                adjacent = remaining[adjacent_mask].tolist()
                for neighbor_index in adjacent:
                    unvisited.remove(neighbor_index)
                    component.append(neighbor_index)
                    queue.append(neighbor_index)

            group_points = points_array[sorted(component)]
            group_id = len(groups)
            groups.append(
                FrontierGroup(
                    group_id=group_id,
                    frontier_points=tuple(
                        _coord_key(point) for point in group_points
                    ),
                    frontier_count=len(group_points),
                    centroid=_coord_key(np.mean(group_points, axis=0)),
                )
            )
        return groups

    @staticmethod
    def _reachable_nodes(distances, include_start=False):
        return [
            node for node, distance in distances.items()
            if distance < 1e8 and (include_start or distance > 0)
        ]

    def _loop_distance_lookup(
        self,
        current,
        node_manager,
        mapped_history,
    ):
        """Return graph distance from every node to the nearest loop node."""
        loop_nodes = set(mapped_history)
        loop_nodes.add(current)
        lookup = {
            node: 1e8 for node in self._all_node_keys(node_manager)
        }
        queue = []
        for loop_node in sorted(loop_nodes):
            if loop_node not in lookup:
                continue
            lookup[loop_node] = 0.0
            heapq.heappush(queue, (0.0, loop_node))

        while queue:
            distance, node = heapq.heappop(queue)
            if distance > lookup[node]:
                continue
            node_entry = node_manager.check_node_exist_in_dict(node)
            if node_entry is None:
                continue
            for neighbor_coords in node_entry.data.neighbor_set:
                neighbor = _coord_key(neighbor_coords)
                if neighbor not in lookup or neighbor == node:
                    continue
                edge_cost = float(
                    np.linalg.norm(np.subtract(neighbor, node))
                )
                candidate_distance = distance + edge_cost
                if candidate_distance < lookup[neighbor]:
                    lookup[neighbor] = candidate_distance
                    heapq.heappush(
                        queue,
                        (candidate_distance, neighbor),
                    )
        return lookup

    def _frontier_candidates(
        self,
        frontier_groups,
        distances,
        loop_distances,
    ):
        reachable_nodes = self._reachable_nodes(distances)
        candidates = []
        for group in frontier_groups:
            if not reachable_nodes:
                break
            centroid = np.asarray(group.centroid, dtype=float)
            target = min(
                reachable_nodes,
                key=lambda node: (
                    np.linalg.norm(np.subtract(node, centroid)),
                    distances[node],
                    node,
                ),
            )
            navigation_distance = float(distances[target])
            loop_distance = float(loop_distances.get(target, 0.0))
            utility = group.frontier_count / (
                navigation_distance + self.escape_distance_eps
            )
            candidates.append(
                EscapeCandidate(
                    group_id=group.group_id,
                    frontier_count=group.frontier_count,
                    centroid=group.centroid,
                    target_node=_coord_key(target),
                    navigation_distance=navigation_distance,
                    loop_distance=loop_distance,
                    utility=float(utility),
                    passed_min_distance=(
                        navigation_distance >= self.escape_min_distance
                    ),
                    passed_min_loop_distance=(
                        loop_distance >= self.escape_min_loop_distance
                    ),
                )
            )
        return candidates

    @staticmethod
    def _candidate_rank(candidate):
        return (
            candidate.utility,
            candidate.frontier_count,
            candidate.loop_distance,
            -candidate.navigation_distance,
            -candidate.group_id,
        )

    def _select_frontier_candidate(self, candidates):
        eligible = [
            candidate for candidate in candidates
            if candidate.passed_min_distance
            and candidate.passed_min_loop_distance
        ]
        if eligible:
            return max(eligible, key=self._candidate_rank), "frontier"
        if candidates:
            return (
                max(candidates, key=self._candidate_rank),
                "frontier_filter_fallback",
            )
        return None, None

    def _lock_target(
        self,
        target,
        path,
        target_type,
        candidate=None,
        candidate_logs=(),
        selection_metrics=None,
    ):
        path = [_coord_key(coords) for coords in path]
        if not path:
            self._reset_trigger_counters()
            return False
        self.locked_escape_target = _coord_key(target)
        self.escape_path = path
        self.recovery_mode = True
        self.recovery_trigger_count += 1
        self.current_recovery_steps = 0
        self.path_failure_count = 0
        self.last_exit_reason = None
        self.selected_escape_candidate = candidate
        self.selected_escape_metrics = dict(selection_metrics or {})
        self.escape_candidate_logs = tuple(candidate_logs)
        self.escape_target_type = target_type
        return True

    def _goal_reached(self, current):
        return (
            self.locked_escape_target is not None
            and np.linalg.norm(
                np.subtract(current, self.locked_escape_target)
            ) <= self.goal_tolerance
        )

    def _plan_to_locked_target(self, current, node_manager):
        """Plan to the same spatial target, remapping only its graph node."""
        if self.locked_escape_target is None:
            return "path_failure"
        graph_target = self._nearest_current_node(
            self.locked_escape_target,
            node_manager,
        )
        if graph_target is None:
            return "target_unreachable"
        dist, prev = node_manager.Dijkstra(current)
        if dist.get(graph_target, 1e8) >= 1e8:
            self.escape_path = []
            return "target_unreachable"
        try:
            path, _ = node_manager.get_Dijkstra_path_and_dist(
                dist,
                prev,
                graph_target,
            )
        except (KeyError, TypeError, ValueError):
            self.escape_path = []
            return "path_failure"
        self.escape_path = [_coord_key(coords) for coords in path]
        if not self.escape_path and not self._goal_reached(current):
            return "path_failure"
        self.path_failure_count = 0
        return None

    def _enter_nearest_node_recovery(
        self,
        dist,
        prev,
        node_manager,
    ):
        self.selected_escape_candidate = None
        self.selected_escape_metrics = {}
        self.escape_candidate_logs = ()
        self.escape_target_type = None
        candidates = [
            node for node, distance in dist.items()
            if distance < 1e8 and distance > self.graph_radius
        ]
        if not candidates:
            self._reset_trigger_counters()
            return False
        target = min(
            candidates,
            key=lambda node: (dist[node], self.visit_counts[node], node),
        )
        try:
            path, _ = node_manager.get_Dijkstra_path_and_dist(
                dist,
                prev,
                target,
            )
        except (KeyError, TypeError, ValueError):
            self._reset_trigger_counters()
            return False
        return self._lock_target(
            target,
            path,
            target_type="nearest_node",
            selection_metrics={
                "navigation_distance": float(dist[target]),
            },
        )

    def _enter_frontier_recovery(
        self,
        current,
        dist,
        prev,
        node_manager,
        frontier_points,
        mapped_history,
        exploration_complete,
    ):
        self.selected_escape_candidate = None
        self.selected_escape_metrics = {}
        self.escape_candidate_logs = ()
        self.escape_target_type = None
        frontier_groups = self.cluster_frontiers(
            () if frontier_points is None else frontier_points
        )
        loop_distances = self._loop_distance_lookup(
            current,
            node_manager,
            mapped_history,
        )
        candidates = self._frontier_candidates(
            frontier_groups,
            dist,
            loop_distances,
        )
        candidate_logs = tuple(
            candidate.as_log_dict() for candidate in candidates
        )
        selected, target_type = self._select_frontier_candidate(candidates)

        if selected is not None:
            target = selected.target_node
            try:
                path, _ = node_manager.get_Dijkstra_path_and_dist(
                    dist,
                    prev,
                    target,
                )
            except (KeyError, TypeError, ValueError):
                self._reset_trigger_counters()
                return False
            return self._lock_target(
                target,
                path,
                target_type=target_type,
                candidate=selected,
                candidate_logs=candidate_logs,
            )

        if exploration_complete:
            self.last_exit_reason = "exploration_complete"
            self._reset_trigger_counters()
            return False

        reachable_nodes = self._reachable_nodes(dist)
        if not reachable_nodes:
            self._reset_trigger_counters()
            return False
        target = max(
            reachable_nodes,
            key=lambda node: (
                loop_distances.get(node, -1.0),
                dist[node],
                node,
            ),
        )
        try:
            path, _ = node_manager.get_Dijkstra_path_and_dist(
                dist,
                prev,
                target,
            )
        except (KeyError, TypeError, ValueError):
            self._reset_trigger_counters()
            return False
        return self._lock_target(
            target,
            path,
            target_type="graph_fallback",
            candidate_logs=candidate_logs,
            selection_metrics={
                "navigation_distance": float(dist[target]),
                "loop_distance": float(loop_distances.get(target, 0.0)),
            },
        )

    def _enter_recovery(
        self,
        current,
        dist,
        prev,
        node_manager,
        frontier_points=None,
        mapped_history=(),
        exploration_complete=False,
    ):
        if not self.use_frontier_escape:
            return self._enter_nearest_node_recovery(
                dist,
                prev,
                node_manager,
            )
        return self._enter_frontier_recovery(
            current,
            dist,
            prev,
            node_manager,
            frontier_points,
            mapped_history,
            exploration_complete,
        )

    def _leave_recovery(self, reason):
        was_active = self.recovery_mode
        self.recovery_mode = False
        self.locked_escape_target = None
        self.escape_path = []
        self.path_failure_count = 0
        self._reset_trigger_counters()
        self.trigger_source = None
        self.last_exit_reason = reason
        if was_active and reason == "target_reached":
            self.target_reached_count += 1
        self.selected_escape_candidate = None
        self.selected_escape_metrics = {}
        self.escape_candidate_logs = ()
        self.escape_target_type = None
        # Re-arm only after fresh post-recovery evidence is collected.
        self.position_history.clear()
        self.coverage_gains.clear()
        return reason

    def _path_next_step_is_valid(self, current, node_manager):
        if not self.escape_path:
            return False
        current_node = node_manager.check_node_exist_in_dict(current)
        if current_node is None:
            return False
        return self.escape_path[0] in {
            _coord_key(coords) for coords in current_node.data.neighbor_set
        }

    def _replan_or_exit(self, current, node_manager):
        for _ in range(self.path_failure_limit):
            failure = self._plan_to_locked_target(current, node_manager)
            if failure is None:
                return None
            if failure == "target_unreachable":
                return self._leave_recovery(failure)
            self.path_failure_count += 1
        return self._leave_recovery("path_failure")

    def begin_step(
        self,
        current_coords,
        node_manager,
        frontier_points=None,
        exploration_complete=False,
    ):
        if not self.enabled:
            return RecoveryDecision()

        current = self._nearest_current_node(current_coords, node_manager)
        if current is None:
            return RecoveryDecision()

        exit_reason = None
        target_reached = False
        while self.escape_path and self.escape_path[0] == current:
            self.escape_path.pop(0)
        if self.recovery_mode and self._goal_reached(current):
            target_reached = True
            exit_reason = self._leave_recovery("target_reached")

        dist, prev = node_manager.Dijkstra(
            current,
            boundary=None if self.recovery_mode else self.graph_radius,
        )
        self.last_neighborhood = {
            node for node, distance in dist.items()
            if distance <= self.graph_radius
        }

        # V4 scores only p[t-H]..p[t-1]; p[t] is appended at the end.
        mapped_history = self._map_history_to_current_graph(node_manager)
        self.revisit_count = sum(
            node in self.last_neighborhood for node in mapped_history
        )
        self.revisit_score = (
            self.revisit_count / self.history_length
            if len(mapped_history) == self.history_length else 0.0
        )
        self.recent_mean_coverage_gain = (
            float(np.mean(self.coverage_gains)) if self.coverage_gains else 0.0
        )
        point_counts = sorted(
            Counter(mapped_history).values(),
            reverse=True,
        )
        self.point_repeat_max_count = (
            int(point_counts[0]) if point_counts else 0
        )
        self.point_repeat_top_k_count = int(sum(
            point_counts[:self.point_repeat_top_k]
        ))
        self.point_repeat_unique_count = len(point_counts)

        evidence_ready = (
            len(mapped_history) == self.history_length
            and len(self.coverage_gains) == self.progress_window
        )
        self.low_progress_detected = (
            evidence_ready
            and self.recent_mean_coverage_gain < self.coverage_threshold
        )
        self.regional_repeat_detected = (
            evidence_ready
            and self.revisit_count >= self.min_count
        )
        self.point_repeat_detected = (
            evidence_ready
            and self.point_repeat_enabled
            and self.point_repeat_max_count
            >= self.point_repeat_max_threshold
            and self.point_repeat_top_k_count
            >= self.point_repeat_top_k_threshold
        )
        regional_abnormal = (
            self.regional_repeat_detected
            and self.low_progress_detected
        )
        point_repeat_abnormal = (
            self.point_repeat_detected
            and self.low_progress_detected
        )
        if not self.recovery_mode:
            self.trigger_source = None
            trigger_source = self._update_trigger_counters(
                regional_abnormal,
                point_repeat_abnormal,
            )
            if trigger_source is not None:
                self.trigger_source = trigger_source
                full_dist, full_prev = node_manager.Dijkstra(current)
                entered_recovery = self._enter_recovery(
                    current,
                    full_dist,
                    full_prev,
                    node_manager,
                    frontier_points=frontier_points,
                    mapped_history=mapped_history,
                    exploration_complete=exploration_complete,
                )
                if not entered_recovery:
                    self.trigger_source = None

        if self.recovery_mode:
            if self.current_recovery_steps >= self.max_recovery_steps:
                exit_reason = self._leave_recovery("max_steps")
            elif not self._path_next_step_is_valid(current, node_manager):
                exit_reason = self._replan_or_exit(current, node_manager)
            if self.recovery_mode and self.escape_path:
                self.recovery_steps += 1
                self.current_recovery_steps += 1
                decision = self._decision(self.escape_path[0], True)
                self.position_history.append(_coord_key(current_coords))
                self.visit_counts[current] += 1
                return decision

        decision = self._decision(
            None,
            False,
            exit_reason=exit_reason,
            target_reached=target_reached,
        )
        self.position_history.append(_coord_key(current_coords))
        self.visit_counts[current] += 1
        return decision

    def end_episode(self, current_coords):
        """Close an active recovery with an explicit terminal reason."""
        if not self.recovery_mode:
            return None
        current = _coord_key(current_coords)
        if self._goal_reached(current):
            return self._leave_recovery("target_reached")
        return self._leave_recovery("episode_end")

    def _decision(
        self,
        next_waypoint,
        is_recovery_action,
        exit_reason=None,
        target_reached=False,
    ):
        target = None
        if self.locked_escape_target is not None:
            target = np.asarray(self.locked_escape_target, dtype=float)
        selected = self.selected_escape_candidate
        selected_metrics = self.selected_escape_metrics
        return RecoveryDecision(
            next_waypoint=(
                np.asarray(next_waypoint, dtype=float)
                if next_waypoint is not None else None
            ),
            is_recovery_action=is_recovery_action,
            revisit_count=self.revisit_count,
            revisit_score=self.revisit_score,
            recent_mean_coverage_gain=self.recent_mean_coverage_gain,
            revisit_trigger_counter=self.trigger_counter,
            point_repeat_max_count=self.point_repeat_max_count,
            point_repeat_top_k_count=self.point_repeat_top_k_count,
            point_repeat_unique_count=self.point_repeat_unique_count,
            point_repeat_trigger_counter=(
                self.point_repeat_trigger_counter
            ),
            regional_repeat_detected=self.regional_repeat_detected,
            point_repeat_detected=self.point_repeat_detected,
            low_progress_detected=self.low_progress_detected,
            trigger_source=self.trigger_source,
            recovery_mode=self.recovery_mode,
            escape_target=target,
            escape_path_length=len(self.escape_path),
            exit_reason=exit_reason,
            target_reached=target_reached,
            candidate_num=len(self.escape_candidate_logs),
            selected_group_id=(
                selected.group_id if selected is not None else None
            ),
            frontier_count=(
                selected.frontier_count if selected is not None else 0
            ),
            navigation_distance=(
                selected.navigation_distance
                if selected is not None
                else selected_metrics.get("navigation_distance")
            ),
            loop_distance=(
                selected.loop_distance
                if selected is not None
                else selected_metrics.get("loop_distance")
            ),
            utility=(selected.utility if selected is not None else None),
            target_type=self.escape_target_type,
            escape_candidates=self.escape_candidate_logs,
        )

    def debug_state(self):
        return {
            "history": list(self.position_history),
            "neighborhood": list(self.last_neighborhood),
            "revisit_count": self.revisit_count,
            "revisit_score": self.revisit_score,
            "recent_mean_coverage_gain": self.recent_mean_coverage_gain,
            "revisit_trigger_counter": self.trigger_counter,
            "point_repeat_max_count": self.point_repeat_max_count,
            "point_repeat_top_k_count": self.point_repeat_top_k_count,
            "point_repeat_unique_count": self.point_repeat_unique_count,
            "point_repeat_trigger_counter": (
                self.point_repeat_trigger_counter
            ),
            "regional_repeat_detected": self.regional_repeat_detected,
            "point_repeat_detected": self.point_repeat_detected,
            "low_progress_detected": self.low_progress_detected,
            "trigger_source": self.trigger_source,
            "recovery_mode": self.recovery_mode,
            "recovery_trigger_count": self.recovery_trigger_count,
            "recovery_steps": self.recovery_steps,
            "current_recovery_steps": self.current_recovery_steps,
            "target_reached_count": self.target_reached_count,
            "escape_target": self.locked_escape_target,
            "escape_path": list(self.escape_path),
            "escape_path_length": len(self.escape_path),
            "path_failure_count": self.path_failure_count,
            "exit_reason": self.last_exit_reason,
            "candidate_num": len(self.escape_candidate_logs),
            "selected_group_id": (
                self.selected_escape_candidate.group_id
                if self.selected_escape_candidate is not None else None
            ),
            "frontier_count": (
                self.selected_escape_candidate.frontier_count
                if self.selected_escape_candidate is not None else 0
            ),
            "navigation_distance": (
                self.selected_escape_candidate.navigation_distance
                if self.selected_escape_candidate is not None
                else self.selected_escape_metrics.get("navigation_distance")
            ),
            "loop_distance": (
                self.selected_escape_candidate.loop_distance
                if self.selected_escape_candidate is not None
                else self.selected_escape_metrics.get("loop_distance")
            ),
            "utility": (
                self.selected_escape_candidate.utility
                if self.selected_escape_candidate is not None else None
            ),
            "target_type": self.escape_target_type,
            "escape_candidates": list(self.escape_candidate_logs),
        }
