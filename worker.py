import json
import os
import time

import torch

from env import Env
from agent import Agent
from utils import *
from model import PolicyNet
from temporal_pose import INITIAL_HEADING, execute_pose_transition
from local_revisit_recovery import LocalRevisitDetector
from recovery_artifacts import TriggeredRecoveryArtifactRecorder
from replay_buffer import (
    EpisodeTrajectory,
    TrajectoryStep,
    build_history_sequence,
    detach_observation,
)

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(
        self,
        meta_agent_id,
        policy_net,
        global_step,
        device='cpu',
        save_image=False,
        rollout_history_cache=None,
        policy_version=0,
        max_episode_steps=None,
    ):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.rollout_history_cache = rollout_history_cache
        self.policy_version = policy_version
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, self.device, self.save_image)

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(16):
            self.episode_buffer.append([])

        self.use_mamba = getattr(policy_net, "use_mamba", False)
        self.trajectory = (
            EpisodeTrajectory(episode_id=global_step)
            if COLLECT_TRAJECTORY or self.use_mamba
            else None
        )
        if self.rollout_history_cache is not None and not self.use_mamba:
            raise ValueError(
                "Rollout history cache requires a Mamba policy"
            )
        self.robot_heading = INITIAL_HEADING
        self.local_revisit = LocalRevisitDetector(
            enabled=USE_LOCAL_REVISIT_RECOVERY,
            history_length=LOCAL_REVISIT_HISTORY_LENGTH,
            graph_radius=LOCAL_REVISIT_GRAPH_RADIUS,
            score_threshold=LOCAL_REVISIT_SCORE_THRESHOLD,
            exit_threshold=LOCAL_REVISIT_EXIT_THRESHOLD,
            progress_window=LOCAL_REVISIT_PROGRESS_WINDOW,
            new_free_cell_threshold=LOCAL_REVISIT_NEW_FREE_CELL_THRESHOLD,
            trigger_steps=LOCAL_REVISIT_TRIGGER_STEPS,
            min_count=LOCAL_REVISIT_MIN_COUNT,
            goal_tolerance=RECOVERY_GOAL_TOLERANCE,
            max_recovery_steps=RECOVERY_MAX_STEPS,
            path_failure_limit=RECOVERY_PATH_FAILURE_LIMIT,
            use_frontier_escape=USE_FRONTIER_ESCAPE,
            escape_min_distance=ESCAPE_MIN_DISTANCE,
            escape_min_loop_distance=ESCAPE_MIN_LOOP_DISTANCE,
            escape_distance_eps=ESCAPE_DISTANCE_EPS,
            frontier_cell_size=FRONTIER_CELL_SIZE,
            point_repeat_enabled=USE_POINT_REPEAT_DETECTION,
            point_repeat_max_count=POINT_REPEAT_MAX_COUNT,
            point_repeat_top_k=POINT_REPEAT_TOP_K,
            point_repeat_top_k_count=POINT_REPEAT_TOP_K_COUNT,
            point_repeat_trigger_steps=POINT_REPEAT_TRIGGER_STEPS,
        )
        self.recovery_diagnostics = []
        self.recovery_trigger_events = []
        self.recovery_action_flags = []
        self.last_successful_recovery_end_step = None
        self.triggered_recovery_recorder = TriggeredRecoveryArtifactRecorder(
            episode_id=global_step,
            map_origin=(
                self.env.belief_origin_x,
                self.env.belief_origin_y,
            ),
            cell_size=self.env.cell_size,
            output_root=TRIGGERED_RECOVERY_ARTIFACT_PATH,
            enabled=(
                SAVE_TRIGGERED_RECOVERY_ARTIFACTS
                and USE_LOCAL_REVISIT_RECOVERY
            ),
            gif_duration=TRIGGERED_RECOVERY_GIF_DURATION,
        )

    def run_episode(self):
        # Worker owns one Env/episode. New episodes require a new Worker.
        if self.trajectory is not None and len(self.trajectory):
            raise RuntimeError("Worker cannot reuse an already collected episode")
        self.robot_heading = INITIAL_HEADING
        self.local_revisit.reset()
        self.recovery_diagnostics = []
        self.recovery_trigger_events = []
        self.recovery_action_flags = []
        self.last_successful_recovery_end_step = None
        self.triggered_recovery_recorder.reset()
        if self.rollout_history_cache is not None:
            parameter = next(self.robot.policy_net.parameters())
            self.rollout_history_cache.start_episode(
                episode_id=self.global_step,
                policy_version=self.policy_version,
                device=parameter.device,
                dtype=parameter.dtype,
                use_mamba=self.use_mamba,
                sequence_length=self.robot.policy_net.mamba_seq_len,
                graph_embedding_dim=(
                    self.robot.policy_net
                    .history_encoder
                    .graph_embedding_dim
                ),
                token_dim=self.robot.policy_net.mamba_d_model,
            )

        try:
            return self._run_episode()
        finally:
            if (
                self.rollout_history_cache is not None
                and self.rollout_history_cache.active
            ):
                self.rollout_history_cache.end_episode(
                    self.global_step
                )

    def _run_episode(self):
        done = False
        decision_times = []
        measure_decisions = os.environ.get('ARIADNE_MEASURE_DECISIONS', '0') == '1'
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()
        self.capture_recovery_artifact_frame()

        if self.save_image:
            self.robot.plot_env()
            self.plot_env_frame(0)

        for i in range(MAX_EPISODE_STEP if self.max_episode_steps is None else self.max_episode_steps):

            current_observation = observation
            if not self.use_mamba:
                self.save_observation(current_observation)

            coverage_before = self.env.get_exploration_rate()
            current_frontiers = get_frontier_in_map(self.env.belief_info)
            frontier_before = len(current_frontiers)
            robot_pose = torch.tensor(
                [
                    self.env.robot_location[0],
                    self.env.robot_location[1],
                    self.robot_heading,
                ],
                dtype=torch.float32,
            )

            if measure_decisions:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                decision_started = time.perf_counter()
            rollout_context = None
            recovery_mode_before = self.local_revisit.recovery_mode
            trigger_count_before = (
                self.local_revisit.recovery_trigger_count
            )
            trigger_location = np.asarray(
                self.env.robot_location,
                dtype=float,
            ).copy()
            recovery = self.local_revisit.begin_step(
                self.robot.location,
                self.robot.node_manager,
                frontier_points=current_frontiers,
                exploration_complete=bool(self.robot.utility.sum() == 0),
            )
            if recovery.exit_reason is not None:
                self.finalize_recovery_event(
                    end_step=i,
                    exit_reason=recovery.exit_reason,
                    coverage_at_end=coverage_before,
                    target_reached=recovery.target_reached,
                )
            self.maybe_record_recovery_trigger_event(
                recovery_mode_before=recovery_mode_before,
                trigger_count_before=trigger_count_before,
                step_index=i,
                robot_location=trigger_location,
                recovery=recovery,
                coverage_at_trigger=coverage_before,
            )
            is_recovery_action = recovery.is_recovery_action
            self.recovery_action_flags.append(
                bool(is_recovery_action)
            )
            if is_recovery_action:
                next_location = recovery.next_waypoint
                action_index = self.robot.action_index_for_waypoint(
                    observation,
                    next_location,
                )
                if self.rollout_history_cache is not None:
                    rollout_context = self.robot.get_rollout_context_cached(
                        observation,
                        history_cache=self.rollout_history_cache,
                        episode_id=self.global_step,
                        policy_version=self.policy_version,
                        decision_step=i,
                    )
            elif self.rollout_history_cache is not None:
                (
                    next_location,
                    action_index,
                    rollout_context,
                ) = self.robot.select_next_waypoint_cached(
                    observation,
                    history_cache=self.rollout_history_cache,
                    episode_id=self.global_step,
                    policy_version=self.policy_version,
                    decision_step=i,
                )
            else:
                history_batch = None
                if self.use_mamba:
                    history_batch = build_history_sequence(
                        self.trajectory,
                        sequence_length=MAMBA_SEQ_LEN,
                    )
                    if history_batch is not None:
                        history_batch = history_batch.to(self.device)

                next_location, action_index = self.robot.select_next_waypoint(
                    observation,
                    history_batch=history_batch,
                )
            if measure_decisions:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                decision_times.append((time.perf_counter()-decision_started)*1000)
            # This is the action that will actually be applied to Env.step.
            # Recovery never stores a counterfactual SAC action in replay.
            executed_action_index = action_index
            if not self.use_mamba:
                self.save_action(executed_action_index)
                self.save_recovery_action(is_recovery_action)

            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location, self.robot.location, node.data.neighbor_set)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            reward, robot_pose, post_pose = execute_pose_transition(
                self.env, next_location, self.robot_heading
            )
            next_heading = float(post_pose[2])


            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
            if not self.use_mamba:
                self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            if not self.use_mamba:
                self.save_next_observations(observation)
            new_free_cell_count = self.env.new_free_cell_count
            self.local_revisit.record_new_free_cell_count(new_free_cell_count)
            frontier_after = len(get_frontier_in_map(self.env.belief_info))
            delta_frontier = frontier_after - frontier_before

            if self.trajectory is not None:
                self.trajectory.append(
                    TrajectoryStep(
                        observation=detach_observation(
                            current_observation
                        ),
                        action=int(executed_action_index.item()),
                        reward=float(reward),
                        next_observation=detach_observation(observation),
                        robot_pose=robot_pose,
                        post_pose=post_pose,
                        new_free_cell_count=int(new_free_cell_count),
                        delta_frontier=int(delta_frontier),
                        done=bool(done),
                        step_index=i,
                        is_recovery_action=is_recovery_action,
                    )
                )
            if self.rollout_history_cache is not None:
                (
                    graph_embedding,
                    action_feature,
                    history_token,
                ) = self.robot.policy_net.finalize_rollout_history_token(
                    rollout_context=rollout_context,
                    action_index=int(executed_action_index.item()),
                    robot_pose=robot_pose,
                    post_pose=post_pose,
                    reward=reward,
                    new_free_cell_count=new_free_cell_count,
                    delta_frontier=delta_frontier,
                )
                self.rollout_history_cache.append(
                    episode_id=self.global_step,
                    policy_version=self.policy_version,
                    transition_id=i,
                    graph_embedding=graph_embedding,
                    action_feature=action_feature,
                    token=history_token,
                )
            self.robot_heading = float(next_heading)

            debug_state = self.local_revisit.debug_state()
            recovery_log = {
                "step": i,
                "new_free_cell_count": int(new_free_cell_count),
                "revisit_count": debug_state["revisit_count"],
                "revisit_score": debug_state["revisit_score"],
                "recent_mean_new_free_cell_count": debug_state[
                    "recent_mean_new_free_cell_count"
                ],
                "revisit_trigger_counter": debug_state[
                    "revisit_trigger_counter"
                ],
                "point_repeat_max_count": debug_state[
                    "point_repeat_max_count"
                ],
                "point_repeat_top_k_count": debug_state[
                    "point_repeat_top_k_count"
                ],
                "point_repeat_unique_count": debug_state[
                    "point_repeat_unique_count"
                ],
                "point_repeat_trigger_counter": debug_state[
                    "point_repeat_trigger_counter"
                ],
                "regional_repeat_detected": debug_state[
                    "regional_repeat_detected"
                ],
                "point_repeat_detected": debug_state[
                    "point_repeat_detected"
                ],
                "low_progress_detected": debug_state[
                    "low_progress_detected"
                ],
                "trigger_source": debug_state["trigger_source"],
                "recovery_mode": debug_state["recovery_mode"],
                "recovery_trigger_count": debug_state[
                    "recovery_trigger_count"
                ],
                "recovery_steps": debug_state["recovery_steps"],
                "escape_target": debug_state["escape_target"],
                "escape_path_length": debug_state["escape_path_length"],
                "target_type": debug_state["target_type"],
                "is_recovery_action": is_recovery_action,
                "action_source": (
                    "recovery" if is_recovery_action else "sac"
                ),
                "exit_reason": recovery.exit_reason,
            }
            self.recovery_diagnostics.append(recovery_log)
            self.capture_recovery_artifact_frame()
            if LOCAL_REVISIT_DEBUG_VISUALIZATION:
                print(
                    "local revisit: "
                    f"step={i} "
                    f"score={recovery.revisit_score:.3f} "
                    "new_free_cell_count="
                    f"{recovery.recent_mean_new_free_cell_count:.6f} "
                    f"counter={recovery.revisit_trigger_counter} "
                    "point_repeat="
                    f"{recovery.point_repeat_max_count}/"
                    f"{recovery.point_repeat_top_k_count} "
                    "point_counter="
                    f"{recovery.point_repeat_trigger_counter} "
                    f"source={recovery.trigger_source} "
                    f"recovery={recovery.recovery_mode} "
                    f"target={recovery.escape_target} "
                    f"path_length={recovery.escape_path_length}"
                )

            if self.save_image:
                self.robot.plot_env(
                    self.local_revisit.debug_state()
                    if LOCAL_REVISIT_DEBUG_VISUALIZATION else None
                )
                self.plot_env_frame(i + 1)

            if done:
                break

        terminal_exit_reason = self.local_revisit.end_episode(
            self.env.robot_location
        )
        if terminal_exit_reason is not None:
            self.finalize_recovery_event(
                end_step=max(0, len(self.recovery_diagnostics) - 1),
                exit_reason=terminal_exit_reason,
                coverage_at_end=self.env.explored_rate,
                target_reached=(terminal_exit_reason == "target_reached"),
            )

        # save metrics
        self.perf_metrics['decision_total_ms'] = sum(decision_times)
        self.perf_metrics['decision_count'] = len(decision_times)
        self.perf_metrics['map_path'] = str(getattr(self.env, 'map_path', ''))
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        episode_steps = len(self.recovery_diagnostics)
        self.perf_metrics['total_recovery_triggers'] = (
            self.local_revisit.recovery_trigger_count
        )
        self.perf_metrics['total_recovery_steps'] = (
            self.local_revisit.recovery_steps
        )
        self.perf_metrics['recovery_step_ratio'] = (
            self.local_revisit.recovery_steps / episode_steps
            if episode_steps else 0.0
        )
        recovery_triggers = self.local_revisit.recovery_trigger_count
        target_reached = self.local_revisit.target_reached_count
        self.perf_metrics['recovery_target_reached'] = target_reached
        self.perf_metrics['recovery_target_reach_rate'] = (
            target_reached / recovery_triggers if recovery_triggers else 0.0
        )
        self.perf_metrics['recovery_mean_steps_per_trigger'] = (
            self.local_revisit.recovery_steps / recovery_triggers
            if recovery_triggers else 0.0
        )
        self.perf_metrics['recovery_retrigger_count'] = sum(
            event.get("steps_since_last_recovery_end") is not None
            for event in self.recovery_trigger_events
        )
        target_types = [
            event.get("target_type")
            for event in self.recovery_trigger_events
        ]
        self.perf_metrics['recovery_frontier_target_count'] = (
            target_types.count("frontier")
        )
        self.perf_metrics['recovery_filter_fallback_count'] = (
            target_types.count("frontier_filter_fallback")
        )
        self.perf_metrics['recovery_graph_fallback_count'] = (
            target_types.count("graph_fallback")
        )
        self.perf_metrics['recovery_nearest_node_count'] = (
            target_types.count("nearest_node")
        )
        trigger_sources = [
            event.get("trigger_source")
            for event in self.recovery_trigger_events
        ]
        self.perf_metrics['recovery_regional_trigger_count'] = (
            trigger_sources.count("regional")
        )
        self.perf_metrics['recovery_point_repeat_trigger_count'] = (
            trigger_sources.count("point_repeat")
        )
        self.perf_metrics['recovery_combined_trigger_count'] = (
            trigger_sources.count("regional+point_repeat")
        )
        self.perf_metrics['recovery_retrigger_within_20_count'] = sum(
            bool(event.get("retrigger_within_20_steps"))
            for event in self.recovery_trigger_events
        )
        self.perf_metrics['recovery_retrigger_within_30_count'] = sum(
            bool(event.get("retrigger_within_30_steps"))
            for event in self.recovery_trigger_events
        )
        self.perf_metrics['recovery_mean_candidate_num'] = self._event_mean(
            "candidate_num"
        )
        self.perf_metrics['recovery_mean_frontier_count'] = self._event_mean(
            "frontier_count"
        )
        self.perf_metrics['recovery_mean_navigation_distance'] = (
            self._event_mean("navigation_distance")
        )
        self.perf_metrics['recovery_mean_loop_distance'] = self._event_mean(
            "loop_distance"
        )
        self.perf_metrics['revisit_score'] = self.local_revisit.revisit_score
        self.perf_metrics['recent_mean_new_free_cell_count'] = (
            self.local_revisit.recent_mean_new_free_cell_count
        )
        self.perf_metrics['revisit_trigger_counter'] = (
            self.local_revisit.trigger_counter
        )
        self.perf_metrics['point_repeat_trigger_counter'] = (
            self.local_revisit.point_repeat_trigger_counter
        )
        self.perf_metrics['recovery_mode'] = self.local_revisit.recovery_mode
        self.perf_metrics['recovery_diagnostics'] = self.recovery_diagnostics
        self.perf_metrics['recovery_trigger_events'] = (
            self.recovery_trigger_events
        )

        if SAVE_RECOVERY_EVENT_LOG and self.recovery_trigger_events:
            self.save_recovery_event_log()

        artifact_paths = self.save_triggered_recovery_artifacts(
            done=done,
            episode_steps=episode_steps,
        )
        self.perf_metrics['triggered_recovery_artifacts'] = artifact_paths

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def _event_mean(self, field):
        values = [
            float(event[field])
            for event in self.recovery_trigger_events
            if event.get(field) is not None
        ]
        return float(np.mean(values)) if values else 0.0

    def capture_recovery_artifact_frame(self):
        self.triggered_recovery_recorder.capture(
            belief_map=self.env.robot_belief,
            robot_location=self.env.robot_location,
            explored_rate=self.env.get_exploration_rate(),
            travel_dist=self.env.travel_dist,
        )

    def save_triggered_recovery_artifacts(self, done, episode_steps):
        return self.triggered_recovery_recorder.save(
            recovery_events=self.recovery_trigger_events,
            recovery_diagnostics=self.recovery_diagnostics,
            recovery_action_flags=self.recovery_action_flags,
            episode_metrics={
                "episode_id": self.global_step,
                "success": done,
                "explored_rate": self.env.explored_rate,
                "travel_dist": self.env.travel_dist,
                "episode_steps": episode_steps,
                "recovery_steps": self.local_revisit.recovery_steps,
            },
        )

    def record_recovery_trigger_event(
        self,
        step_index,
        robot_location,
        recovery,
        coverage_at_trigger,
    ):
        location = np.asarray(robot_location, dtype=float).reshape(2)
        escape_target = recovery.escape_target
        event = {
            "episode_id": int(self.global_step),
            "trigger_index": len(self.recovery_trigger_events) + 1,
            "step": int(step_index),
            "step_index": int(step_index),
            "x": float(location[0]),
            "y": float(location[1]),
            "robot_x": float(location[0]),
            "robot_y": float(location[1]),
            "trigger_step": int(step_index),
            "trigger_position": location.tolist(),
            "revisit_count": int(recovery.revisit_count),
            "revisit_score": float(recovery.revisit_score),
            "trigger_source": getattr(recovery, "trigger_source", None),
            "regional_repeat_detected": bool(getattr(
                recovery,
                "regional_repeat_detected",
                False,
            )),
            "point_repeat_detected": bool(getattr(
                recovery,
                "point_repeat_detected",
                False,
            )),
            "low_progress_detected": bool(getattr(
                recovery,
                "low_progress_detected",
                False,
            )),
            "point_repeat_max_count": int(getattr(
                recovery,
                "point_repeat_max_count",
                0,
            )),
            "point_repeat_top_k_count": int(getattr(
                recovery,
                "point_repeat_top_k_count",
                0,
            )),
            "point_repeat_unique_count": int(getattr(
                recovery,
                "point_repeat_unique_count",
                0,
            )),
            "revisit_trigger_counter": int(getattr(
                recovery,
                "revisit_trigger_counter",
                0,
            )),
            "point_repeat_trigger_counter": int(getattr(
                recovery,
                "point_repeat_trigger_counter",
                0,
            )),
            "recent_mean_new_free_cell_count": float(
                recovery.recent_mean_new_free_cell_count
            ),
            "mean_new_free_cell_count": float(recovery.recent_mean_new_free_cell_count),
            "recovery_start_step": int(step_index),
            "escape_target": (
                np.asarray(escape_target, dtype=float).tolist()
                if escape_target is not None
                else None
            ),
            "candidate_num": int(getattr(recovery, "candidate_num", 0)),
            "selected_group_id": getattr(
                recovery,
                "selected_group_id",
                None,
            ),
            "frontier_count": int(getattr(recovery, "frontier_count", 0)),
            "navigation_distance": getattr(
                recovery,
                "navigation_distance",
                None,
            ),
            "loop_distance": getattr(recovery, "loop_distance", None),
            "utility": getattr(recovery, "utility", None),
            "target_node": (
                np.asarray(escape_target, dtype=float).tolist()
                if escape_target is not None
                else None
            ),
            "target_type": getattr(recovery, "target_type", None),
            "escape_target_type": getattr(recovery, "target_type", None),
            "candidates": list(getattr(recovery, "escape_candidates", ())),
            "recovery_end_step": None,
            "recovery_steps": None,
            "exit_reason": None,
            "target_reached": False,
            "coverage_at_trigger": float(coverage_at_trigger),
            "coverage_at_recovery_end": None,
            "post_recovery_coverage_gain": None,
            "steps_since_last_recovery_end": (
                int(step_index - self.last_successful_recovery_end_step)
                if self.last_successful_recovery_end_step is not None
                else None
            ),
            "retrigger_within_20_steps": False,
            "retrigger_within_30_steps": False,
        }
        steps_since_end = event["steps_since_last_recovery_end"]
        if steps_since_end is not None:
            event["retrigger_within_20_steps"] = steps_since_end <= 20
            event["retrigger_within_30_steps"] = steps_since_end <= 30
            for previous in reversed(self.recovery_trigger_events):
                if previous.get("target_reached"):
                    previous["retrigger_within_20_steps"] = steps_since_end <= 20
                    previous["retrigger_within_30_steps"] = steps_since_end <= 30
                    break
        self.recovery_trigger_events.append(event)
        return event

    def finalize_recovery_event(
        self,
        end_step,
        exit_reason,
        coverage_at_end,
        target_reached,
    ):
        for event in reversed(self.recovery_trigger_events):
            if event.get("exit_reason") is not None:
                continue
            event["recovery_end_step"] = int(end_step)
            event["recovery_steps"] = int(
                self.local_revisit.current_recovery_steps
            )
            event["exit_reason"] = str(exit_reason)
            event["target_reached"] = bool(target_reached)
            event["coverage_at_recovery_end"] = float(coverage_at_end)
            event["post_recovery_coverage_gain"] = float(
                coverage_at_end - event["coverage_at_trigger"]
            )
            if target_reached:
                self.last_successful_recovery_end_step = int(end_step)
            return event
        return None

    def maybe_record_recovery_trigger_event(
        self,
        recovery_mode_before,
        trigger_count_before,
        step_index,
        robot_location,
        recovery,
        coverage_at_trigger,
    ):
        entered_recovery = (
            not recovery_mode_before
            and self.local_revisit.recovery_trigger_count
            > trigger_count_before
        )
        if not entered_recovery:
            return None
        return self.record_recovery_trigger_event(
            step_index=step_index,
            robot_location=robot_location,
            recovery=recovery,
            coverage_at_trigger=coverage_at_trigger,
        )

    def plot_env_frame(self, step):
        if ENABLE_RECOVERY_VIS:
            self.env.plot_env(
                step,
                recovery_events=self.recovery_trigger_events,
                recovery_action_flags=self.recovery_action_flags,
            )
        else:
            # Preserve the exact baseline 3.1 visualization call path.
            self.env.plot_env(step)

    def save_recovery_event_log(self, log_path=None):
        log_path = log_path or RECOVERY_EVENT_LOG_PATH
        os.makedirs(log_path, exist_ok=True)
        filename = os.path.join(
            log_path,
            f"episode_{self.global_step:06d}_recovery.json",
        )
        with open(filename, 'w', encoding='utf-8') as event_file:
            json.dump(
                self.recovery_trigger_events,
                event_file,
                ensure_ascii=False,
                indent=2,
            )
        return filename

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_recovery_action(self, is_recovery_action):
        self.episode_buffer[15] += torch.tensor(
            [bool(is_recovery_action)],
            dtype=torch.bool,
            device=self.device,
        ).reshape(1, 1, 1)

    def save_next_observations(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()


if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    # checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['policy_model'])
    worker = Worker(0, model, 77, save_image=True)
    worker.run_episode()
