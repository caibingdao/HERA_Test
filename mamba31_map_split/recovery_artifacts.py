import json
import os
from collections import Counter

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


class TriggeredRecoveryArtifactRecorder:
    """Buffer lightweight episode frames and archive only triggered episodes."""

    def __init__(
        self,
        episode_id,
        map_origin,
        cell_size,
        output_root,
        enabled=True,
        gif_duration=0.5,
    ):
        self.episode_id = int(episode_id)
        self.map_origin = np.asarray(map_origin, dtype=float).reshape(2)
        self.cell_size = float(cell_size)
        self.output_root = output_root
        self.enabled = bool(enabled)
        self.gif_duration = float(gif_duration)
        self.frames = []

    def reset(self):
        self.frames = []

    def capture(self, belief_map, robot_location, explored_rate, travel_dist):
        if not self.enabled:
            return
        self.frames.append({
            "belief": np.asarray(belief_map, dtype=np.uint8).copy(),
            "robot_location": np.asarray(
                robot_location,
                dtype=float,
            ).reshape(2).copy(),
            "explored_rate": float(explored_rate),
            "travel_dist": float(travel_dist),
        })

    def _world_to_cell(self, coords):
        coords = np.asarray(coords, dtype=float)
        return (coords - self.map_origin) / self.cell_size

    @staticmethod
    def _visible_events(recovery_events, frame_index):
        return [
            event for event in recovery_events
            if int(event["trigger_step"]) <= frame_index
        ]

    def _render_gif(
        self,
        gif_path,
        final_png_path,
        recovery_events,
        recovery_action_flags,
    ):
        positions = np.asarray(
            [frame["robot_location"] for frame in self.frames],
            dtype=float,
        )
        trajectory_cells = self._world_to_cell(positions)
        figure, axis = plt.subplots(figsize=(8, 6))
        final_image = None
        try:
            with imageio.get_writer(
                gif_path,
                mode="I",
                duration=self.gif_duration,
                loop=0,
            ) as writer:
                for frame_index, frame in enumerate(self.frames):
                    axis.clear()
                    axis.imshow(frame["belief"], cmap="gray", vmin=0, vmax=255)
                    axis.axis("off")
                    visible_trajectory = trajectory_cells[:frame_index + 1]
                    axis.plot(
                        visible_trajectory[:, 0],
                        visible_trajectory[:, 1],
                        color="blue",
                        linewidth=2,
                        zorder=2,
                    )
                    for action_index in range(min(
                        frame_index,
                        len(recovery_action_flags),
                    )):
                        if not recovery_action_flags[action_index]:
                            continue
                        segment = trajectory_cells[
                            action_index:action_index + 2
                        ]
                        if len(segment) == 2:
                            axis.plot(
                                segment[:, 0],
                                segment[:, 1],
                                color="gold",
                                linewidth=3,
                                zorder=3,
                            )

                    robot_cell = trajectory_cells[frame_index]
                    axis.scatter(
                        robot_cell[0],
                        robot_cell[1],
                        c="magenta",
                        s=45,
                        zorder=5,
                    )
                    for event in self._visible_events(
                        recovery_events,
                        frame_index,
                    ):
                        trigger_cell = self._world_to_cell(
                            event["trigger_position"]
                        )
                        axis.scatter(
                            trigger_cell[0],
                            trigger_cell[1],
                            c="red",
                            marker="x",
                            s=80,
                            linewidths=2,
                            zorder=6,
                        )
                        axis.text(
                            trigger_cell[0] + 2,
                            trigger_cell[1] - 2,
                            f"R{event['trigger_index']}",
                            color="red",
                            fontweight="bold",
                            zorder=7,
                        )
                        target = event.get("target_node")
                        if target is None:
                            target = event.get("escape_target")
                        if target is None:
                            continue
                        target_cell = self._world_to_cell(target)
                        axis.scatter(
                            target_cell[0],
                            target_cell[1],
                            c="lime",
                            marker="*",
                            s=130,
                            zorder=6,
                        )
                        axis.text(
                            target_cell[0] + 2,
                            target_cell[1] - 2,
                            f"T{event['trigger_index']}",
                            color="green",
                            fontweight="bold",
                            zorder=7,
                        )
                    axis.set_title(
                        f"Episode {self.episode_id} | frame {frame_index} | "
                        f"explored {frame['explored_rate']:.4f} | "
                        f"distance {frame['travel_dist']:.1f} m"
                    )
                    figure.tight_layout()
                    figure.canvas.draw()
                    image = np.asarray(
                        figure.canvas.buffer_rgba(),
                        dtype=np.uint8,
                    )[:, :, :3].copy()
                    writer.append_data(image)
                    final_image = image
        finally:
            plt.close(figure)
        if final_image is not None:
            imageio.imwrite(final_png_path, final_image)

    @staticmethod
    def build_summary(recovery_events, episode_metrics):
        def event_mean(field):
            values = [
                float(event[field])
                for event in recovery_events
                if event.get(field) is not None
            ]
            return float(np.mean(values)) if values else 0.0

        target_type_counts = Counter(
            event.get("target_type") or "unknown"
            for event in recovery_events
        )
        trigger_source_counts = Counter(
            event.get("trigger_source") or "unknown"
            for event in recovery_events
        )
        return {
            "episode_id": int(episode_metrics["episode_id"]),
            "trigger_count": len(recovery_events),
            "target_type_counts": dict(sorted(target_type_counts.items())),
            "trigger_source_counts": dict(sorted(
                trigger_source_counts.items()
            )),
            "exit_reason_counts": dict(sorted(Counter(
                event.get("exit_reason") or "unfinished"
                for event in recovery_events
            ).items())),
            "target_reached_count": sum(
                bool(event.get("target_reached"))
                for event in recovery_events
            ),
            "retrigger_within_20_count": sum(
                bool(event.get("retrigger_within_20_steps"))
                for event in recovery_events
            ),
            "retrigger_within_30_count": sum(
                bool(event.get("retrigger_within_30_steps"))
                for event in recovery_events
            ),
            "mean_candidate_num": event_mean("candidate_num"),
            "mean_frontier_count": event_mean("frontier_count"),
            "mean_navigation_distance": event_mean(
                "navigation_distance"
            ),
            "mean_loop_distance": event_mean("loop_distance"),
            "mean_utility": event_mean("utility"),
            "success": bool(episode_metrics["success"]),
            "explored_rate": float(episode_metrics["explored_rate"]),
            "travel_dist": float(episode_metrics["travel_dist"]),
            "episode_steps": int(episode_metrics["episode_steps"]),
            "recovery_steps": int(episode_metrics["recovery_steps"]),
        }

    def save(
        self,
        recovery_events,
        recovery_diagnostics,
        recovery_action_flags,
        episode_metrics,
    ):
        if not self.enabled or not recovery_events or not self.frames:
            return None

        episode_directory = os.path.join(
            self.output_root,
            f"episode_{self.episode_id:06d}",
        )
        os.makedirs(episode_directory, exist_ok=True)
        stem = f"episode_{self.episode_id:06d}_recovery"
        paths = {
            "directory": episode_directory,
            "gif": os.path.join(episode_directory, f"{stem}.gif"),
            "final_png": os.path.join(
                episode_directory,
                f"{stem}_final.png",
            ),
            "events": os.path.join(
                episode_directory,
                f"{stem}_events.json",
            ),
            "diagnostics": os.path.join(
                episode_directory,
                f"{stem}_steps.json",
            ),
            "summary": os.path.join(
                episode_directory,
                f"{stem}_summary.json",
            ),
        }
        with open(paths["events"], "w", encoding="utf-8") as event_file:
            json.dump(recovery_events, event_file, ensure_ascii=False, indent=2)
        with open(
            paths["diagnostics"],
            "w",
            encoding="utf-8",
        ) as diagnostics_file:
            json.dump(
                recovery_diagnostics,
                diagnostics_file,
                ensure_ascii=False,
                indent=2,
            )
        summary = self.build_summary(recovery_events, episode_metrics)
        summary["frame_count"] = len(self.frames)
        summary["recovery_action_count"] = sum(
            bool(flag) for flag in recovery_action_flags
        )
        summary["action_source_counts"] = dict(sorted(Counter(
            diagnostic.get("action_source", "unknown")
            for diagnostic in recovery_diagnostics
        ).items()))
        summary["files"] = {
            key: os.path.basename(path)
            for key, path in paths.items()
            if key != "directory"
        }
        with open(paths["summary"], "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)

        self._render_gif(
            paths["gif"],
            paths["final_png"],
            recovery_events,
            recovery_action_flags,
        )
        return paths
