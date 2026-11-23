import os
import json
from functools import lru_cache
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy

from sensor import sensor_work
from utils import *


@lru_cache(maxsize=None)
def load_training_map_paths(split_file, split_name='train', map_dir=None):
    """Resolve and validate one manifest's training-map paths."""
    if split_name != 'train':
        raise ValueError('policy training may only load the train split')
    manifest_path = Path(split_file).expanduser().resolve()
    with manifest_path.open('r', encoding='utf-8') as manifest_stream:
        manifest = json.load(manifest_stream)
    entries = manifest.get('train_maps')
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f'manifest has no non-empty train_maps list: {manifest_path}'
        )
    if len(entries) != len(set(entries)):
        raise ValueError(
            f'manifest train_maps contains duplicates: {manifest_path}'
        )

    project_root = Path(__file__).resolve().parent
    map_dir = (
        project_root / 'maps'
        if map_dir is None
        else Path(map_dir).expanduser().resolve()
    )
    paths = []
    for entry in entries:
        if not isinstance(entry, str) or not entry:
            raise ValueError(
                f'invalid train_maps entry in {manifest_path}: {entry!r}'
            )
        entry_path = Path(entry)
        if entry_path.is_absolute():
            resolved_path = entry_path.resolve()
        elif entry_path.parts and entry_path.parts[0] == 'maps':
            resolved_path = (project_root / entry_path).resolve()
        else:
            resolved_path = (map_dir / entry_path).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f'training map does not exist: {resolved_path}'
            )
        paths.append(str(resolved_path))
    return tuple(paths)


def configured_training_map_paths():
    if MAP_SPLIT_FILE is None:
        return None
    return load_training_map_paths(
        MAP_SPLIT_FILE, MAP_SPLIT_NAME, map_dir=os.environ.get('ARIADNE_MAP_DIR')
    )


class Env:
    exploration_progress_signal = "new_free_cell_count_v1"

    def __init__(self, episode_index, plot=False):
        self.episode_index = episode_index
        self.plot = plot
        self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter

        self.robot_location = np.array([0.0, 0.0])  # meter

        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)  # meter

        self.global_frontiers = set()

        self.sensor_range = SENSOR_RANGE  # meter
        self.travel_dist = 0  # meter
        self.explored_rate = 0
        self.new_free_cell_count = 0

        self.robot_belief = sensor_work(self.robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        self.old_belief = deepcopy(self.robot_belief)

        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        if self.plot:
            self.frame_files = []
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]

    def import_ground_truth(self, episode_index):
        configured_maps = configured_training_map_paths()
        if configured_maps is None:
            map_dir = Path(__file__).resolve().parent / 'maps'
            map_list = os.listdir(map_dir)
            map_index = episode_index % np.size(map_list)
            map_path = map_dir / map_list[map_index]
        else:
            map_index = episode_index % len(configured_maps)
            map_path = Path(configured_maps[map_index])
        # Read-only provenance for visualization/export tools. This attribute
        # does not participate in environment dynamics.
        self.map_path = str(map_path.resolve())
        ground_truth = (io.imread(str(map_path), 1) * 255).astype(int)

        ground_truth = block_reduce(ground_truth, 2, np.min)

        robot_cell = np.nonzero(ground_truth == 208)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])
        if self.plot:
            self.trajectory_x.append(self.robot_location[0])
            self.trajectory_y.append(self.robot_location[1])

    def update_robot_belief(self):
        # Snapshot before sensor_work mutates the belief array in place.
        previously_unknown = self.robot_belief == UNKNOWN
        self.robot_belief = sensor_work(self.robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)
        self.new_free_cell_count = int(np.count_nonzero(
            previously_unknown & (self.robot_belief == FREE)
        ))

    def calculate_reward(self, dist):
        reward = 0
        reward -= dist / UPDATING_MAP_SIZE * 5

        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)

        reward += delta_num / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def evaluate_exploration_rate(self):
        self.explored_rate = self.get_exploration_rate()

    def get_exploration_rate(self):
        return np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoint):
        dist = np.linalg.norm(self.robot_location - next_waypoint)
        self.update_robot_location(next_waypoint)
        self.update_robot_belief()

        self.travel_dist += dist
        self.evaluate_exploration_rate()

        reward = self.calculate_reward(dist)

        return reward

    def _plot_recovery_diagnostics(
        self,
        recovery_events,
        recovery_action_flags,
    ):
        if recovery_action_flags:
            for index, is_recovery_action in enumerate(
                recovery_action_flags
            ):
                if not is_recovery_action:
                    continue
                segment_x = (
                    np.asarray(self.trajectory_x[index:index + 2])
                    - self.belief_origin_x
                ) / self.cell_size
                segment_y = (
                    np.asarray(self.trajectory_y[index:index + 2])
                    - self.belief_origin_y
                ) / self.cell_size
                plt.plot(
                    segment_x,
                    segment_y,
                    color=RECOVERY_PATH_COLOR,
                    linewidth=RECOVERY_PATH_LINE_WIDTH,
                    zorder=3,
                )

        if not recovery_events:
            return
        trigger_cells = np.asarray(
            [
                [
                    (
                        event['robot_x'] - self.belief_origin_x
                    ) / self.cell_size,
                    (
                        event['robot_y'] - self.belief_origin_y
                    ) / self.cell_size,
                ]
                for event in recovery_events
            ],
            dtype=float,
        )
        plt.scatter(
            trigger_cells[:, 0],
            trigger_cells[:, 1],
            c=RECOVERY_TRIGGER_COLOR,
            marker=RECOVERY_TRIGGER_MARKER,
            s=RECOVERY_TRIGGER_MARKER_SIZE,
            linewidths=RECOVERY_TRIGGER_LINE_WIDTH,
            zorder=6,
        )
        label_offset_x, label_offset_y = (
            RECOVERY_TRIGGER_LABEL_OFFSET_CELLS
        )
        for event, cell in zip(recovery_events, trigger_cells):
            plt.text(
                cell[0] + label_offset_x,
                cell[1] + label_offset_y,
                f"R{event['trigger_index']}",
                color=RECOVERY_TRIGGER_COLOR,
                fontsize=RECOVERY_TRIGGER_LABEL_FONT_SIZE,
                fontweight='bold',
                zorder=7,
            )
            escape_target = event.get('escape_target')
            if escape_target is None:
                continue
            target_cell = (
                np.asarray(escape_target, dtype=float)
                - np.asarray(
                    [self.belief_origin_x, self.belief_origin_y]
                )
            ) / self.cell_size
            plt.scatter(
                target_cell[0],
                target_cell[1],
                c=RECOVERY_TARGET_COLOR,
                marker=RECOVERY_TARGET_MARKER,
                s=RECOVERY_TARGET_MARKER_SIZE,
                zorder=6,
            )
            plt.text(
                target_cell[0] + label_offset_x,
                target_cell[1] + label_offset_y,
                f"T{event['trigger_index']}",
                color=RECOVERY_TARGET_COLOR,
                fontsize=RECOVERY_TRIGGER_LABEL_FONT_SIZE,
                fontweight='bold',
                zorder=7,
            )

    def plot_env(
        self,
        step,
        recovery_events=None,
        recovery_action_flags=None,
    ):

        plt.subplot(1, 3, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                 (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
                 (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 'b', linewidth=2, zorder=1)
        self._plot_recovery_diagnostics(
            recovery_events,
            recovery_action_flags,
        )
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, self.travel_dist))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.close()
        self.frame_files.append(frame)
