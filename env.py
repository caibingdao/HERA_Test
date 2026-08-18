import os
from pathlib import Path

import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy

from sensor import sensor_work
from utils import *


class Env:
    def __init__(
        self,
        episode_index,
        plot=False,
        map_list=None,
        map_path=None,
        start_seed=None,
    ):
        self.episode_index = episode_index
        self.plot = plot
        self.map_list = None if map_list is None else list(map_list)
        self.explicit_map_path = map_path
        self.start_seed = start_seed
        self.start_resample_count = 0
        self.map_path = None
        self.map_id = None
        self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index)
        self.start_cell = self.robot_cell.copy()
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter

        self.robot_location = np.array([0.0, 0.0])  # meter
        self.robot_trajectory = [self.robot_location.copy()]

        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)  # meter

        self.global_frontiers = set()

        self.sensor_range = SENSOR_RANGE  # meter
        self.travel_dist = 0  # meter
        self.explored_rate = 0

        self.robot_belief = sensor_work(self.robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        self.old_belief = deepcopy(self.robot_belief)

        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        if self.plot:
            self.frame_files = []
            self.trajectory_x = [self.robot_location[0]]
            self.trajectory_y = [self.robot_location[1]]

    def _select_map_path(self, episode_index):
        if self.explicit_map_path is not None and self.map_list is not None:
            raise ValueError("provide either map_path or map_list, not both")
        if self.explicit_map_path is not None:
            return Path(self.explicit_map_path)
        if self.map_list is not None:
            if not self.map_list:
                raise ValueError("map_list must contain at least one map")
            map_index = int(episode_index) % len(self.map_list)
            return Path(self.map_list[map_index])

        # Preserve the original command's behavior when no split is supplied.
        map_dir = Path("maps")
        map_list = os.listdir(map_dir)
        if not map_list:
            raise FileNotFoundError(f"no maps found in {map_dir}")
        map_index = int(episode_index) % np.size(map_list)
        return map_dir / map_list[map_index]

    def _choose_start_cell(self, marker_cells, ground_truth):
        if marker_cells.shape[0] == 0:
            raise ValueError(
                f"map contains no valid 208 start marker: {self.map_path}"
            )

        # The legacy environment always selected marker index 10. Optional
        # experiment seeds select a reproducible pixel from the same marker
        # region without changing the default training behavior.
        if self.start_seed is None:
            candidate_order = list(range(marker_cells.shape[0]))
            first_index = 10
            if marker_cells.shape[0] <= first_index:
                raise ValueError(
                    "map has fewer than 11 start-marker cells required by "
                    f"the legacy rule: {self.map_path}"
                )
            candidate_order.remove(first_index)
            candidate_order.insert(0, first_index)
        else:
            rng = np.random.default_rng(int(self.start_seed))
            candidate_order = rng.permutation(marker_cells.shape[0]).tolist()

        for attempt, candidate_index in enumerate(candidate_order):
            row, column = marker_cells[candidate_index]
            if (
                0 <= row < ground_truth.shape[0]
                and 0 <= column < ground_truth.shape[1]
                and ground_truth[row, column] == FREE
            ):
                self.start_resample_count = attempt
                return np.array([column, row], dtype=int)

        raise ValueError(
            f"all start-marker cells are invalid or occupied: {self.map_path}"
        )

    def import_ground_truth(self, episode_index):
        selected_path = self._select_map_path(episode_index).expanduser()
        if not selected_path.is_file():
            raise FileNotFoundError(f"map file does not exist: {selected_path}")
        self.map_path = str(selected_path.resolve())
        self.map_id = selected_path.name

        ground_truth = (
            io.imread(str(selected_path), as_gray=True) * 255
        ).astype(int)

        ground_truth = block_reduce(ground_truth, 2, np.min)

        marker_cells = np.argwhere(ground_truth == 208)

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1
        robot_cell = self._choose_start_cell(marker_cells, ground_truth)

        return ground_truth, robot_cell

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_trajectory.append(np.asarray(robot_location).copy())
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])
        if self.plot:
            self.trajectory_x.append(self.robot_location[0])
            self.trajectory_y.append(self.robot_location[1])

    def update_robot_belief(self):
        self.robot_belief = sensor_work(self.robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)

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

    def plot_env(self, step):

        plt.subplot(1, 3, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                 (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
                 (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 'b', linewidth=2, zorder=1)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, self.travel_dist))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.close()
        self.frame_files.append(frame)
