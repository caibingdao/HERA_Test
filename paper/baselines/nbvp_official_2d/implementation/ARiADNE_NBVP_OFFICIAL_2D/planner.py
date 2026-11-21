"""Planar port of ETH ASL RrtTree and the RH-NBVP planning loop."""
import ctypes
from functools import lru_cache
import math
import os
from dataclasses import dataclass

import numpy as np


@lru_cache(maxsize=1)
def geometry_library():
    library = ctypes.CDLL(os.environ["RH_NBVP_GEOMETRY_LIB"])
    prefix = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.edge_free.argtypes = prefix + [ctypes.c_double] * 4
    library.visible_unknown.argtypes = prefix + [ctypes.c_double] * 3
    return library


@dataclass
class Node:
    position: np.ndarray
    parent: int
    distance_m: float
    gain: float


class RecedingHorizonNBVP:
    def __init__(self, config, cell_size, sensor_range, rng):
        self.config = config
        self.cell_size = cell_size
        self.radius = sensor_range / cell_size
        self.rng = rng
        self.remaining_branch = []
        self.history = []

    def plan(self, belief, position):
        grid = np.ascontiguousarray(belief, dtype=np.uint8)
        height, width = grid.shape
        library = geometry_library()
        geometry = (grid.ctypes.data, width, height)
        root = np.asarray(position, dtype=float)
        nodes = [Node(root, -1, 0.0, 0.0)]
        # The upstream cutoff check permits one insertion past cutoff.
        locations = np.empty((self.config["cutoff_nodes"] + 2, 2))
        locations[0] = root
        best = 0

        def insert(sample):
            nonlocal best
            difference = locations[:len(nodes)] - sample
            parent_index = int(np.argmin(np.einsum("ij,ij->i", difference, difference)))
            parent = nodes[parent_index]
            direction = sample - parent.position
            length = math.hypot(*direction)
            if length < 1e-8:
                return
            unit = direction / length
            edge_m = min(length * self.cell_size, self.config["extension_m"])
            target = parent.position + unit * (edge_m / self.cell_size)
            if np.all(np.floor(target + 0.5) == np.floor(parent.position + 0.5)):
                return
            overshoot = target + unit * (self.config["overshoot_m"] / self.cell_size)
            if not library.edge_free(*geometry, *parent.position, *overshoot):
                return

            visible_area = library.visible_unknown(*geometry, *target, self.radius) * self.cell_size ** 2
            distance = parent.distance_m + edge_m
            gain = parent.gain + visible_area * math.exp(-self.config["distance_discount_per_m"] * distance)
            index = len(nodes)
            locations[index] = target
            nodes.append(Node(target, parent_index, distance, gain))
            if gain > nodes[best].gain:
                best = index

        for sample in self.remaining_branch:
            insert(sample)
        attempts = 0
        high = np.array([width - 1, height - 1], dtype=float)
        while best == 0 or len(nodes) - 1 < self.config["initial_nodes"]:
            if len(nodes) - 1 > self.config["cutoff_nodes"]:
                return None, "no_gain"
            if attempts > self.config["attempts_per_node"] * len(nodes):
                if self.history:
                    previous = self.history[-1]
                    if library.edge_free(*geometry, *root, *previous):
                        self.history.pop()
                        return previous, "upstream_backtrack"
                return None, "sampling_exhausted"
            # Conditioning the uniform root-centered disk on hard rectangular
            # bounds is uniform over the rectangle because the disk spans it.
            insert(self.rng.uniform(0.0, high))
            attempts += 1

        branch = []
        while best:
            branch.append(nodes[best].position)
            best = nodes[best].parent
        branch.reverse()
        self.remaining_branch = branch[1:]
        self.history.append(root)
        return branch[0], "best_branch"
