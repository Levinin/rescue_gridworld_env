"""Room generation using force-directed placement and iterative growth.

Rooms are scattered in a simulation grid, repelled apart via
closest-neighbour repulsion, then grown outward from minimum
footprints until they can't expand further.
"""

import math
from typing import List, Optional, Tuple

import numpy as np


def create_room_data_grid(
    height: int,
    width: int,
    min_room_size: int = 9,
    num_rooms: int = 200,
    room_padding: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[List[int]], int, int]:
    """Generate room positions and dimensions.

    Args:
        height: Target physical grid height (rows). Will be expanded if the
            simulation grid derived from it is too small.
        width: Target physical grid width (columns). Same expansion rule.
        min_room_size: Minimum room side length (must be odd).
        num_rooms: Number of rooms to generate.
        room_padding: Padding used when computing the simulation grid density.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        room_bounds: List of [r_min, r_max, c_min, c_max] per room.
        physical_rows: Final grid height (>= height).
        physical_cols: Final grid width  (>= width).
    """
    if rng is None:
        rng = np.random.default_rng()

    assert min_room_size % 2 == 1, "Min room size must be odd"

    # num_rows, num_cols, safety_margin = ensure_capacity(height, min_room_size, num_rooms, room_padding, width)

    # safety_margin = room_padding // 2
    # num_rows = height
    # num_cols = width
    min_padded_room = min_room_size + room_padding
    safety_margin = math.ceil(min_padded_room / 2.0)
    num_cols, num_rows, safety_margin = ensure_capacity(height - safety_margin, min_room_size,
                                                        num_rooms, room_padding, width - safety_margin)

    num_rows -= 2 * safety_margin
    num_cols -= 2 * safety_margin

    # ------------------------------------------------------------------
    # 1. Scatter room centres
    # ------------------------------------------------------------------
    row_points = (rng.random(num_rooms) * num_rows).tolist()
    col_points = (rng.random(num_rooms) * num_cols).tolist()
    coordinates = [[r, c] for r, c in zip(row_points, col_points)]

    # ------------------------------------------------------------------
    # 2. Asynchronous closest-neighbour repulsion
    # ------------------------------------------------------------------
    max_iterations = 30
    initial_step_size = 1.5

    for counter in range(max_iterations):
        for i, coord in enumerate(coordinates):
            current_step = initial_step_size * (1.5 - (counter / max_iterations))
            min_dist = float("inf")
            closest_vector = np.array([0.0, 0.0])

            for j, other in enumerate(coordinates):
                if i == j:
                    continue

                row_dist = coord[0] - other[0]
                col_dist = coord[1] - other[1]

                # Toroidal wrap
                if row_dist > num_rows / 2.0:
                    row_dist -= num_rows
                elif row_dist < -num_rows / 2.0:
                    row_dist += num_rows

                if col_dist > num_cols / 2.0:
                    col_dist -= num_cols
                elif col_dist < -num_cols / 2.0:
                    col_dist += num_cols

                dist = math.sqrt(row_dist ** 2 + col_dist ** 2)

                if dist < min_dist:
                    min_dist = dist
                    closest_vector = np.array([row_dist, col_dist])

            if min_dist > 0:
                direction = closest_vector / min_dist
                coordinates[i][0] = (coord[0] + direction[0] * current_step) % num_rows
                coordinates[i][1] = (coord[1] + direction[1] * current_step) % num_cols
            else:
                coordinates[i][0] = (coord[0] + 0.1) % num_rows
                coordinates[i][1] = (coord[1] + 0.1) % num_cols

    # ------------------------------------------------------------------
    # 3. Project to physical domain
    # ------------------------------------------------------------------
    physical_rows = num_rows + (2 * safety_margin)
    physical_cols = num_cols + (2 * safety_margin)

    physical_data = np.zeros((physical_rows, physical_cols))
    final_coordinates = []

    for coord in coordinates:
        r_sim = int(math.floor(coord[0])) % num_rows
        c_sim = int(math.floor(coord[1])) % num_cols

        r_phys = r_sim + safety_margin
        c_phys = c_sim + safety_margin
        final_coordinates.append((r_phys, c_phys))

    # ------------------------------------------------------------------
    # 4. Room expansion – variable-size growth
    # ------------------------------------------------------------------
    half_size = min_room_size // 2
    room_bounds: List[List[int]] = []

    for r, c in final_coordinates:
        r_min, r_max = r - half_size, r + half_size
        c_min, c_max = c - half_size, c + half_size
        room_bounds.append([r_min, r_max, c_min, c_max])

        physical_data[r_min : r_max + 1, c_min : c_max + 1] = 0.5
        physical_data[r, c] = 1.0

    gap = 3
    growing = True

    while growing:
        growing = False
        for i, bounds in enumerate(room_bounds):
            r_min, r_max, c_min, c_max = bounds

            # Try to grow UP
            if r_min - 1 - gap >= 0:
                if np.all(
                    physical_data[r_min - 1 - gap : r_min, c_min - 2 : c_max + 3] == 0
                ):
                    physical_data[r_min - 1, c_min : c_max + 1] = 0.5
                    room_bounds[i][0] -= 1
                    r_min -= 1
                    growing = True

            # Try to grow DOWN
            if r_max + 1 + gap < physical_rows:
                if np.all(
                    physical_data[
                        r_max + 1 : r_max + 2 + gap, c_min - 2 : c_max + 3
                    ]
                    == 0
                ):
                    physical_data[r_max + 1, c_min : c_max + 1] = 0.5
                    room_bounds[i][1] += 1
                    r_max += 1
                    growing = True

            # Try to grow LEFT
            if c_min - 1 - gap >= 0:
                if np.all(
                    physical_data[
                        r_min - 2 : r_max + 3, c_min - 1 - gap : c_min
                    ]
                    == 0
                ):
                    physical_data[r_min : r_max + 1, c_min - 1] = 0.5
                    room_bounds[i][2] -= 1
                    c_min -= 1
                    growing = True

            # Try to grow RIGHT
            if c_max + 1 + gap < physical_cols:
                if np.all(
                    physical_data[
                        r_min - 2 : r_max + 3, c_max + 1 : c_max + 2 + gap
                    ]
                    == 0
                ):
                    physical_data[r_min : r_max + 1, c_max + 1] = 0.5
                    room_bounds[i][3] += 1
                    c_max += 1
                    growing = True

    return room_bounds, physical_rows, physical_cols


def ensure_capacity(height: int, min_room_size: int, num_rooms: int, room_padding: int, width: int) -> tuple[
    int, int, int]:
    min_padded_room = min_room_size + room_padding
    safety_margin = math.ceil(min_padded_room / 2.0)

    # Ensure simulation grid is large enough for all rooms
    min_area = min_padded_room ** 2 * num_rooms
    aspect = height / max(width, 1)

    sim_cols = math.ceil(math.sqrt(min_area / max(aspect, 0.01)))
    sim_rows = math.ceil(min_area / sim_cols)

    # Derive simulation grid from physical dimensions
    # num_rows = max(sim_rows - 2 * safety_margin, min_padded_room)
    # num_cols = max(sim_cols - 2 * safety_margin, min_padded_room)
    num_rows = max(sim_rows, min_padded_room) + 2 * safety_margin
    num_cols = max(sim_cols, min_padded_room) + 2 * safety_margin

    return num_cols, num_rows, safety_margin

