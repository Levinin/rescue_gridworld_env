import math
import os
import random
import traceback
from copy import deepcopy
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from rescue_gridworld.create_rooms import create_room_data_grid, ensure_capacity
from rescue_gridworld.env_constants import *


@dataclass
class Person:
    pos: Tuple[int, int]
    following: bool = False
    rescued: bool = False


@dataclass
class Cupboard:
    pos: Tuple[int, int]
    locked: bool = True
    has_keycard: bool = True
    chain_id: Optional[int] = None  # which door chain it belongs to


@dataclass
class DoorInfo:
    pos: Tuple[int, int]
    locked: bool
    room_id: int = 0
    chain_id: Optional[int] = None  # None => generic/unbound


class RescueGridworldEnv(gym.Env):
    """
    A procedural gridworld with rooms, doors, cupboards, keys, keycards, people, and an exit.
    The agent must collect keys -> unlock cupboards -> get keycards -> unlock doors -> talk to people (so they follow) -> bring all people to the exit to finish.
    When initialising the environment, at least 1 key and keycard must be specified to create chains for key->cupboard->keycard->door. If keys and keycards are set to 0, the doors can still be locked but no keycards or keys will be placed in the environment, instead it is expected the obtain_all_chain_keys_keycards method is used to obtain the keys and keycards.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        width: int = 12,
        height: int = 12,
        num_rooms: int = 3,
        num_people: int = 1,
        num_keys: int = 3,
        num_keycards: int = 3,
        door_lock_prob: float = 0.5,
        cabinet_lock_prob: float = 0.5,
        max_steps: int = 1000,
        tile_size: int = 32,
        seed: Optional[int] = None,
        debug_draw_chains: bool = False,
        reset_options: Dict[str, Any] = {},
        stochastic_transition_chance: float = 0.0,
        obs_window_size: int = 7,  # Must be odd
        has_fire: bool = False,
        fire_speed: int = 60,
    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode
        self.num_rooms = max(3 if has_fire else 2, num_rooms)
        self.num_keycards = num_keycards
        self.num_people = num_people
        self.num_keys = num_keys
        self.door_lock_prob = door_lock_prob
        self.cabinet_lock_prob = cabinet_lock_prob
        self.max_steps = max_steps
        self.tile_size = tile_size

        self._rng = np.random.default_rng(seed)
        self._seed = seed

        self.reset_options = reset_options

        self.stochastic_transition_chance = stochastic_transition_chance
        self.has_fire = has_fire
        self.fire_speed = fire_speed

        self.action_space = spaces.Discrete(10)
        assert obs_window_size % 2 == 1, "The window size must be an odd number."
        self.obs_window_size = obs_window_size
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, obs_window_size, obs_window_size),
                    dtype=np.uint8,
                ),
                "chain_grid": spaces.Box(
                    low=-2,
                    high=np.inf,
                    shape=(1, obs_window_size, obs_window_size),
                    dtype=np.int64,
                ),
            }
        )

        # dynamic state
        self.grid: np.ndarray
        self.chain_id_grid: np.ndarray
        self.door_temp_checked: Dict[Tuple[int, int], bool] = {}  # Doors temp checked and if hot.
        self.rooms: List[
            Tuple[int, int, int, int]
        ] = []  # list of room rects (y1, x1, y2, x2)
        self.doors: Dict[Tuple[int, int], DoorInfo] = {}  # pos -> DoorInfo
        self.room_graph: Dict[int, List[int]] = {}        # adjacency on room indices
        self.cupboards: Dict[Tuple[int, int], Cupboard] = {}
        self.people: List[Person] = []
        self.people_following: int = 0  # How many people have been saved
        self.people_died: int = 0  # How many people have died in the fire
        self.previous_traversable_code: List[int] = []
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.exit_pos: Tuple[int, int] = (0, 0)
        self.start_pos: Tuple[int, int] = (0, 0)
        self.fire_pos: Tuple[int, int] = (0, 0)
        self.fire_counter: int = 0
        self.inventory = {
            "keys": 0,  # generic keys (for unbound cupboards only)
            "keycards": 0,  # generic keycards (for unbound doors only)
            "key_ids": set(),  # chain_ids of keys held
            "keycard_ids": set(),  # chain_ids of keycards held
        }
        self._chain_at_pos: Dict[
            Tuple[int, int], Tuple[str, int]
        ] = {}  # (y,x) -> ("key", chain_id)
        self._step_count = 0
        self.solvable_plan: List[dict] = []
        self._person_move_step_count = 0
        self._person_move_step = lambda x: (x + 1) % 4  # Loop 3 steps

        # rendering
        self._screen = None
        self._clock = None
        self._surface = None

        self._episode_rewards = 0

        num_cols, num_rows, safety_margin = ensure_capacity(
            height, 9, num_rooms, 6, width
        )
        self.height = num_rows
        self.width = num_cols

        _, _, _ = create_room_data_grid(num_rows, num_cols, 9, 50, 6)

        # Fog of war
        self.discovered_grid = np.zeros((self.height, self.width), dtype=bool)

        # Plan render
        self.debug_draw_chains = debug_draw_chains
        self.chain_plan: Dict[
            int, Dict[str, Tuple[int, int]]
        ] = {}  # chain_id -> {"key":(y,x), "cupboard":(y,x), "door":(y,x)}
        self._font_small = None  # lazy-init in render
        self.passable_tiles = [EMPTY, KEY_TILE, PERSON_TILE, EXIT, FIRE, DOOR_OPEN]

        self._passable_tiles_set = {EMPTY, KEY_TILE, PERSON_TILE, EXIT, FIRE, DOOR_OPEN}
        self._los_paths = self._precompute_los_paths(self.obs_window_size)

        self._current_direction = NORTH

        self.door_types = [DOOR_LOCKED, DOOR_UNLOCKED, DOOR_LOCKED_HOT, DOOR_UNLOCKED_HOT, DOOR_LOCKED_COLD, DOOR_UNLOCKED_COLD]
        self.door_locked_types = [DOOR_LOCKED, DOOR_LOCKED_HOT, DOOR_LOCKED_COLD]
        self.door_unlocked_types = [DOOR_UNLOCKED, DOOR_UNLOCKED_HOT, DOOR_UNLOCKED_COLD]

        # To render orientation
        s = tile_size // 3
        self.dir_offsets = {
            0: [(0, -s), (-s, s), (s, s)],    # North
            1: [(s, 0), (-s, -s), (-s, s)],   # East
            2: [(0, s), (s, -s), (-s, -s)],   # South
            3: [(-s, 0), (s, -s), (s, s)]     # West
        }

        self.opened_door_original_type: Dict[Tuple[int, int], int] = {}

    # --------------- Gym API ---------------
    def reset(self, *, seed: Optional[int] = None, options: Dict[str, Any] | None = {}):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        self._generate_level()
        if self.reset_options.get("get_all_chains", False):
            self.obtain_all_chain_keys_keycards()

        obs = self._get_obs()
        self.opened_door_original_type.clear()
        self._step_count = 0
        self.people_following = 0
        self.people_died = 0
        self._episode_rewards = 0
        self._fire_counter = 0
        self._current_direction = NORTH
        info = self._update_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"ERROR: {action} is invalid."
        reward: float = -0.05

        self._fire_counter = (self._fire_counter + 1) % self.fire_speed
        if self._fire_counter == 0:
            self._expand_fire()

        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        info["action_success"] = False
        info["action_code"] = int(action)
        info["Truncate_Reason"] = "N/A"

        # Sometimes a random action will occur.
        if random.random() < self.stochastic_transition_chance:
            action = random.randint(0, 8)

        # Did the fire catch us?
        if self.grid[self.agent_pos] == FIRE:
            reward -= 10.0
            info["Truncate_Reason"] = "Caught by fire"
            truncated = True

        if not (terminated or truncated):
            # If we took a temperature reading on the previous action then revert the door to its original state
            # since the environment should not remember sensor readings.
            self._remove_temp_sensor_reading()
            r, terminated, truncated = self._perform_action(action, info)
            reward += r

        # Did we step into fire?
        if self.grid[self.agent_pos] == FIRE:
            reward -= 10.0
            info["Truncate_Reason"] = "Stepped into fire"
            truncated = True

        self._move_people()

        info.update(self._update_info())

        self._step_count += 1
        if self._step_count >= self.max_steps:
            info["Truncate_Reason"] = "Episode length exceeded"
            truncated = True


        self._episode_rewards += reward

        return self._get_obs(), reward, terminated, truncated, info

    def _remove_temp_sensor_reading(self) -> None:
        """Remove the just taken temp sensor reading from the grid since the env
        should not remember sensor readings, that is the agent's job."""
        drow, dcol = forward_offset.get(self._current_direction, (0, 0))
        row, col = self.agent_pos[0] + drow, self.agent_pos[1] + dcol
        if self.grid[row, col] in [DOOR_LOCKED_COLD, DOOR_LOCKED_HOT]:
            self.grid[row, col] = DOOR_LOCKED
        elif self.grid[row, col] in [DOOR_UNLOCKED_HOT, DOOR_UNLOCKED_COLD]:
            self.grid[row, col] = DOOR_UNLOCKED

    def _perform_action(self, action: int, info: dict[str, Any]) -> Tuple[float, bool, bool]:
        """Perform the action in the environment and return an update to the rewarad."""

        terminated, truncated = False, False
        reward = 0

        if action in (FORWARD, ROT_LEFT, ROT_RIGHT):
            r, terminated, truncated = self._action_navigate(info, action)
            reward += r    # Do this to avoid accidents if we change rewards earlier in the method.

        elif action == PICK_KEY:
            r, terminated, truncated = self._action_pick_key(info)
            reward += r

        elif action == UNLOCK_CUPBOARD:
            r, terminated, truncated = self._action_unlock_cupboard(info)
            reward += r

        elif action == PICK_KEYCARD:
            r, terminated, truncated = self._action_pick_keycard(info)
            reward += r

        elif action == CHECK_DOOR_TEMP:
            # Checks the temp of the door ahead of the agent
            # No reward for checking the door temp
            self._action_check_door_temp(info)

        elif action == UNLOCK_DOOR:
            r, terminated, truncated = self._action_unlock_door(info)
            reward += r

        elif action == OPEN_DOOR:
            # No reward for opening the door.
            self._action_open_door(info)

        elif action == TALK_PERSON:
            r, terminated, truncated = self._action_talk_people(info)
            reward += r

        return reward, terminated, truncated

    def _action_navigate(self, info: Dict[str, Any], action: int) -> Tuple[float, bool, bool]:
        reward = 0
        row, col = self.agent_pos
        drow, dcol = 0, 0
        terminated, truncated = False, False

        if action == FORWARD:
            drow, dcol = forward_offset.get(self._current_direction, (0, 0))
        elif action == ROT_LEFT:
            self._current_direction = (self._current_direction - 1) % 4
        else:
            self._current_direction = (self._current_direction + 1) % 4

        nrow, ncol = row + drow, col + dcol
        if self.grid[nrow, ncol] in self.passable_tiles:
            self.agent_pos = (nrow, ncol)
            # If we turn away from the door, or move through it, close the door to the previous type
            # Happens after movement so we don't close it on ourselves as we move through the door.
            open_door_loc = self._adjacent_of_type({DOOR_OPEN})
            if open_door_loc:
                self.grid[open_door_loc] = self.opened_door_original_type.pop(open_door_loc)
            info["action_success"] = True

        if self.grid[self.agent_pos] == EXIT:
            terminated = True
            reward += 5.0           # Add base reward for exiting.
            if not self.people:     # Everyone has been rescued.
                reward += 50.0
                info["success"] = True

        return reward, terminated, truncated

    def _action_pick_key(self, info: Dict[str, Any]) -> Tuple[float, bool, bool]:
        if self.grid[self.agent_pos] == KEY_TILE:
            self.grid[self.agent_pos] = EMPTY
            self.chain_id_grid[self.agent_pos] = -1
            info["action_success"] = True
            entry = self._chain_at_pos.pop((self.agent_pos), None)
            if entry is not None:
                kind, cid = entry
                if kind == "key":
                    self.inventory["key_ids"].add(cid)
                return 5., False, False
        return 0, False, False

    def _action_unlock_cupboard(self, info:Dict[str, Any]) -> Tuple[float, bool, bool]:
        cpos = self._ahead_of_type_with_chain_id({CUPBOARD_LOCKED})
        if cpos is not None:
            cup = self.cupboards.get(cpos)
            if cup and cup.locked:
                if cup.chain_id in self.inventory["key_ids"]:
                    self.inventory["key_ids"].remove(cup.chain_id)
                    self._unlock_cupboard(cpos)
                    info["action_success"] = True
                    return 5, False, False
        return 0, False, False

    def _action_pick_keycard(self, info: Dict[str, Any]) -> Tuple[float, bool, bool]:
        cpos = self._forward_cupboard_with_keycard_unlocked()
        if cpos is not None:
            cup = self.cupboards[cpos]
            cup.has_keycard = False
            if cup.chain_id is not None:
                self.inventory["keycard_ids"].add(cup.chain_id)
            else:
                self.inventory["keycards"] += 1
            self.grid[cpos] = CUPBOARD_UNLOCKED
            info["action_success"] = True
            return 5., False, False
        return 0, False, False

    def _action_check_door_temp(self, info: Dict[str, Any]) -> None:
        offset = forward_offset[self._current_direction]
        ar, ac = self.agent_pos
        pos = (ar + offset[0], ac + offset[1])
        dpos = pos if self.grid[pos] in self.door_types else None
        if dpos is not None:
            door_hot: bool = self._check_if_door_is_hot(dpos)
            if door_hot and self.grid[dpos] in self.door_locked_types:
                self.grid[dpos]  = DOOR_LOCKED_HOT
            elif door_hot and self.grid[dpos] in self.door_unlocked_types:
                self.grid[dpos] = DOOR_UNLOCKED_HOT
            elif not door_hot and self.grid[dpos] in self.door_locked_types:
                self.grid[dpos] = DOOR_LOCKED_COLD
            elif not door_hot and self.grid[dpos] in self.door_unlocked_types:
                self.grid[dpos] = DOOR_UNLOCKED_COLD
            self.door_temp_checked[dpos] = door_hot
            info["action_success"] = True    # No reward here

    def _action_unlock_door(self, info: Dict[str, Any]) -> Tuple[float, bool, bool]:
        dpos = self._ahead_of_type_with_chain_id({DOOR_LOCKED_COLD, DOOR_LOCKED_HOT, DOOR_LOCKED})
        reward = 0
        if dpos is not None:
            dinfo = self.doors.get(dpos)
            if dinfo and dinfo.locked:
                if dinfo.chain_id in self.inventory["keycard_ids"]:
                    self.inventory["keycard_ids"].remove(dinfo.chain_id)
                    self._unlock_door(dpos)
                    # Only give a reward for unlocking a door if we have checked the temp first.
                    if self.door_temp_checked.get(dpos, 9) != 9:
                        reward = 5
                        if self.door_temp_checked.get(dpos):
                            self.grid[dpos] = DOOR_UNLOCKED_HOT
                        else:
                            self.grid[dpos] = DOOR_UNLOCKED_COLD
                    info["action_success"] = True
        return reward, False, False

    def _action_open_door(self, info: Dict[str, Any]) -> None:
        # The door must be ahead of us and unlocked.
        dir_o = forward_offset[self._current_direction]
        r, c = self.agent_pos[0] + dir_o[0], self.agent_pos[1] + dir_o[1]
        if self.grid[r, c] in self.door_unlocked_types:
            self.opened_door_original_type[(r,c)] = self.grid[r, c]
            self.grid[r, c] = DOOR_OPEN
            info["action_success"] = True
            # Penalty for opening the door if there is fire here:
            if self._check_if_door_is_hot((r,c)):
                [self._expand_fire() for _ in range(3)]

    def _action_talk_people(self, info: Dict[str, Any]) -> Tuple[float, bool, bool]:
        num_people = len(self.people)
        for idx in range(num_people):
            if self.people[idx].pos == self.agent_pos:
                self.grid[self.people[idx].pos] = self.previous_traversable_code[idx]
                self.previous_traversable_code.pop(idx)
                self.people.pop(idx)
                self.people_following += 1
                reward = 5
                info["action_success"] = True
                return reward, False, False
        return 0, False, False

    def _check_if_door_is_hot(self, dpos: Tuple[int, int]) -> bool:
        """Checks the door temp by detecting fire 7x8 squares forward centred on the door directly through the door."""
        # First determine the direction of the door relative to the agent
        direction = (dpos[0] - self.agent_pos[0], dpos[1] - self.agent_pos[1])
        if direction[0] == 0:
            row_start = self.agent_pos[0] - 3
            row_end = self.agent_pos[0] + 3
            col_start = self.agent_pos[1] + direction[1]
            col_end = self.agent_pos[1] + direction[1] * 8
        else:
            row_start = self.agent_pos[0] + direction [0]
            row_end = self.agent_pos[0] + direction[0] * 8
            col_start = self.agent_pos[1] - 3
            col_end = self.agent_pos[1] + 3
        if FIRE in self.grid[min(row_start, row_end):max(row_start, row_end),
            min(col_start, col_end):max(col_start, col_end)]:
            return True
        return False

    def render(self):
        if self.render_mode is None:
            return

        tile = self.tile_size
        w_px, h_px = int(self.width * tile), int(self.height * tile)
        if self._surface is None:
            # Select a headless driver ONLY for non-human rendering if none configured
            if (
                self.render_mode != "human"
                and os.environ.get("SDL_VIDEODRIVER") is None
            ):
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            self._clock = pygame.time.Clock()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((w_px, h_px))
            self._surface = pygame.Surface((w_px, h_px))

        if self._font_small is None:
            try:
                self._font_small = pygame.font.SysFont(
                    None, max(12, self.tile_size // 2)
                )
            except Exception:
                self._font_small = pygame.font.Font(None, max(12, self.tile_size // 2))

        surf = self._surface
        surf.fill((230, 230, 230))

        # draw grid
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * tile, y * tile, tile, tile)
                code = self.grid[y, x]
                color = (240, 240, 240)
                if code == WALL:
                    color = (60, 60, 60)
                elif code == DOOR_LOCKED:
                    color = (200, 60, 60)
                elif code == DOOR_LOCKED_HOT:
                    color = (200, 60, 60)
                elif code == DOOR_LOCKED_COLD:
                    color = (200, 60, 60)
                elif code == DOOR_UNLOCKED:
                    color = (60, 200, 60)
                elif code == DOOR_UNLOCKED_COLD:
                    color = (60, 200, 60)
                elif code == DOOR_UNLOCKED_HOT:
                    color = (60, 200, 60)
                elif code == DOOR_OPEN:
                    color = (100, 255, 100)
                elif code == CUPBOARD_LOCKED:
                    color = (139, 90, 43)
                elif code == CUPBOARD_UNLOCKED:
                    color = (184, 134, 11)
                elif code == CUPBOARD_UNLOCKED_KEYCARD:
                    color = (224, 184, 51)
                elif code == KEY_TILE:
                    color = (240, 240, 240)  # floor; we'll draw key icon below
                elif code == EXIT:
                    color = (168, 85, 247)
                elif code == FIRE:
                    color = (252, 186, 3)
                pygame.draw.rect(surf, color, rect)
                pygame.draw.rect(surf, (210, 210, 210), rect, width=1)

                if code == KEY_TILE:
                    pygame.draw.circle(surf, (255, 215, 0), rect.center, tile // 5)

        for pos, c in self.cupboards.items():
            if c.has_keycard:
                y, x = pos
                cx, cy = x * tile + tile // 2, y * tile + tile // 2
                pygame.draw.rect(
                    surf,
                    (0, 0, 0),
                    (
                        x * tile + tile // 4,
                        y * tile + tile // 4,
                        tile - tile // 2,
                        tile - tile // 2,
                    ),
                    width=1,
                )  # (cx - 3, cy - 8, 6, 16), width=1)

        for p in self.people:
            if p.following:
                continue
            y, x = p.pos
            rect = pygame.Rect(x * tile, y * tile, tile, tile)
            pygame.draw.circle(surf, (66, 135, 245), rect.center, tile // 3)  # person

        # ---- DEBUG OVERLAY: chain IDs & paths ----
        if self.debug_draw_chains and self.chain_plan:
            tile = self.tile_size
            surf = self._surface

            # A) Label items with chain_id
            for cid, items in self.chain_plan.items():
                color = self._color_for_chain(cid)

                # Key label (if key still exists at that pos OR always draw in debug)
                ky, kx = items["key"]
                kr = pygame.Rect(kx * tile, ky * tile, tile, tile)
                text = self._font_small.render(f"K{cid}", True, color)
                surf.blit(text, (kr.x + 2, kr.y + 2))

                # Cupboard label
                cy, cx = items["cupboard"]
                cr = pygame.Rect(cx * tile, cy * tile, tile, tile)
                text = self._font_small.render(f"C{cid}", True, color)
                surf.blit(text, (cr.x + 2, cr.y + 2))

                # Door label
                dy, dx = items["door"]
                dr = pygame.Rect(dx * tile, dy * tile, tile, tile)
                text = self._font_small.render(f"D{cid}", True, color)
                surf.blit(text, (dr.x + 2, dr.y + 2))

            # B) Draw planned path: key -> cupboard -> door -> exit (per chain)
            #    We draw small rectangles along cells in the path using the chain colour.
            for cid, items in self.chain_plan.items():
                color = self._color_for_chain(cid)
                points: List[Tuple[int, int]] = []

                p1 = self._bfs_path_cells(items["key"], items["cupboard"])
                p2 = self._bfs_path_cells(items["cupboard"], items["door"])
                p3 = self._bfs_path_cells(items["door"], self.exit_pos)

                for seg in (p1, p2, p3):
                    if seg:
                        # Avoid duplicating last point between segments
                        if points and seg and points[-1] == seg[0]:
                            points.extend(seg[1:])
                        else:
                            points.extend(seg)

                # Draw the polyline as tiny filled rects on each cell
                for yy, xx in points:
                    rect = pygame.Rect(
                        xx * tile + tile // 4,
                        yy * tile + tile // 4,
                        max(2, tile // 2),
                        max(2, tile // 2),
                    )
                    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    s.fill((*color, 120))  # translucent
                    surf.blit(s, (rect.x, rect.y))

        if self.debug_draw_chains:
            # Label Start (S) and Exit (E)
            sy, sx = self.agent_pos
            sr = pygame.Rect(sx * tile, sy * tile, tile, tile)
            surf.blit(
                self._font_small.render("S", True, (0, 0, 0)),
                (sr.x + tile - 12, sr.y + 2),
            )

            ey, ex = self.exit_pos
            er = pygame.Rect(ex * tile, ey * tile, tile, tile)
            surf.blit(
                self._font_small.render("E", True, (0, 0, 0)),
                (er.x + tile - 12, er.y + 2),
            )

        # ---- Fog of War ----
        fog_surf = pygame.Surface((tile, tile), pygame.SRCALPHA)
        fog_surf.fill((100, 100, 100, 180))  # grey, semi-transparent
        for y in range(self.height):
            for x in range(self.width):
                if not self.discovered_grid[y, x]:
                    surf.blit(fog_surf, (x * tile, y * tile))

        ay, ax = self.agent_pos
        # Calculate absolute center (World Space)
        cx = ax * tile + tile // 2
        cy = ay * tile + tile // 2

        # Fetch the relative points for our current direction
        local_points = self.dir_offsets[self._current_direction]

        # Apply the translation offset to create the final render points
        points = [(cx + dx, cy + dy) for dx, dy in local_points]

        # Render
        pygame.draw.polygon(surf, (250, 250, 250), points)
        pygame.draw.polygon(surf, (0, 0, 0), points, width=2)

        if self.render_mode == "human":
            # Handle events so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self._surface = None
                    self._screen = None
                    self._clock = None
                    return
            # blit and flip
            assert self._screen is not None
            self._screen.blit(surf, (0, 0))
            pygame.display.set_caption(
                f"RescueGridworld | Step:{self._step_count} | Episode Reward:{round(self._episode_rewards, 1)}"
            )
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(surf)  # (w,h,3)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self._surface is not None:
            pygame.quit()
        self._surface = None
        self._screen = None
        self._clock = None

    def obtain_all_chain_keys_keycards(self) -> None:
        """Method to allow obtaining all chain keys and keycards for training purposes."""
        # Cycle through doors and add the chain keys and keycards to the inventory.
        for _, door_info in list(self.doors.items()):
            if door_info.chain_id is not None:
                self.inventory["key_ids"].add(door_info.chain_id)
                self.inventory["keycard_ids"].add(door_info.chain_id)

        # Cycle through cupboards and add the chain keys and keycards to the inventory.
        for _, cup_info in list(self.cupboards.items()):
            if cup_info.chain_id is not None:
                self.inventory["key_ids"].add(cup_info.chain_id)
                self.inventory["keycard_ids"].add(cup_info.chain_id)

    def _get_random_adjacent_traversable_square(self, p: Person) -> Tuple[int, int]:
        r, c = p.pos

        # Define the 4 possible adjacent coordinates
        candidates = ((r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c))

        # Filter valid positions using a comprehension.
        valid_moves = [pos for pos in candidates if self.grid[pos] == EMPTY]

        return random.choice(valid_moves) if valid_moves else p.pos

    def _expand_fire(self) -> None:
        """Cycle through the grid and expand fire in adjacent non-wall squares."""
        m = deepcopy(self.grid)
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == FIRE:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (self.grid[nr, nc] != WALL):
                            m[nr, nc] = FIRE
        self.grid = m

    # PRIVATE METHODS
    def _move_people(self) -> None:
        self._person_move_step_count = self._person_move_step(
            self._person_move_step_count
        )
        if self._person_move_step_count != 0:  # Only move once per cycle.
            return

        people = len(self.people)
        for i in range(people - 1, -1, -1):
            if self.grid[self.people[i].pos] == FIRE:
                self.people.pop(i)   # Fire caught them, so let's remove them.
                self.previous_traversable_code.pop(i)
                self.people_died += 1
                continue
            empty_adjacent: Tuple[int, int] = self._get_random_adjacent_traversable_square(self.people[i])
            self.grid[self.people[i].pos] = self.previous_traversable_code[i]
            self.previous_traversable_code[i] = self.grid[empty_adjacent]
            self.people[i].pos = empty_adjacent
            self.grid[self.people[i].pos] = PERSON_TILE

    def _update_info(self) -> Dict[str, Any]:
        """Add the dynamic data to the info dict."""
        attrs = [
            "grid",
            "rooms",
            "doors",
            "room_graph",
            "cupboards",
            "people",
            "people_following",
            "people_died",
            "agent_pos",
            "door_temp_checked",
            "exit_pos",
            "inventory",
            "_chain_at_pos",
            "_step_count",
            "solvable_plan",
            "chain_plan",
            "discovered_grid",
        ]

        # Build the dict dynamically
        info = {name: getattr(self, name) for name in attrs}
        return info

    def _room_boundary_cells(self, rect):
        y1, x1, y2, x2 = rect
        pairs = []
        for x in range(x1, x2 + 1):
            if y1 - 1 >= 0:
                pairs.append(((y1 - 1, x), (y1, x)))
            if y2 + 1 < self.height:
                pairs.append(((y2 + 1, x), (y2, x)))
        for y in range(y1, y2 + 1):
            if x1 - 1 >= 0:
                pairs.append(((y, x1 - 1), (y, x1)))
            if x2 + 1 < self.width:
                pairs.append(((y, x2 + 1), (y, x2)))
        return pairs

    def _find_or_create_door_into_room(self, room_idx) -> Tuple[int, int] | None:
        rect = self.rooms[room_idx]
        # Prefer existing corridor contact
        for outside, inside in self._room_boundary_cells(rect):
            oy, ox = outside
            iy, ix = inside
            if not self._is_within(oy, ox) or not self._is_within(iy, ix):
                continue
            if self.grid[iy, ix] == EMPTY and self.grid[oy, ox] == EMPTY:
                self.grid[oy, ox] = DOOR_UNLOCKED
                self.doors[(oy, ox)] = DoorInfo(
                    (oy, ox), locked=False, chain_id=None, room_id=room_idx
                )
                return (oy, ox)
        # Otherwise carve a one-tile stub outside and place a door
        for outside, inside in self._room_boundary_cells(rect):
            oy, ox = outside
            iy, ix = inside
            if not self._is_within(oy, ox) or not self._is_within(iy, ix):
                continue
            if self.grid[iy, ix] == EMPTY and self.grid[oy, ox] == WALL:
                self.grid[oy, ox] = DOOR_UNLOCKED
                self.doors[(oy, ox)] = DoorInfo(
                    (oy, ox), locked=False, chain_id=None, room_id=room_idx
                )
                return (oy, ox)
        return None

    # --------------- Level generation (with solvable chain) ---------------
    def _generate_level(self):

        MAX_ATTEMPTS = 1000
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # if self._try_generate_one():
            #     return
            try:
                if self._try_generate_one():
                    return
            except Exception as e:
                last_error = e
                # print(e, traceback.format_exc())
        raise RuntimeError(
            f"Level generation failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}. {traceback.format_exc()}. Try increasing grid size or lowering num_rooms."
        )

    def _reset_class_variables(self, H, W):
        """To reset class state ahead of creating a new level"""
        self.grid = np.full((H, W), WALL, dtype=np.int8)
        self.chain_id_grid = np.full((H, W), -1, dtype=np.int16)
        self.rooms.clear()
        self.doors.clear()
        self.room_graph = {}
        self.cupboards.clear()
        self.people.clear()
        self.previous_traversable_code.clear()
        self.inventory = {
            "keys": 0,
            "keycards": 0,
            "key_ids": set(),
            "keycard_ids": set(),
        }
        self._chain_at_pos.clear()
        self._step_count = 0
        self.solvable_plan = []
        self.chain_plan = {}
        self.discovered_grid = np.zeros((H, W), dtype=bool)

    def _try_generate_one(self):
        H, W = self.height, self.width
        self._reset_class_variables(H, W)

        if not self._place_rooms(H, W):
            return False

        self._connect_rooms()
        start_idx = self._place_start_exit_fire()
        self._place_items(start_idx)
        self._place_people()

        if len(self.doors) == 0:
            return False
        return True

    def _place_rooms(self, H: int, W: int) -> bool:
        """Place rooms using force-directed layout from create_rooms module."""
        room_bounds, phys_h, phys_w = create_room_data_grid(
            height=H,
            width=W,
            min_room_size=9,
            num_rooms=self.num_rooms,
            room_padding=6,
            rng=self._rng,
        )

        # If create_room_data_grid expanded the grid, resize our arrays
        if phys_h != H or phys_w != W:
            self.height = phys_h
            self.width = phys_w
            self.grid = np.full((phys_h, phys_w), WALL, dtype=np.int8)
            self.chain_id_grid = np.full((phys_h, phys_w), -1, dtype=np.int16)
            self.discovered_grid = np.zeros((phys_h, phys_w), dtype=bool)

        # Convert room_bounds [r_min, r_max, c_min, c_max] to
        # self.rooms format (y1, x1, y2, x2)
        for r_min, r_max, c_min, c_max in room_bounds:
            self.grid[r_min : r_max + 1, c_min : c_max + 1] = EMPTY
            self.rooms.append((r_min, c_min, r_max, c_max))

        return len(self.rooms) >= 1

    def _connect_rooms(self):
        centers = [((r[0] + r[2]) // 2, (r[1] + r[3]) // 2) for r in self.rooms]

        # --- Build MST over room centers ---
        nodes = list(range(len(centers)))
        edges = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                a, b = centers[i], centers[j]
                d = abs(a[0] - b[0]) + abs(a[1] - b[1])
                edges.append((d, i, j))
        edges.sort()
        parent = list(range(len(nodes)))

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru
                return True
            return False

        tree_edges = []
        for d, i, j in edges:
            if union(i, j):
                tree_edges.append((i, j))

        # --- Carve corridors and place boundary door tiles (initially unlocked) ---
        room_incoming_doors = {i: [] for i in range(len(self.rooms))}
        corridor_elbows = {}

        height, width = self.grid.shape

        for i, j in tree_edges:
            (row0, col0), (row1, col1) = centers[i], centers[j]

            room_i = self.rooms[i]
            room_j = self.rooms[j]

            x_step = 1 if col1 >= col0 else -1
            y_step = 1 if row1 >= row0 else -1

            # --- compute elbow and push it one tile away from adjacent room if needed ---
            elbow_row, elbow_col = row0, col1

            # only care when the path is actually L-shaped
            if col0 != col1 and row0 != row1:
                # Check for corridor collisions with other rooms.
                jr_step = 1 if row0 > row1 else -1
                jc_step = 1 if col0 > col1 else -1
                ir_step = 1 if row1 > row0 else -1
                ic_step = 1 if col1 > col0 else -1
                for count in range(20):
                    collision_j = self._collision_check((elbow_row, elbow_col, row1, col1), i, j)
                    collision_i = self._collision_check((elbow_row, elbow_col, row0, col0), i, j)

                    if not collision_j and not collision_i and count > 0:  # Do 1 full pass
                        break

                    # Move the colliding corridor sideways
                    if collision_j:
                        if elbow_row == row1:
                            # If j's line is horizontal and i is further down than j, move j's line down, otherwise up.
                            elbow_row = row1 = np.clip(row1 + jr_step, room_j[0]+1, room_j[2]-1)

                        elif elbow_col == col1:
                            # If j's line is vertical and i is further right than j, move j's line right, otherwise left.
                            elbow_col = col1 = np.clip(col1 + jc_step, room_j[1]+1, room_j[3]-1)

                    if collision_i:
                        if elbow_row == row0:
                            # If i's line is horizontal and j is further down than i, move i's line down, otherwise up.
                            elbow_row = row0 = np.clip(row0 + ir_step, room_i[0]+1, room_i[2]-1)

                        elif elbow_col == col0:
                            # If i's line is vertical and j is further right than i, move i's line right, otherwise left.
                            elbow_col = col0 = np.clip(col0 + ic_step, room_i[1]+1, room_i[3]-1)

                    # Move a bit further if we are now up against the wall of the room
                    if elbow_row == room_j[0] - 1:
                        elbow_row = row0 = room_j[0] + 1
                    elif elbow_row == room_j[2] + 1:
                        elbow_row = row0 = room_j[2] - 1
                    if elbow_col == room_j[1] - 1:
                        elbow_col = col0 = room_j[1] + 1
                    elif elbow_col == room_j[3] + 1:
                        elbow_col = col0 = room_j[3] - 1
                    if elbow_row == room_i[0] - 1:
                        elbow_row = row1 = room_i[0] + 1
                    elif elbow_row == room_i[2] + 1:
                        elbow_row = row1 = room_i[2] - 1
                    if elbow_col == room_i[1] - 1:
                        elbow_col = col1 = room_i[1] + 1
                    elif elbow_col == room_i[3] + 1:
                        elbow_col = col1 = room_i[3] - 1

                else:
                    print(f"Failed to create corridor between {i} and {j}")
                    exit(1)

                # Adjust the corridor dimensions.
                centers[i] = (row0, col0)
                centers[j] = (row1, col1)

            elbow = (elbow_row, elbow_col)
            corridor_elbows[(i, j)] = elbow
            corridor_elbows[(j, i)] = elbow
            (row0, col0), (row1, col1) = centers[i], centers[j]
            # --- carve corridor exactly as before (from center to center) ---
            for x in range(col0, col1 + x_step, x_step):
                self.grid[row0, x] = EMPTY
            for y in range(row0, row1 + y_step, y_step):
                self.grid[y, col1] = EMPTY

            # --- doors use the *adjusted* elbow ---
            door_i = self._boundary_door_position(centers[i], elbow)
            if door_i and self.grid[door_i] == EMPTY:
                self.grid[door_i] = DOOR_UNLOCKED
                self.doors[door_i] = DoorInfo(
                    door_i, locked=False, chain_id=None, room_id=i
                )
                room_incoming_doors[i].append(door_i)

            door_j = self._boundary_door_position(centers[j], elbow)
            if door_j and self.grid[door_j] == EMPTY:
                self.grid[door_j] = DOOR_UNLOCKED
                self.doors[door_j] = DoorInfo(
                    door_j, locked=False, chain_id=None, room_id=j
                )
                room_incoming_doors[j].append(door_j)

        # Build room graph (adjacency) from MST
        self.room_graph = {i: [] for i in range(len(self.rooms))}
        for a, b in tree_edges:
            self.room_graph[a].append(b)
            self.room_graph[b].append(a)

    def _collision_check(self, line: Tuple, start_room_idx: int, end_room_idx: int) -> bool:
        """Check if the corridor segment intersects with any other room's boundary."""
        lr0, lc0, lr1, lc1 = line
        for i in range(len(self.rooms)):
            if i == start_room_idx or i == end_room_idx:
                continue
            r0, c0, r1, c1 = self.rooms[i]
            r0e = r0 - 3
            c0e = c0 - 3
            r1e = r1 + 3
            c1e = c1 + 3

            for line_row in range(lr0 if lr0 <= lr1 else lr1, (lr1 if lr0 <= lr1 else lr0) + 1):
                for line_col in range(lc0 if lc0 <= lc1 else lc1, (lc1 if lc0 <= lc1 else lc0) + 1):
                    # if (r0 <= line_row <= r1 and c0 <= line_col <= c1):
                    if (r0e <= line_row <= r1e and c0 <= line_col <= c1) or (r0 <= line_row <= r1 and c0e <= line_col <= c1e):
                        return True
        return False

    def _get_location_in_wall(self, room_idx: int) -> tuple[int, int] | None:
        """Returns a random location in the wall for a room or None otherwise."""

        ey1, ex1, ey2, ex2 = self.rooms[room_idx]
        # Step down the left wall.
        for i in range(1, ey2 - ey1, 1):
            if (
                self.grid[ey1 + i, ex1 - 1] == WALL
                and self.grid[ey1 + i + 1, ex1 - 1] == WALL
                and self.grid[ey1 + i - 1, ex1 - 1] == WALL
                and self.grid[ey1 + i, ex1] == EMPTY
                and self.grid[ey1 + i + 1, ex1] == EMPTY
                and self.grid[ey1 + i - 1, ex1] == EMPTY
                and self.grid[ey1 + i, ex1 - 2] == WALL
                and self.grid[ey1 + i + 1, ex1 - 2] == WALL
                and self.grid[ey1 + i - 1, ex1 - 2] == WALL
            ):
                return (ey1 + i, ex1 - 1)

        # Step down the right wall.
        for i in range(1, ey2 - ey1, 1):
            if (
                self.grid[ey1 + i, ex2 + 1] == WALL
                and self.grid[ey1 + i + 1, ex2 + 1] == WALL
                and self.grid[ey1 + i - 1, ex2 + 1] == WALL
                and self.grid[ey1 + i, ex2] == EMPTY
                and self.grid[ey1 + i + 1, ex2] == EMPTY
                and self.grid[ey1 + i - 1, ex2] == EMPTY
                and self.grid[ey1 + i, ex2 + 2] == WALL
                and self.grid[ey1 + i + 1, ex2 + 2] == WALL
                and self.grid[ey1 + i - 1, ex2 + 2] == WALL
            ):
                return (ey1 + i, ex2 + 1)

        return None

    def _place_start_exit_fire(self) -> int:
        """Place the start and exit positions in the grid and choose the start location of the fire."""
        # --- Choose start/exit rooms ---
        sp = None
        start_idx = 0
        while sp is None:
            start_idx = int(self._rng.integers(0, len(self.rooms)))
            sp = self._get_location_in_wall(start_idx)

        self.start_pos = sp
        self.grid[self.start_pos] = EXIT

        # Place the exit in a wall
        exit_options = list(range(len(self.rooms)))
        exit_options.remove(start_idx)
        for _ in range(100):
            exit_idx = random.choice(exit_options)
            assert start_idx != exit_idx, (
                "Fatal Error: The start and finish rooms should not be the same."
            )
            ep = self._get_location_in_wall(exit_idx)
            if ep is None:
                continue
            self.exit_pos = ep
            self.grid[self.exit_pos] = EXIT
            break
        else:
            assert False, (
                "Fatal Error: Unable to place exit while creating environment."
            )

        if self.has_fire:
            # Find the start location of the fire.
            # We don't want the fire to start in the same room as the start or exit.
            fire_options = list(range(len(self.rooms)))
            fire_options.remove(start_idx)
            fire_options.remove(exit_idx)
            fire_idx = random.choice(fire_options)
            fire_pos = self._random_empty_in_rect(self.rooms[fire_idx])
            self.fire_pos = fire_pos
            self.grid[self.fire_pos] = FIRE

        # Place agent in start room next to start location
        if self.grid[self.start_pos[0], self.start_pos[1] + 1] == EMPTY:
            self.agent_pos = (self.start_pos[0], self.start_pos[1] + 1)
        elif self.grid[self.start_pos[0], self.start_pos[1] - 1] == EMPTY:
            self.agent_pos = (self.start_pos[0], self.start_pos[1] - 1)
        else:
            assert False, "Fatal error, unable to place agent in start room."
        self.agent_start = self.agent_pos
        return start_idx

    def _place_items(self, start_room_idx: int):
        # -------------------------
        # Item placement planning with strict chains
        # -------------------------
        self._lock_doors_and_assign_chains()
        placed_keys, placed_keycards = self._place_chain_items(start_room_idx)
        placed_keys, placed_keycards = self._place_extra_keycards(
            start_room_idx, placed_keys, placed_keycards
        )
        self._place_extra_keys(placed_keys)

    def _lock_doors_and_assign_chains(self):
        # 1) Randomly lock doors by lock_prob; keep others unlocked and assign chain_ids to locked ones
        next_chain_id = 0
        for dpos, dinfo in list(self.doors.items()):
            if self._rng.random() < self.door_lock_prob:
                dinfo.locked = True
                dinfo.chain_id = next_chain_id
                self.grid[dpos] = DOOR_LOCKED
                self.chain_id_grid[dpos] = dinfo.chain_id
                next_chain_id += 1
            else:
                dinfo.locked = False
                dinfo.chain_id = None
                self.grid[dpos] = DOOR_UNLOCKED
                self.chain_id_grid[dpos] = -1

    def _place_chain_items(self, start_room_idx: int) -> Tuple[int, int]:
        placed_keys = 0
        placed_keycards = 0

        # For each locked door, create its cupboard/key with the same chain_id
        for dpos, dinfo in list(self.doors.items()):
            if not dinfo.locked or dinfo.chain_id is None:
                continue

            child_room = dinfo.room_id
            # child_room = self._get_door_child_room(dpos)
            if child_room is None:
                # if door can't be resolved to a room, unlock to avoid dead-ends
                dinfo.locked = False
                dinfo.chain_id = None
                self.grid[dpos] = DOOR_UNLOCKED
                continue

            rpath = self._get_room_path(start_room_idx, child_room)
            if not rpath:
                dinfo.locked = False
                dinfo.chain_id = None
                self.grid[dpos] = DOOR_UNLOCKED
                continue

            # We may have locked doors but not want to place keys or keycards for training.
            # In this case, we don't place the keys or keycards.
            if self.num_keys > 0 and self.num_keycards > 0:
                # Cupboard room: between start and door (exclude child_room)
                cupboard_room_candidates = (
                    rpath[:-1] if len(rpath) >= 1 else [start_room_idx]
                )
                if not cupboard_room_candidates:
                    cupboard_room_candidates = [start_room_idx]
                # cupboard_room = cupboard_room_candidates[-1]

                # Place cupboard (locked/unlocked, with keycard) with chain_id
                cup_pos, cupboard_room = self._place_cupboard_in_room(
                    cupboard_room_candidates
                )
                self.chain_id_grid[cup_pos] = dinfo.chain_id
                if random.random() < self.cabinet_lock_prob:
                    self.grid[cup_pos] = CUPBOARD_LOCKED
                    self.cupboards[cup_pos] = Cupboard(
                        pos=cup_pos,
                        locked=True,
                        has_keycard=True,
                        chain_id=dinfo.chain_id,
                    )
                else:
                    self.grid[cup_pos] = CUPBOARD_UNLOCKED_KEYCARD
                    self.cupboards[cup_pos] = Cupboard(
                        pos=cup_pos,
                        locked=False,
                        has_keycard=True,
                        chain_id=dinfo.chain_id,
                    )
                placed_keycards += 1

                # Key room: between start and cupboard room
                key_room_path = self._get_room_path(start_room_idx, cupboard_room)
                assert len(key_room_path) > 0, "No key room path found"
                # key_room = random.choice(key_room_path)   #  key_room_path[-random.randint(1,2)]

                key_pos, key_room = self._place_key_in_room(key_room_path)
                self.grid[key_pos] = KEY_TILE
                self.chain_id_grid[key_pos] = dinfo.chain_id
                self._chain_at_pos[key_pos] = ("key", dinfo.chain_id)
                placed_keys += 1

                self.solvable_plan.append(
                    {
                        "chain_id": dinfo.chain_id,
                        "door_pos": dpos,
                        "door_room": child_room,
                        "cupboard_room": cupboard_room,
                        "cupboard_pos": cup_pos,
                        "key_room": key_room,
                        "key_pos": key_pos,
                    }
                )

                cid = int(dinfo.chain_id)
                self.chain_plan[cid] = {
                    "key": key_pos,
                    "cupboard": cup_pos,
                    "keycard": cup_pos,
                    "door": dpos,
                }
        return placed_keys, placed_keycards

    def _place_extra_keycards(
        self, start_room_idx: int, placed_keys: int, placed_keycards: int
    ) -> Tuple[int, int]:
        # Add extra locked/unlocked cabinets with keycards to requested number of keycards
        while placed_keycards < self.num_keycards:
            # cupboard_room = random.randint(0, len(self.rooms) - 1) # Random room index
            pos, cupboard_room = self._place_cupboard_in_room(
                list(range(len(self.rooms) - 1))
            )
            # Set chain id to None since there is no locked door for this keycard
            if random.random() < self.cabinet_lock_prob:
                self.cupboards[pos] = Cupboard(
                    pos=pos,
                    locked=True,
                    has_keycard=True,
                    chain_id=100 + placed_keycards,
                )  # Keep chain id high
                self.grid[pos] = CUPBOARD_LOCKED
                self.chain_id_grid[pos] = 100 + placed_keycards
                # If the cupboard is locked, place a key for it.
                # for _ in range(100):  # Retry loop for key placement
                # Key room: between start and cupboard room
                if self.num_keys > 0:
                    key_room_path = self._get_room_path(start_room_idx, cupboard_room)
                    # key_room = random.choice(key_room_path)   #key_room_path[-random.randint(1, 2)]

                    key_pos, key_room = self._place_key_in_room(key_room_path)
                    self.grid[key_pos] = KEY_TILE
                    self.chain_id_grid[key_pos] = 100 + placed_keycards
                    self._chain_at_pos[key_pos] = ("key", 100 + placed_keycards)
                    placed_keys += 1
            else:
                self.cupboards[pos] = Cupboard(
                    pos=pos, locked=False, has_keycard=True, chain_id=None
                )
                self.grid[pos] = CUPBOARD_UNLOCKED_KEYCARD
                self.chain_id_grid[pos] = -1
            placed_keycards += 1
        return placed_keys, placed_keycards

    def _place_extra_keys(self, placed_keys: int):
        # Add extra random generic keys to reach requested num_keys
        while placed_keys < self.num_keys:
            for _ in range(100):  # Retry loop for key placement
                pos = self._random_empty_global()

                if self.grid[pos] == EMPTY and self._adjacent_squares_empty(pos):
                    self.grid[pos] = KEY_TILE
                    # generic key (no chain tag)
                    placed_keys += 1
                    break
            else:
                # If we can't find a suitable position after 100 attempts, stop adding keys
                break

    def _get_room_path(self, src_idx: int, dst_idx: int) -> List[int]:
        from collections import deque

        q = deque([src_idx])
        parent = {src_idx: None}
        visited = []
        while q:
            u = q.popleft()
            if u in visited:
                continue
            visited.append(u)
            if u == dst_idx:
                break
            for v in self.room_graph[u]:
                if v in visited:
                    continue
                if v not in parent:
                    parent[v] = u
                    q.append(v)

        if dst_idx not in parent and dst_idx != src_idx:
            return []
        path = [dst_idx]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def _get_door_child_room(self, door_pos: Tuple[int, int]) -> Optional[int]:
        for ny, nx in self._neighbors4(door_pos):
            if not self._is_within(ny, nx):
                continue
            if self._inside_any_room((ny, nx)) and self.grid[ny, nx] != WALL:
                ridx = self._room_index_of_pos((ny, nx))
                if ridx is not None:
                    return ridx
        return None

    def _dist_to_nearest_door(self, pos: Tuple[int, int]) -> int:
        y, x = pos
        if not self.doors:
            return 9999
        return min(abs(y - dy) + abs(x - dx) for (dy, dx) in self.doors.keys())

    def _place_cupboard_in_room(self, room_idx_path: List) -> Tuple:
        room_idx_path = room_idx_path[-len(room_idx_path) // 2 :]
        for attempt in range(100):  # Retry loop
            cum_weights = list(
                accumulate([(i + 1) ** 2 for i in range(len(room_idx_path))])
            )
            room_idx = random.choices(room_idx_path, cum_weights=cum_weights, k=1)[0]
            # room_idx = random.choice(room_idx_path)
            pos = self._random_empty_in_rect(self.rooms[room_idx])
            if pos is None or not self._adjacent_squares_empty(pos):
                continue

            return pos, room_idx

        raise IndexError("Failed to place cupboard after 100 attempts.")

    def _place_key_in_room(self, _key_room_path: List) -> Tuple[Tuple[int, int], int]:
        _key_room_path = _key_room_path[-len(_key_room_path) // 2 :]
        for attempt in range(100):  # Retry loop
            cum_weights = list(
                accumulate([(i + 1) ** 2 for i in range(len(_key_room_path))])
            )
            room_idx = random.choices(_key_room_path, cum_weights=cum_weights, k=1)[0]
            # room_idx = random.choice(_key_room_path)
            pos = self._random_empty_in_rect(self.rooms[room_idx])
            if pos is None or not self._adjacent_squares_empty(pos):
                continue
            return pos, room_idx
        raise IndexError("Failed to place key after 100 attempts.")

    def _place_people(self):
        # Place people (any room), avoiding collisions
        for _ in range(self.num_people):
            for _ in range(1000):
                ridx = int(self._rng.integers(0, len(self.rooms)))
                y1, x1, y2, x2 = self.rooms[ridx]
                pos = self._random_empty_in_rect((y1, x1, y2, x2))
                if (
                    pos is not None
                    and self.grid[pos] == EMPTY
                    and self._adjacent_squares_empty(pos)
                ):
                    self.people.append(Person(pos=pos))
                    self.previous_traversable_code.append(self.grid[pos])
                    self.grid[pos] = PERSON_TILE
                    break
            else:
                assert False, "ERROR: Unable to place person after 1,000 attempts."

    def _adjacent_squares_empty(self, pos: Tuple[int, int]) -> bool:
        for n in self._neighbors4(pos):
            if (
                self.grid[n] != EMPTY
                or n in (self.agent_pos, self.exit_pos, self.start_pos)
                or n in self.cupboards
            ):
                return False
        else:  # All neighbours seem fine.
            return True

    def _boundary_door_position(self, center, elbow):
        # find the room that contains 'center'
        r = None
        for rect in self.rooms:
            y1, x1, y2, x2 = rect
            cy, cx = center
            if y1 <= cy <= y2 and x1 <= cx <= x2:
                r = rect
                break
        if r is None:
            return None
        y1, x1, y2, x2 = r
        ey, ex = elbow
        candidates = []
        for x in range(x1, x2 + 1):
            candidates.append((y1 - 1, x))
            candidates.append((y2 + 1, x))
        for y in range(y1, y2 + 1):
            candidates.append((y, x1 - 1))
            candidates.append((y, x2 + 1))
        candidates.sort(key=lambda p: abs(p[0] - ey) + abs(p[1] - ex))
        for cy, cx in candidates:
            if self.grid[cy, cx] == EMPTY:
                for ny, nx in self._neighbors4((cy, cx)):
                    if self.grid[ny, nx] == EMPTY and self._inside_any_room((ny, nx)):
                        return cy, cx
        return None

    # --------------- Helpers ---------------
    def _neighbors4(self, pos):
        y, x = pos
        return [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]

    def _color_for_chain(self, cid: int) -> Tuple[int, int, int]:
        # Deterministic bright-ish color per chain_id
        r = (37 * (cid + 1) * 73) % 200 + 30
        g = (91 * (cid + 1) * 53) % 200 + 30
        b = (29 * (cid + 1) * 97) % 200 + 30
        return int(r), int(g), int(b)

    def _is_pathable_for_debug(self, y: int, x: int) -> bool:
        """Passable for debug path drawing: allow empty, keys, exit, and doors (locked or unlocked)."""
        if not self._is_within(y, x):
            return False
        code = self.grid[y, x]
        if code in (EMPTY, KEY_TILE, EXIT, DOOR_LOCKED, DOOR_UNLOCKED):
            return True
        return False  # cupboards/walls are blocked for path

    def _bfs_path_cells(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Grid BFS path; if goal is not pathable (e.g., cupboard), we path to its nearest 4-neighbor cell."""
        from collections import deque

        def nearest_adjacent_targets(t: Tuple[int, int]) -> List[Tuple[int, int]]:
            # If target itself is pathable, use it; else use its valid neighbors
            ty, tx = t
            if self._is_pathable_for_debug(ty, tx):
                return [t]
            nbrs = [
                (ny, nx)
                for (ny, nx) in self._neighbors4(t)
                if self._is_pathable_for_debug(ny, nx)
            ]
            return nbrs or [t]  # fallback to the tile itself if nothing else

        targets = set(nearest_adjacent_targets(goal))
        if not targets:
            return []

        q = deque([start])
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {start: None}
        found: Tuple[int, int] = None

        while q:
            cur = q.popleft()
            if cur in targets:
                found = cur
                break
            for ny, nx in self._neighbors4(cur):
                if (ny, nx) not in prev and self._is_pathable_for_debug(ny, nx):
                    prev[(ny, nx)] = cur
                    q.append((ny, nx))

        if found is None:
            return []

        # Reconstruct
        path: List[Tuple[int, int]] = [found]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        path.reverse()
        # If we ended adjacent to a non-pathable goal (cupboard), optionally append the goal itself for drawing
        if goal not in path:
            path.append(goal)
        return path

    def _is_within(self, y, x):
        return 0 <= y < self.height and 0 <= x < self.width

    def _inside_any_room(self, pos):
        y, x = pos
        for y1, x1, y2, x2 in self.rooms:
            if y1 <= y <= y2 and x1 <= x <= x2:
                return True
        return False

    def _room_index_of_pos(self, pos: Tuple[int, int]) -> Optional[int]:
        y, x = pos
        for idx, (y1, x1, y2, x2) in enumerate(self.rooms):
            if y1 <= y <= y2 and x1 <= x <= x2:
                return idx
        return None

    def _pos_in_rect(self, pos, rect):
        y, x = pos
        y1, x1, y2, x2 = rect
        return y1 <= y <= y2 and x1 <= x <= x2

    def _random_empty_in_rect(self, rect) -> Tuple[int, int]:
        """Checks a given rectangle of squares for an empty space, returning either a random empty location or None."""
        r1, c1, r2, c2 = rect
        empties: List[Tuple[int, int]] = []
        for row in range(r1, r2 + 1):
            for col in range(c1, c2 + 1):
                if (
                    self.grid[row, col] == EMPTY
                    and (row, col)
                    not in [self.agent_pos, self.exit_pos, self.start_pos]
                    and (row, col) not in self.cupboards
                ):
                    empties.append((row, col))
        return random.choice(empties) if empties else None

    def _random_empty_global(self) -> Tuple[int, int]:
        locs = np.argwhere(self.grid == EMPTY)
        # Filter out occupied positions
        valid_positions = []
        for l in locs:
            l = tuple(l)
            if (
                l not in (self.agent_pos, self.exit_pos, self.start_pos)
                # not any(_a == l for _a in (self.agent_pos, self.exit_pos))
                and l not in self.cupboards
                and not any(p.pos == l for p in self.people)
            ):
                valid_positions.append(l)

        return random.choice(valid_positions) if valid_positions else None

    def _adjacent_of_type(self, codes: set) -> Optional[Tuple[int, int]]:
        for pos in self._neighbors4(self.agent_pos):
            if self.grid[pos] in codes:
                return pos
        return None

    def _ahead_of_type_with_chain_id(self, codes: Set[int]) -> Tuple[int, int]:
        offset = forward_offset[self._current_direction]
        arow, acol = self.agent_pos
        row, col = arow + offset[0], acol + offset[1]
        if self.grid[row, col] in codes:
            if (
                self.grid[row, col] == CUPBOARD_LOCKED
                and self.cupboards[(row, col)].chain_id in self.inventory["key_ids"]
            ):
                return row, col
            if (
                self.grid[row, col] in self.door_locked_types
                and self.doors[(row, col)].chain_id in self.inventory["keycard_ids"]
            ):
                return row, col

    def _ahead_of_type(self, codes: Set[int]) -> Tuple[int, int]:
        offset = forward_offset[self._current_direction]
        arow, acol = self.agent_pos
        row, col = arow + offset[0], acol + offset[1]
        if self.grid[row, col] in codes:
            return row, col
        return None

    def _forward_cupboard_with_keycard_unlocked(self) -> Optional[Tuple[int, int]]:
        offset = forward_offset[self._current_direction]
        arow, acol = self.agent_pos
        row, col = arow + offset[0], acol + offset[1]
        if self.grid[row, col] == CUPBOARD_UNLOCKED_KEYCARD:
            c = self.cupboards.get((row, col))
            if c and (not c.locked) and c.has_keycard:
                return row, col
        return None

    def _adjacent_doors(self) -> Optional[List[Tuple[int, int]]]:
        doors = []
        for row, col in self._neighbors4(self.agent_pos):
            if self.grid[row, col] in self.door_types:
                doors.append((row, col))
        return doors if doors else None

    def _unlock_cupboard(self, pos: Tuple[int, int]):
        c = self.cupboards[pos]
        c.locked = False
        if c.has_keycard:
            self.grid[pos] = CUPBOARD_UNLOCKED_KEYCARD
        else:
            self.grid[pos] = CUPBOARD_UNLOCKED

    def _unlock_door(self, pos: Tuple[int, int]):
        dinfo = self.doors[pos]
        dinfo.locked = False
        self.grid[pos] = DOOR_UNLOCKED

    def _precompute_los_paths(
        self, window_size: int
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        paths = {}
        r = window_size // 2
        start = (r, r)

        for row in range(window_size):
            for col in range(window_size):
                if (row, col) == start:
                    paths[(row, col)] = []
                    continue

                path = self.get_path(start, (row, col))

                paths[(row, col)] = path
        return paths

    def get_path(self, start, end) -> List[Tuple[int, int]]:
        r0, c0 = start[0] + 0.5, start[1] + 0.5
        r1, c1 = end[0] + 0.5, end[1] + 0.5

        assert start != end, "Start and end are the same"

        r_grad = r1 - r0
        c_grad = c1 - c0

        num_steps = max(abs(r_grad), abs(c_grad))
        direction = (
            "r" if abs(r_grad) >= abs(c_grad) else "c"
        )  # Is row or column the axis to step along?

        path = []
        if direction == "r":
            grad = abs(c_grad / r_grad)
            for s in range(int(num_steps) + 1):
                path.append(
                    (
                        int(start[0] + ((r_grad + 1e-6) / (abs(r_grad) + 1e-6)) * s),
                        int(
                            math.floor(
                                c0
                                + ((c_grad + 1e-6) / (abs(c_grad) + 1e-6)) * (s * grad)
                            )
                        ),
                    )
                )
        else:
            grad = abs(r_grad / c_grad)
            for s in range(int(num_steps) + 1):
                path.append(
                    (
                        int(
                            (
                                math.floor(
                                    r0
                                    + ((r_grad + 1e-6) / (abs(r_grad) + 1e-6))
                                    * (s * grad)
                                )
                            )
                        ),
                        int(start[1] + ((c_grad + 1e-6) / (abs(c_grad) + 1e-6)) * s),
                    )
                )

        return path

    def _has_los(self, target: Tuple[int, int], subgrid: np.ndarray) -> bool:
        path = self._los_paths[target]
        if not path:  # It's the center point
            return True

        prev_r, prev_c = path[0][0], path[0][1]

        for r, c in path[1:]:
            if subgrid[r, c] not in self._passable_tiles_set and (r, c) != path[-1]:
                return False

            if prev_r != r and prev_c != c:
                if (
                    subgrid[r, prev_c] not in self._passable_tiles_set
                    and subgrid[prev_r, c] not in self._passable_tiles_set
                ):
                    return False

            prev_r, prev_c = r, c

        return True

    def _create_7x7_observation(
        self, ay: int, ax: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        window_size = self.obs_window_size
        r = window_size // 2
        H, W = self.grid.shape

        # Default unseen = UNKNOWN (255) for tiles, -2 for chain_ids
        window = np.full((window_size, window_size), UNKNOWN, dtype=np.uint8)
        window_chains = np.full((window_size, window_size), -2, dtype=np.int64)
        subgrid = np.full((window_size, window_size), WALL, dtype=np.uint8)
        subgrid_chains = np.full((window_size, window_size), -1, dtype=np.int64)

        y0 = max(0, ay - r)
        y1 = min(H, ay + r + 1)
        x0 = max(0, ax - r)
        x1 = min(W, ax + r + 1)

        dy = ay - y0
        dx = ax - x0
        w_start_y = 0 + (r - dy)
        w_start_x = 0 + (r - dx)
        w_size_y = y1 - y0
        w_size_x = x1 - x0

        subgrid[w_start_y : w_start_y + w_size_y, w_start_x : w_start_x + w_size_x] = (
            self.grid[y0:y1, x0:x1]
        )
        subgrid_chains[
            w_start_y : w_start_y + w_size_y, w_start_x : w_start_x + w_size_x
        ] = self.chain_id_grid[y0:y1, x0:x1]

        # Fill window with visible tiles
        for row in range(window_size):
            for col in range(window_size):
                if self._has_los((row, col), subgrid):
                    window[row, col] = subgrid[row, col]
                    window_chains[row, col] = subgrid_chains[row, col]

                    # Update global discovered_grid
                    gy = y0 + (row - w_start_y)
                    gx = x0 + (col - w_start_x)
                    if 0 <= gy < H and 0 <= gx < W:
                        self.discovered_grid[gy, gx] = True

        # Double-scan to account for initial scan-order issues
        for row in range(window_size - 1, -1, -1):
            for col in range(window_size - 1, -1, -1):
                if not self._has_los((row, col), window):
                    window[row, col] = 100
                    window_chains[row, col] = -1

        return window, window_chains

    def _get_obs(self):
        ppl_pos = np.zeros((self.num_people, 2), dtype=np.int64)
        ppl_follow = np.zeros((self.num_people,), dtype=np.int8)
        ppl_resc = np.zeros((self.num_people,), dtype=np.int8)
        for i, p in enumerate(self.people):
            ppl_pos[i, 0] = p.pos[0]
            ppl_pos[i, 1] = p.pos[1]
            # ppl_pos[i] = np.array([p.pos[0], p.pos[1]], dtype=np.int64)
            ppl_follow[i] = 1 if p.following else 0
            ppl_resc[i] = 1 if p.rescued else 0
        ay, ax = self.agent_pos
        window, window_chains = self._create_7x7_observation(ay, ax)

        obs = {
            "grid": np.expand_dims(window, axis=0).astype(np.uint8).copy(),
            "chain_grid": np.expand_dims(window_chains, axis=0).astype(np.int64).copy(),
        }
        return obs
