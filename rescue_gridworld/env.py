import math
import os
import random
import traceback
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

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
    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.num_rooms = max(1, num_rooms)
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

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=255, shape=(1, 7, 7), dtype=np.uint8),
                "chain_grid": spaces.Box(low=-2, high=500, shape=(1, 7, 7), dtype=np.int16),
            }
        )

        # dynamic state
        self.grid = None  # np.ndarray
        self.chain_id_grid = None  # np.ndarray
        self.rooms: List[
            Tuple[int, int, int, int]
        ] = []  # list of room rects (y1, x1, y2, x2)
        self.doors: Dict[Tuple[int, int], DoorInfo] = {}  # pos -> DoorInfo
        self.room_graph: Dict[int, List[int]] = {}  # adjacency on room indices
        self.cupboards: Dict[Tuple[int, int], Cupboard] = {}
        self.people: List[Person] = []
        self.people_following: int = 0  # How many people are following the agent
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.exit_pos: Tuple[int, int] = (0, 0)
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
        self._person_move_step = lambda x: (x + 1) % 3  # Loop 3 steps

        # rendering
        self._screen = None
        self._clock = None
        self._surface = None

        self._episode_rewards = 0

        # Plan render
        self.debug_draw_chains = debug_draw_chains
        self.chain_plan: Dict[
            int, Dict[str, Tuple[int, int]]
        ] = {}  # chain_id -> {"key":(y,x), "cupboard":(y,x), "door":(y,x)}
        self._font_small = None  # lazy-init in render
        self.passable_tiles = [EMPTY, KEY_TILE, PERSON_TILE, DOOR_UNLOCKED, EXIT]

    # --------------- Gym API ---------------
    def reset(self, *, seed: Optional[int] = None, options: Dict[str, Any] | None = {}):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        self._generate_level()
        if self.reset_options.get("get_all_chains", False):
            # print("Getting all chains")
            self.obtain_all_chain_keys_keycards()

        obs = self._get_obs()
        self._step_count = 0
        self.people_following = 0
        self._episode_rewards = 0
        info = self._update_info()
        # if self.render_mode == "human":
        #     self.render()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"ERROR: {action} is invalid."
        reward = -0.05
        terminated = False
        truncated = False
        info = {}
        info["action_success"] = False
        info["action_code"] = int(action)

        # Sometimes a random action will occur.
        if random.random() < self.stochastic_transition_chance:
            action = random.randint(0, 8)

        y, x = self.agent_pos
        dy, dx = 0, 0
        if action in (UP, DOWN, LEFT, RIGHT):
            if action == UP:
                dy = -1
            elif action == DOWN:
                dy = 1
            elif action == LEFT:
                dx = -1
            else:
                dx = 1
            ny, nx = y + dy, x + dx
            if self.grid[ny, nx] in self.passable_tiles:
                # if self._is_within(ny, nx) and self._can_enter(ny, nx):
                self.agent_pos = (ny, nx)
                info["action_success"] = True

            if self.grid[self.agent_pos] == EXIT:
                terminated = True
                reward += 5.0
                if all(p.following for p in self.people):
                    reward += 50.0
                    info["success"] = True

        elif action == PICK_KEY:
            if self.grid[y, x] == KEY_TILE:
                self.grid[y, x] = EMPTY
                self.chain_id_grid[y, x] = -1
                # Chain-bound or generic?
                info["action_success"] = True
                entry = self._chain_at_pos.pop((y, x), None)
                if entry is not None:
                    kind, cid = entry
                    if kind == "key":
                        self.inventory["key_ids"].add(cid)
                else:
                    self.inventory["keys"] += 1
                reward += 5

        elif action == UNLOCK_CUPBOARD:
            cpos = self._adjacent_of_type_with_chain_id({CUPBOARD_LOCKED})
            # cpos = self._adjacent_of_type({CUPBOARD_LOCKED})
            if cpos is not None:
                cup = self.cupboards.get(cpos)
                if cup and cup.locked:
                    if cup.chain_id is not None:
                        # Strict: require matching chain key id
                        if cup.chain_id in self.inventory["key_ids"]:
                            self.inventory["key_ids"].remove(cup.chain_id)
                            self._unlock_cupboard(cpos)
                            reward += 5
                            info["action_success"] = True
                    else:
                        # Unbound cupboard: use generic key if available
                        if self.inventory["keys"] > 0:
                            self.inventory["keys"] -= 1
                            self._unlock_cupboard(cpos)
                            reward += 5
                            info["action_success"] = True

        elif action == PICK_KEYCARD:
            cpos = self._adjacent_cupboard_with_keycard_unlocked()
            if cpos is not None:
                cup = self.cupboards[cpos]
                cup.has_keycard = False
                if cup.chain_id is not None:
                    self.inventory["keycard_ids"].add(cup.chain_id)
                else:
                    self.inventory["keycards"] += 1
                self.grid[cpos] = CUPBOARD_UNLOCKED
                reward += 5
                info["action_success"] = True

        elif action == UNLOCK_DOOR:
            dpos = self._adjacent_of_type_with_chain_id({DOOR_LOCKED})
            # dpos = self._adjacent_of_type({DOOR_LOCKED})
            if dpos is not None:
                dinfo = self.doors.get(dpos)
                if dinfo and dinfo.locked:
                    if dinfo.chain_id is not None:
                        if dinfo.chain_id in self.inventory["keycard_ids"]:
                            self.inventory["keycard_ids"].remove(dinfo.chain_id)
                            self._unlock_door(dpos)
                            reward += 5
                            info["action_success"] = True
                    else:
                        if self.inventory["keycards"] > 0:
                            self.inventory["keycards"] -= 1
                            self._unlock_door(dpos)
                            reward += 5
                            info["action_success"] = True

        elif action == TALK_PERSON:
            for p in self.people:
                if not p.following and p.pos == self.agent_pos:
                    p.following = True
                    self.grid[p.pos] = EMPTY
                    self.people_following += 1
                    reward += 5
                    # print("Found someone.", p.pos)
                    info["action_success"] = True

        self._move_people()

        info.update(self._update_info())

        self._step_count += 1
        if self._step_count >= self.max_steps:
            truncated = True

        # if self.render_mode == "human":
        #     self.render()
        self._episode_rewards += reward

        return self._get_obs(), reward, terminated, truncated, info

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
                elif code == DOOR_UNLOCKED:
                    color = (60, 200, 60)
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
                pygame.draw.rect(surf, color, rect)
                pygame.draw.rect(surf, (210, 210, 210), rect, width=1)

                if code == KEY_TILE:
                    pygame.draw.circle(surf, (255, 215, 0), rect.center, tile // 5)

        for pos, c in self.cupboards.items():
            if c.has_keycard:
                y, x = pos
                cx, cy = x * tile + tile // 2, y * tile + tile // 2
                pygame.draw.rect(surf, (0, 0, 0), (x * tile + tile // 4, y * tile + tile // 4, tile - tile // 2, tile - tile // 2), width=1) #(cx - 3, cy - 8, 6, 16), width=1)

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
            #    We draw small rectangles along cells in the path using the chain color.
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

        ay, ax = self.agent_pos
        rect = pygame.Rect(ax * tile, ay * tile, tile, tile)
        pygame.draw.circle(surf, (250, 250, 250), rect.center, tile // 3)
        pygame.draw.circle(surf, (0, 0, 0), rect.center, tile // 3, width=2)

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
                f"RescueGridworld | Step:{self._step_count} | Episode Reward:{round(self._episode_rewards,1)}"
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

    # PRIVATE METHODS
    def _move_people(self) -> None:
        self._person_move_step_count = self._person_move_step(
            self._person_move_step_count
        )
        if self._person_move_step_count != 0:  # Only move once per cycle.
            return

        def get_empty_adjacent_squares(p: Person) -> Tuple[int, int]:
            adj = []
            if self.grid[p.pos[0], p.pos[1] + 1] == EMPTY:
                adj.append((p.pos[0], p.pos[1] + 1))
            if self.grid[p.pos[0], p.pos[1] - 1] == EMPTY:
                adj.append((p.pos[0], p.pos[1] - 1))
            if self.grid[p.pos[0] + 1, p.pos[1]] == EMPTY:
                adj.append((p.pos[0] + 1, p.pos[1]))
            if self.grid[p.pos[0] - 1, p.pos[1]] == EMPTY:
                adj.append((p.pos[0] - 1, p.pos[1]))
            if len(adj) == 0:  # Just in case there are no adjacent empty locations.
                adj.append((p.pos[0], p.pos[1]))
            return random.choice(adj)

        for p in self.people:
            if p.following:
                continue
            empty_adjacent: Tuple[int, int] = get_empty_adjacent_squares(p)
            self.grid[p.pos[0], p.pos[1]] = EMPTY
            p.pos = empty_adjacent
            self.grid[p.pos[0], p.pos[1]] = PERSON_TILE

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
            "agent_pos",
            "exit_pos",
            "inventory",
            "_chain_at_pos",
            "_step_count",
            "solvable_plan",
            "chain_plan",
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

    def _find_or_create_door_into_room(self, room_idx) -> tuple or None:
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

    def _ensure_capacity(self):
        min_w, min_h = 12, 12
        spacing = 4
        cw = min_w + spacing
        ch = min_h + spacing

        # Prevent division by zero
        if self.width - 2 <= 0: self.width = 4
        if self.height - 2 <= 0: self.height = 4

        eff_w = max(self.width - 2, 0)
        eff_h = max(self.height - 2, 0)
        
        tiles_x = eff_w // cw
        tiles_y = eff_h // ch
        current_capacity = max(tiles_x * tiles_y, 1)

        # If the grid is already large enough and meets absolute minimums, return
        if current_capacity >= self.num_rooms and self.width >= min_w + 2 and self.height >= min_h + 2:
            return

        # 1. Calculate the ideal ratio of X tiles to Y tiles to maintain aspect ratio
        aspect_ratio_tiles = (self.width * ch) / (self.height * cw)

        # 2. Derive the exact minimum number of abstract tiles needed
        required_tiles_y = math.ceil(math.sqrt(self.num_rooms / aspect_ratio_tiles))
        required_tiles_x = math.ceil(self.num_rooms / required_tiles_y)

        # 3. Translate abstract tiles back to minimum physical grid dimensions
        min_required_width = (required_tiles_x * (cw + 2)) + 2
        min_required_height = (required_tiles_y * (ch + 2)) + 2

        # 4. Extract the uniform scale factor
        scale_x = min_required_width / self.width
        scale_y = min_required_height / self.height
        uniform_scale = max(scale_x, scale_y, 1.0)

        # 5. Apply scale
        self.width = math.ceil(self.width * uniform_scale)
        self.height = math.ceil(self.height * uniform_scale)
        
        # Absolute minimums
        self.width = max(self.width, min_w + 2)
        self.height = max(self.height, min_h + 2)

    # --------------- Level generation (with solvable chain) ---------------
    def _generate_level(self):
        # Retry loop for robust generation
        self._ensure_capacity()
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
                print(e, traceback.format_exc())
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

    def _try_generate_one(self):
        H, W = self.height, self.width
        self._reset_class_variables(H, W)

        if not self._place_rooms(H, W):
            return False

        self._connect_rooms()
        start_idx = self._place_start_exit()
        self._place_items(start_idx)
        self._place_people()

        if len(self.doors) == 0:
            return False
        return True

    def _place_rooms(self, H: int, W: int) -> bool:
        # --- Room placement ---
        min_w, min_h = 8, 8
        attempts = 0
        min_spacing = 4
        overlaps: bool = False
        while len(self.rooms) < self.num_rooms and attempts < 800:
            overlaps = False
            attempts += 1
            rw = int(self._rng.integers(min_w, max(min(W - 4, 12), min_w + 1)))
            rh = int(self._rng.integers(min_h, max(min(H - 4, 10), min_h + 1)))
            x1 = int(self._rng.integers(1, W - rw - 1))
            y1 = int(self._rng.integers(1, H - rh - 1))
            x2, y2 = x1 + rw - 1, y1 + rh - 1

            for ry1, rx1, ry2, rx2 in self.rooms:
                if not (
                    x2 + min_spacing < rx1
                    or rx2 + min_spacing < x1
                    or y2 + min_spacing < ry1
                    or ry2 + min_spacing < y1
                ):
                    overlaps = True
                    break
            if overlaps:
                continue

            self.grid[y1 : y2 + 1, x1 : x2 + 1] = EMPTY
            self.rooms.append((y1, x1, y2, x2))

        if overlaps:
            return False

        assert len(self.rooms) >= 1, "No rooms placed."
        return True

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
            (y0, x0), (y1, x1) = centers[i], centers[j]

            x_step = 1 if x1 >= x0 else -1
            y_step = 1 if y1 >= y0 else -1

            # --- compute elbow and push it one tile away from adjacent room if needed ---
            elbow_y, elbow_x = y0, x1

            # only care when the path is actually L-shaped
            if x0 != x1 and y0 != y1:
                # check 4-neighbourhood for existing EMPTY (room) tiles
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = elbow_y + dy, elbow_x + dx
                    if (
                        self.grid[elbow_y, elbow_x] != EMPTY
                        and self.grid[ny, nx] == EMPTY
                    ):
                        # move elbow one step *away* from that empty neighbour
                        # print("FOUND THE PROBLEM CASE")
                        elbow_y -= dy
                        elbow_x -= dx
                        # Adjust the centres to match the elbows.
                        centers[i] = (centers[i][0] - dy, centers[i][1] - dx)
                        centers[j] = (centers[j][0] - dy, centers[j][1] - dx)
                        break

            elbow = (elbow_y, elbow_x)
            corridor_elbows[(i, j)] = elbow
            corridor_elbows[(j, i)] = elbow
            (y0, x0), (y1, x1) = centers[i], centers[j]
            # --- carve corridor exactly as before (from center to center) ---
            for x in range(x0, x1 + x_step, x_step):
                self.grid[y0, x] = EMPTY
            for y in range(y0, y1 + y_step, y_step):
                self.grid[y, x1] = EMPTY

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

    def _place_start_exit(self) -> int:
        # --- Choose start/exit rooms ---
        start_idx = int(self._rng.integers(0, len(self.rooms)))

        # Place the exit in a wall
        for _ in range(100):
            exit_set = False
            exit_options = list(range(len(self.rooms)))
            exit_options.remove(start_idx)
            exit_idx = random.choice(exit_options)
            assert start_idx != exit_idx, (
                "Fatal Error: The start and finish rooms should not be the same."
            )

            # Place exit
            ey1, ex1, ey2, ex2 = self.rooms[exit_idx]
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
                    self.exit_pos = (ey1 + i, ex1 - 1)
                    # self.exit_pos = self._random_empty_in_rect((ey1+2, ex1+2, ey2-2, ex2-2))
                    self.grid[self.exit_pos] = EXIT
                    exit_set = True
                    break
            if exit_set:
                break
        else:
            assert False, (
                "Fatal Error: Unable to place exit while creating environment."
            )

        # Place agent in start room
        sy1, sx1, sy2, sx2 = self.rooms[start_idx]
        self.agent_pos = self._random_empty_in_rect((sy1, sx1, sy2, sx2))
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
            # for min_d in (3, 2):
            #     for pos in candidates:
            #         if self._dist_to_nearest_door(pos) >= min_d:
            #             return pos, room_idx

        # print(f"Failed to place cupboard in {room_idx_path} after 100 attempts.")
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
        # Fallback after all attempts
        # print(f"Unable to place key in {_key_room_path}")
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
                    self.grid[pos] = PERSON_TILE
                    break
            else:
                assert False, "ERROR: Unable to place person after 1,000 attempts."

    def _adjacent_squares_empty(self, pos: Tuple[int, int]) -> bool:
        for n in self._neighbors4(pos):
            if (
                self.grid[n] != EMPTY
                or n in (self.agent_pos, self.exit_pos)
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
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        found: Optional[Tuple[int, int]] = None

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
        path = [found]
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

    def _random_empty_in_rect(self, rect) -> Optional[Tuple[int, int]]:
        """Checks a given rectangle of squares for an empty space, returning either a random empty location or None."""
        r1, c1, r2, c2 = rect
        empties: List[Tuple[int, int]] = []
        for row in range(r1, r2 + 1):
            for col in range(c1, c2 + 1):
                if (
                    self.grid[row, col] == EMPTY
                    and (row, col) not in [self.agent_pos, self.exit_pos]
                    and (row, col) not in self.cupboards
                ):
                    empties.append((row, col))
        return random.choice(empties) if empties else None

    def _random_empty_global(self) -> Tuple[int, int] | None:
        locs = np.argwhere(self.grid == EMPTY)
        # Filter out occupied positions
        valid_positions = []
        for l in locs:
            l = tuple(l)
            if (
                l not in (self.agent_pos, self.exit_pos)
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

    def _adjacent_of_type_with_chain_id(self, codes: set) -> Optional[Tuple[int, int]]:
        """Check if the neighbours are of type code and we have the chain_id item we need"""
        for row, col in self._neighbors4(self.agent_pos):
            if self.grid[row, col] in codes:
                if (
                    self.grid[row, col] == CUPBOARD_LOCKED
                    and self.cupboards[(row, col)].chain_id in self.inventory["key_ids"]
                ):
                    return row, col
                if (
                    self.grid[row, col] == DOOR_LOCKED
                    and self.doors[(row, col)].chain_id in self.inventory["keycard_ids"]
                ):
                    return row, col
        return None

    def _adjacent_cupboard_with_keycard_unlocked(self) -> Optional[Tuple[int, int]]:
        for row, col in self._neighbors4(self.agent_pos):
            if self.grid[row, col] == CUPBOARD_UNLOCKED_KEYCARD:
                c = self.cupboards.get((row, col))
                if c and (not c.locked) and c.has_keycard:
                    return row, col
        return None

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

    def _create_7x7_observation(self, ay: int, ax: int) -> Tuple[np.ndarray, np.ndarray]:
        window_size = 7
        r = window_size // 2
        H, W = self.grid.shape

        # Default unseen = UNKNOWN (255) for tiles, -2 for chain_ids
        window = np.full((window_size, window_size), UNKNOWN, dtype=np.uint8)
        window_chains = np.full((window_size, window_size), -2, dtype=np.int16)
        subgrid = np.full((window_size, window_size), WALL, dtype=np.uint8)
        subgrid_chains = np.full((window_size, window_size), -1, dtype=np.int16)

        y0 = max(0, ay - r)
        y1 = min(H, ay + r + 1)
        x0 = max(0, ax - r)
        x1 = min(W, ax + r + 1)

        dy = ay - y0
        dx = ax - x0
        w_start_y = 0 + (3 - dy)
        w_start_x = 0 + (3 - dx)
        w_size_y = y1 - y0
        w_size_x = x1 - x0

        subgrid[w_start_y : w_start_y + w_size_y, w_start_x : w_start_x + w_size_x] = (
            self.grid[y0:y1, x0:x1]
        )
        subgrid_chains[w_start_y : w_start_y + w_size_y, w_start_x : w_start_x + w_size_x] = (
            self.chain_id_grid[y0:y1, x0:x1]
        )

        def get_path(start, end) -> List[Tuple[int, int]]:
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
                            int(
                                start[0] + ((r_grad + 1e-6) / (abs(r_grad) + 1e-6)) * s
                            ),
                            int(
                                math.floor(
                                    c0
                                    + ((c_grad + 1e-6) / (abs(c_grad) + 1e-6))
                                    * (s * grad)
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
                            int(
                                start[1] + ((c_grad + 1e-6) / (abs(c_grad) + 1e-6)) * s
                            ),
                        )
                    )

            return path

        def has_los(start, end, _subgrid) -> bool:

            if start == end:
                return True

            path = get_path(start, end)

            prev_r, prev_c = start[0], start[1]
            for r, c in path[
                1:
            ]:  # We have start, and don't care if the target is passable
                # check passable.
                if _subgrid[r, c] not in self.passable_tiles and (r, c) != path[-1]:
                    return False

                # Check if this is a corner
                if prev_r != r and prev_c != c:
                    # If there is a pinch then we have no path
                    if (
                        _subgrid[r, prev_c] not in self.passable_tiles
                        and _subgrid[prev_r, c] not in self.passable_tiles
                    ):
                        return False

                prev_r, prev_c = r, c

            # We got through the full path!
            return True

        # Fill window with visible tiles
        for row in range(window_size):
            for col in range(window_size):
                if has_los((r, r), (row, col), subgrid):
                    window[row, col] = subgrid[row, col]
                    window_chains[row, col] = subgrid_chains[row, col]

        # Double-scan to account for initial scan-order issues
        for row in range(window_size - 1, -1, -1):
            for col in range(window_size - 1, -1, -1):
                if not has_los((r, r), (row, col), window):
                    window[row, col] = 100
                    window_chains[row, col] = -1

        return window, window_chains

    def _get_obs(self):
        ppl_pos = np.zeros((self.num_people, 2), dtype=np.int16)
        ppl_follow = np.zeros((self.num_people,), dtype=np.int8)
        ppl_resc = np.zeros((self.num_people,), dtype=np.int8)
        for i, p in enumerate(self.people):
            ppl_pos[i] = np.array([p.pos[0], p.pos[1]], dtype=np.int16)
            ppl_follow[i] = 1 if p.following else 0
            ppl_resc[i] = 1 if p.rescued else 0
        ay, ax = self.agent_pos
        window, window_chains = self._create_7x7_observation(ay, ax)

        obs = {
            "grid": np.expand_dims(window, axis=0).astype(np.uint8).copy(),
            "chain_grid": np.expand_dims(window_chains, axis=0).astype(np.int16).copy(),
        }
        return obs
