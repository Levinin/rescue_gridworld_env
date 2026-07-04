from gymnasium.envs.registration import register
from .env_constants import *
from .env import RescueGridworldEnv
from .create_rooms import create_room_data_grid, ensure_capacity

# Register the environment
register(
    id="RescueGridworld-v1",
    entry_point="rescue_gridworld.env:RescueGridworldEnv",
)

__all__ = ["RescueGridworldEnv"]
