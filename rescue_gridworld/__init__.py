from gymnasium.envs.registration import register
from .env_constants import *
from .env import RescueGridworldEnv

# Register the environment
register(
    id="RescueGridworld-v0",
    entry_point="rescue_gridworld.env:RescueGridworldEnv",
)

__all__ = ["RescueGridworldEnv"]
