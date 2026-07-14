import argparse

import rescue_gridworld
import gymnasium as gym

def run_random_agent(num_steps: int, num_episodes: int, render: bool, tile_size: int, rooms: int, people: int) -> None:

    env = gym.make("RescueGridworld-v2", render_mode="human",
                   tile_size=tile_size, num_rooms=rooms, num_people=people)

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        print(f"--- Starting Episode {episode + 1} ---")

        for step in range(num_steps):
            if render:
                env.render()

            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # The dict obs has 'grid' (uint8) and 'chain_grid' (int16)
            # You can access them like:
            grid = obs['grid']
            chain_grid = obs['chain_grid']
            print(grid)
            print(chain_grid)

            if terminated or truncated:
                print(f"Episode finished after {step + 1} steps. Total Reward: {episode_reward:.2f}")
                break
        else:
            print(f"Reached maximum steps ({num_steps}) for this episode. Total Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a random agent in RescueGridworldEnv.")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum number of steps per episode.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--render", action="store_true", default=True, help="Whether to render the environment.")
    parser.add_argument("--tilesize", type=int, default=12, help="Tile size when rendering.")
    parser.add_argument("--rooms", type=int, default=12, help="Number of rooms.")
    parser.add_argument("--people", type=int, default=30, help="Number of people.")

    args = parser.parse_args()

    run_random_agent(args.steps, args.episodes, args.render, args.tilesize, args.rooms, args.people)
