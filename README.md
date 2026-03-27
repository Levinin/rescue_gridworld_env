# Rescue Gridworld

This environment provides a Search and Rescue gridworld where the objective is to rescue all the people from the maze. To achieve this requires completion of a temporally extended sequence of tasks as follows.

- Exploration of the maze, opening locked doors to progress. Unlocking a door requires completion of the subtask sequence *collect key* -> *unlock cupboard* -> *collect keycard* -> *unlock door*.
- Talk to people discovered so they follow the agent, rescuing them. People are non-stationary within the environment moving randomly every 3 timestaps.
- The objective is to exit having rescued all the people, but it is also possible to exit without having done so.

## Installation
```bash
pip install rescue_gridworld
```

## Quick Start
You can run a random agent to test the environment by copying run_random_agent.py to a local directory and running:
```bash
python run_random_agent.py --render --steps 1000 --rooms 5
```

## Manual Usage
```python
import gymnasium as gym
import rescue_gridworld

env = gym.make("RescueGridworld-v0", render_mode="human", tile_size=tile_size, num_rooms=5)
obs, info = env.reset()
# ... step the env ...
```

## Action Space

There are 9 actions in the environment:
- 0: Move up.
- 1: Move down.
- 2: Move left.
- 3: Move right.
- 4: Collect key.
- 5: Unlock cupboard.
- 6: Collect keycard.
- 7: Unlock door.
- 8: Talk to person.


## Observation Space

The observation provides 2 7x7 "windows". 
- The first is a line-of-sight observation of the local environment.
- The second is a line-of-sight filtered set of "chain id's". 

The Python definitions are as follows to allow the grid to be viewed as an image.:
```python
    self.observation_space = spaces.Dict(
        {
            "grid": spaces.Box(low=0, high=255, shape=(1, 7, 7), dtype=np.uint8),
            "chain_grid": spaces.Box(low=-2, high=500, shape=(1, 7, 7), dtype=np.int16),
        }
    )
```


## Rewards

Reward of 5 is given for completion of subtasks:
- Collect key.
- Unlock cupboard.
- Collect keycard.
- Unlock door.
- Talk to person.
- Exit.

An additional reward of 50 is given if all people have been saved before exiting.

A step penalty of -0.01 is applied to encourage efficiency.


## Episode termination conditions

Episodes terminate after max steps, or if the agent moves onto the exit square.
