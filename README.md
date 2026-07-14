# Rescue Gridworld

This environment provides a Search and Rescue gridworld where the objective is to rescue all the people from the maze. To achieve this requires completion of a temporally extended sequence of tasks as follows.

- Exploration of the maze, opening locked doors to progress. Unlocking a door requires completion of the subtask sequence *collect key* -> *unlock cupboard* -> *collect keycard* -> *unlock door*.
- Talk to people discovered so they follow the agent, rescuing them. People are non-stationary within the environment moving randomly every 4 timestaps.
- The objective is to exit having rescued all the people, but it is also possible to exit without having done so.
- There is an optional fire spreading through the maze which the agent must avoid. This fire can prevent the agent reaching all the people and potentially cut the agent off from exits.

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

There are 10 actions in the environment.

Action codes are defined in `env_constants.py` and can be imported as:
```python
import rescue_gridworld.env_constants as ec
```

The action codes are defined as follows:
```python
MOVE_FORWARD = 0
ROTATE_LEFT = 1
ROTATE_RIGHT = 2
COLLECT_KEY = 3
UNLOCK_CUPBOARD = 4
COLLECT_KEYCARD = 5
UNLOCK_DOOR = 6
TALK_TO_PERSON = 7              # This rescues the person.
CHECK_DOOR_TEMPERATURE = 8
OPEN_DOOR = 9
```

## Observation Space

The observation provides 2 *n x n* "windows" (default *n = 7*, configurable by setting obs_window_size to any odd number when creating the environment).
- The first, ["grid"], is a line-of-sight observation of the local environment.
- The second, ["chain_grid"], is a line-of-sight filtered set of "chain id's". 

The Python definitions are as follows to allow the grid to be viewed as an image.:
```python
    self.observation_space = spaces.Dict(
        {
            "grid": spaces.Box(low=0, high=255, shape=(1, n, n), dtype=np.uint8),
            "chain_grid": spaces.Box(low=-2, high=500, shape=(1, n, n), dtype=np.int16),
        }
    )
```

Tile codes are specified in env_constants.py and can be imported as:
```python
import rescue_gridworld.env_constants as ec
```

The codes are defined as follows:
```python
EMPTY = 0                         # passable
WALL = 5
DOOR_LOCKED = 10
DOOR_LOCKED_HOT = 15
DOOR_LOCKED_COLD = 17
DOOR_UNLOCKED = 20
DOOR_OPEN = 22                    # passable
DOOR_UNLOCKED_HOT = 25
DOOR_UNLOCKED_COLD = 27
CUPBOARD_LOCKED = 30
CUPBOARD_UNLOCKED = 40
CUPBOARD_UNLOCKED_KEYCARD = 50
KEY_TILE = 60                    # passable
PERSON_TILE = 70                 # passable
EXIT = 80                        # passable
FIRE = 90                        # passable
UNKNOWN = 255
```

## Rewards

Reward of 5 is given for completion of subtasks:
- Collect key.
- Unlock cupboard.
- Collect keycard.
- Unlock door *confirmed not to be hot*.
- Rescue person.
- Exit.

An additional reward of 50 is given if all people have been saved before exiting.

A step penalty of -0.05 is applied to encourage efficiency.


## Info Dict

The info dict contains much of the underlying environment state to help debug. This includes the agent global position, the true underlying state, agent key and keycard inventory, people and their locations, and so on. 


## Dynamics

The agent can move forward, and rotate left or right. The agent can not move through walls, closed doors, or cupboards. The agent can step into fire and will cease to function, ending the episode. 

Observations are *n x n* snapshots and are LoS only, with walls, closed doors and cupboards occluding the view.

Fire expands from a random starting location at a configurable rate. It will overtake people who have not been rescued, and overtake the agent if it does not move away. 

If the agent opens a door with fire close to it, the fire will expand 3 increments immediately to simulate an influx of oxygen to the fire.

To understand whether the fire is near the door, the agent can check the door temperature when adjacent to the door. In this case, the environment will provide a temperature-based tile code on that timestep. However, the environment will not remember this reading and the following timestep will not include this reading, meaning the agent must rember for itself.

If the agent is cut off from exits by the fire, the episode will continue until the fire overtakes the agent or the episode reaches maximum timesteps. Note that in some timesteps the fire and agent may move at the same time and therefore attempt to occupy the same tile.

People move randomly every 4 timesteps. They do not have directionality and so move in any of the 4 cardinal directions.

Doors have auto-close mechanisms so any open door will close once the agent either turns away from it or moves through it.


## Key environment challenges

- Rewards in the environment are sparse and non-Markovian. 
- Observations are non-Markovian, .
- Observations are partial, only revealing a limited view of the environment.
- People, doors and fire are non-stationary within the environment.
- The moving fire-front may cut the agent off from people and exits.  
- The agent must decide to what extent to risk its own safety to potentially save additional people.
- Configurations are randomly generated for each episode, meaning each episode is different.
- Configurations have variable task sequences since some doors and cupboards are locked while others are not.


## Episode termination conditions

Episodes terminate after max steps, or if the agent moves onto the exit square, or if the agent is on a fire tile.
