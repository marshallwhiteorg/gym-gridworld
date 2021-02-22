import gym
import enum
import numpy as np
from collections import namedtuple

class Action(enum.IntEnum):
    Left = 0
    Right = 1
    Up = 2
    Down = 3

StepResult = namedtuple('StepResult', ['obs', 'reward', 'done', 'info'])

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}
    living_reward = -1
    win_reward = 10
    
    start_state = np.array([0, 0])
    state = start_state
    done = False
    terminal_state = None
    height = -1
    width = -1
    

    def __init__(self, width=4, height=4):
        assert height > 1, width > 1
        self.action_space = gym.spaces.Discrete(len(list(Action)))
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]), high=np.array([width-1, height-1]))
        self.terminal_state = (width-1, height-1)
        self.height = height
        self.width = width

    def step(self, action: int) -> StepResult:
        # Must call reset() after the episode is over
        assert self.done is False, self.action_space.contains(action)
        next_state = self._advance(Action(action))
        next_reward = self._reward(next_state)
        done = np.array_equal(next_state, self.terminal_state)

        self.state = next_state
        self.done = done
        return StepResult(next_state, next_reward, done, {})

    def reset(self):
        self.state = self.start_state
        self.done = False
        return self.state

    def render(self, mode='ansi'):
        grid_repr = ''
        for row in range(self.height):
            if row > 0:
                grid_repr += '\n'
            for col in range(self.width):
                if np.array_equal([col, row], self.state):
                    grid_repr += ' A '
                elif np.array_equal([col, row], self.terminal_state):
                    grid_repr += ' T '
                else:
                    grid_repr += ' o '
        print(grid_repr + '\n')

    def close(self):
        pass

    def _advance(self, action):
        maybe_state = self.state + self._action_mod(action)
        if self.observation_space.contains(maybe_state):
            return maybe_state
        else:
            return self.state

    ''' The (x, y) modification resulting from the action. '''
    def _action_mod(self, action):
        if action == Action.Left:
            mod = [-1, 0]
        elif action == Action.Right:
            mod = [1, 0]
        elif action == Action.Up:
            mod = [0, -1]
        else:
            mod = [0, 1]
        return np.array(mod)

    def _reward(self, next_state):
        if np.array_equal(next_state, self.terminal_state):
            return self.win_reward
        else:
            return self.living_reward