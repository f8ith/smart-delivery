import gym
from gym import spaces
import numpy as np


class Robot:
    def __init__(self, location):
        self.location = location
        self.max_capacity = 10
        self.books_carried = 0


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = 6  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Discrete(38)

        self.action_space = spaces.MultiDiscrete([4, 4, 7, 7])

    def reset(self, seed=None, options=None):

        # Choose the agent's location uniformly at random
        self.robot_a = Robot(np.array([5, 5], dtype=np.int8))
        self.robot_b = Robot(np.array([0, 0], dtype=np.int8))

        self.delivery_map = np.array(
            [
                [0, [1, 0, 0, 1]],
                [6, [1, 0, 1, 1]],
                [4, [1, 0, 1, 1]],
                [7, [1, 0, 1, 1]],
                [5, [1, 0, 1, 1]],
                [3, [0, 0, 1, 1]],
            ],
            [
                [3, [1, 1, 0, 1]],
                [5, [1, 1, 1, 1]],
                [3, [1, 1, 1, 1]],
                [4, [1, 1, 1, 1]],
                [3, [1, 1, 1, 1]],
                [4, [0, 1, 1, 1]],
            ],
            [
                [4, [1, 1, 0, 1]],
                [2, [1, 1, 1, 1]],
                [0, [1, 1, 1, 1]],
                [2, [1, 1, 1, 0]],
                [8, [1, 1, 1, 1]],
                [2, [0, 1, 1, 1]],
            ],
            [
                [2, [1, 1, 0, 1]],
                [4, [1, 1, 1, 1]],
                [0, [1, 1, 1, 1]],
                [0, [1, 0, 1, 1]],
                [3, [1, 1, 1, 1]],
                [1, [0, 1, 1, 1]],
            ],
            [
                [5, [1, 1, 0, 1]],
                [3, [1, 1, 1, 1]],
                [6, [1, 1, 1, 1]],
                [5, [1, 1, 1, 1]],
                [6, [1, 1, 1, 1]],
                [5, [0, 1, 1, 1]],
            ],
            [
                [6, [1, 1, 0, 0]],
                [6, [1, 1, 1, 0]],
                [2, [1, 1, 1, 0]],
                [6, [1, 1, 1, 0]],
                [2, [1, 1, 1, 0]],
                [0, [0, 1, 1, 0]],
            ],
            dtype=np.int8,
        )

        self.distance_map = np.array(
            [[[0.7, 0, 0, 0.6], []]],
            [[0.6, 0.6, 0, 0.3]],
            [[1.5, 0, 0, 1.1]],
            [[1.7, 0, 0, 0.5], [[2.0, 0, 0, 0.8]]],
            [
                [
                    0.7,
                ]
            ],
            dtype=float,
        )

        observation = self.delivery_map
        info = None

        return observation, info
