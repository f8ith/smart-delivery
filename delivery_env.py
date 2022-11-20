import gym
from gym import spaces
import numpy as np


class Robot:
    def __init__(self, pos, speed):
        self.pos = pos
        self.speed = speed
        self.max_capacity = 10
        self.books_carried = 0
        self.ahead_time = 0


class DeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = 6  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Dict(
            {
                "books_remaining": spaces.MultiDiscrete(
                    [
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        128,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                        9,
                    ]
                ),
                "a_pos": spaces.MultiDiscrete([6, 6]),
                "b_pos": spaces.MultiDiscrete([6, 6]),
            }
        )

        self.action_space = spaces.MultiDiscrete([4, 4, 11, 11])

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self.action_mask_map = np.array(
            [
                [
                    [True, False, False, True],
                    [True, False, True, True],
                    [True, False, True, True],
                    [True, False, True, True],
                    [True, False, True, True],
                    [False, False, True, True],
                ],
                [
                    [True, True, False, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, True, True, True],
                ],
                [
                    [True, True, False, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, False],
                    [True, True, True, True],
                    [False, True, True, True],
                ],
                [
                    [True, True, False, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, False, True, True],
                    [True, True, True, True],
                    [False, True, True, True],
                ],
                [
                    [True, True, False, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, True, True, True],
                ],
                [
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [False, True, True, False],
                ],
            ]
        )

        self.distance_map = np.array(
            [
                [
                    [0.7, 0, 0, 0.6],
                    [0.7, 0, 0.7, 0.5],
                    [0.5, 0, 0.7, 0.4],
                    [1.1, 0, 0.5, 0.9],
                    [0.5, 0, 1.1, 0.3],
                    [0, 0, 0.5, 0.4],
                ],
                [
                    [0.6, 0.6, 0, 0.3],
                    [1.1, 0.5, 0.6, 1.2],
                    [0.7, 0.4, 1.1, 1.1],
                    [0.6, 0.9, 0.7, 0.4],
                    [0.9, 0.3, 0.6, 1.1],
                    [0, 0.4, 0.9, 0.9],
                ],
                [
                    [1.5, 0.3, 0, 1.1],
                    [1.3, 1.2, 1.5, 1.8],
                    [1.0, 1.1, 1.3, 2.2],
                    [0.3, 0.4, 1.0, 0],
                    [0.8, 1.1, 0.3, 2.1],
                    [0, 0.9, 0.8, 1.4],
                ],
                [
                    [1.7, 1.1, 0, 0.5],
                    [0.7, 1.7, 1.8, 0.8],
                    [0.4, 2.2, 0.7, 0.7],
                    [0.8, 0, 0.4, 0.5],
                    [0.7, 2.1, 0.8, 0.8],
                    [0, 1.4, 0.7, 0.6],
                ],
                [
                    [2.0, 0.5, 0, 0.8],
                    [1.0, 0.8, 2.0, 1.0],
                    [0.5, 0.7, 1.0, 1.0],
                    [0.6, 0.5, 0.5, 0.6],
                    [0.5, 0.8, 0.6, 1.1],
                    [0, 0.6, 0.5, 0.5],
                ],
                [
                    [0.7, 0.8, 0, 0],
                    [0.6, 1.0, 0.7, 0],
                    [0.6, 1.0, 0.6, 0],
                    [0.4, 0.6, 0.6, 0],
                    [0.3, 1.1, 0.4, 0],
                    [0, 0.5, 0.3, 0],
                ],
            ],
            dtype=float,
        )

    def _get_obs(self):
        return {
            "books_remaining": self.delivery_map.flatten(),
            "a_pos": self.robot_a.pos,
            "b_pos": self.robot_b.pos,
        }

    def reset(self, seed=None, options=None):

        # Choose the agent's location uniformly at random
        self.robot_a = Robot(np.array([5, 5], dtype=np.int8), 8)
        self.robot_b = Robot(np.array([0, 0], dtype=np.int8), 10)

        self.delivery_map = np.array(
            [
                [
                    0,
                    6,
                    4,
                    7,
                    5,
                    3,
                ],
                [
                    3,
                    5,
                    3,
                    4,
                    3,
                    4,
                ],
                [
                    4,
                    2,
                    0,
                    2,
                    8,
                    2,
                ],
                [
                    2,
                    4,
                    127,
                    0,
                    3,
                    1,
                ],
                [
                    5,
                    3,
                    6,
                    5,
                    6,
                    5,
                ],
                [
                    6,
                    6,
                    2,
                    6,
                    2,
                    0,
                ],
            ],
            dtype=np.int8,
        )

        observation = self._get_obs()

        return observation

    def step(self, action: np.ndarray):
        a = self.robot_a
        b = self.robot_b

        if np.array_equal(a.pos, [3, 4]):
            a.books_carried = max(0, a.books_carried - action[2])
        else:
            a.books_carried = max(10, a.books_carried + action[2])

        if np.array_equal(b.pos, [3, 4]):
            b.books_carried = max(0, b.books_carried - action[2])
        else:
            b.books_carried = max(10, b.books_carried + action[2])

        self.delivery_map[a.pos[1], a.pos[0]] = max(
            0, self.delivery_map[a.pos[1], a.pos[0]] - action[2]
        )

        self.delivery_map[b.pos[1], b.pos[0]] = max(
            0, self.delivery_map[b.pos[1], b.pos[0]] - action[2]
        )

        terminated = True if self.delivery_map[3, 2] == 0 else False
        a_time_elapsed = (
            a.speed / self.distance_map[a.pos[1], a.pos[0]][action[0]] + a.ahead_time
        )
        b_time_elapsed = (
            b.speed / self.distance_map[b.pos[1], b.pos[0]][action[1]] + b.ahead_time
        )

        time_difference = a_time_elapsed - b_time_elapsed

        if time_difference < 0:
            a.ahead_time = time_difference
        else:
            b.ahead_time = time_difference

        a.pos = np.clip(a.pos + self._action_to_direction[action[0]], 0, self.size - 1)
        b.pos = np.clip(b.pos + self._action_to_direction[action[1]], 0, self.size - 1)

        observation = self._get_obs()
        reward = -max(a_time_elapsed, b_time_elapsed) + action[2] + action[3]

        info = {}

        return observation, reward, terminated, info

    def action_masks(self):
        a = self.robot_a
        b = self.robot_b

        a_action_mask = np.zeros(11, dtype=bool)
        b_action_mask = np.zeros(11, dtype=bool)

        if np.array_equal(a.pos, [3, 4]):
            a_action_mask[a.books_carried] = True
        else:
            max_return = min(
                (a.max_capacity - a.books_carried),
                self.delivery_map[a.pos[1], a.pos[0]],
            )
            a_action_mask[0:max_return] = True

        if np.array_equal(b.pos, [3, 4]):
            b_action_mask[b.books_carried] = True
        else:
            max_return = min(
                (b.max_capacity - b.books_carried),
                self.delivery_map[b.pos[1], b.pos[0]],
            )
            b_action_mask[0:max_return] = True

        return np.concatenate(
            (
                self.action_mask_map[a.pos[1], a.pos[0]],
                self.action_mask_map[b.pos[1], b.pos[0]],
                a_action_mask,
                b_action_mask,
            )
        )
