import gym
from gym import spaces
import numpy as np

a = spaces.MultiDiscrete([6, 6])

b = np.array([5, 5], dtype=np.int8)

print(a.contains(b))
