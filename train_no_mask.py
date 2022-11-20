import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from delivery_env import DeliveryEnv

env = make_vec_env(DeliveryEnv, n_envs=4)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("smart-delivery-no-mask")