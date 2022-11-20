from delivery_env import DeliveryEnv
from stable_baselines3.common.env_checker import check_env

env = DeliveryEnv()

check_env(env, warn=True)