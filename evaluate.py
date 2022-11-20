import gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO
from delivery_env import DeliveryEnv
from sb3_contrib.common.maskable.utils import get_action_masks

from sb3_contrib.common.maskable.evaluation import evaluate_policy


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_masks()  # type: ignore


env = DeliveryEnv()
env = ActionMasker(env, mask_fn)
model = MaskablePPO.load("smart-delivery")
# evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

obs = env.reset()
fitness = 0
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, rewards, dones, info = env.step(action)
    fitness += rewards
    print(obs)

print(fitness)