import gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO
from delivery_env import DeliveryEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_masks()  # type: ignore


env = DeliveryEnv()
env = ActionMasker(env, mask_fn)

model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, verbose=1)
model.learn(total_timesteps=1000)
model.save("smart-delivery")