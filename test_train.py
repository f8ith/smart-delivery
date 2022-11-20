from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvMultiDiscrete

env = InvalidActionEnvMultiDiscrete(dims=[4, 6], n_invalid_actions=5)
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(5000)
model.save("maskable_toy_env")