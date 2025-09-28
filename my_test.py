# %%
from gym_sepsis.envs.sepsis_env import SepsisEnv

env = SepsisEnv()

# %%
import numpy as np

successes = []

for action in range(25):
    results = [0] * 100

    for j in range(100):
        done = False
        env.reset()
        while not done:
            state, reward, done, prob = env.step(action)
        results[j] = reward
    
    successes.append(results.count(15))
# %%

# Action 0: 91 15s, 9 -15s
# Action 1: 91
# Action 2: 96
# Action 3: 88
# Action 12: 94
# Action 24: 93
# np.random.randint(0, 25): 96
# [96, 91, 91, 91, 96, 95, 91, 88, 89, 93, 94, 91, 95, 93, 94, 90, 91, 94, 91, 97, 95, 98, 93, 98, 95]
# %%
