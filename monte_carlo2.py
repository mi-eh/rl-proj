from gym_sepsis.envs.sepsis_env import SepsisEnv
import numpy as np
import pandas as pd

env = SepsisEnv()

# %%
# As in simple_monte_carlo, but choosing just the three qSOFA subscores and making them
# each binary.

def reduce_state(state):
    features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
                'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
                'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
                'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
                'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
                'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
                'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
                'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
                'blood_culture_positive']
    
    selected_features = ['qsofa_sysbp_score', 'qsofa_gcs_score', 'qsofa_resprate_score']
    indices = [features.index(f) for f in selected_features]
    subset = [state.flatten()[i] for i in indices]

    def my_round(x):
        if x >= 0:
            return 1
        return 0


    reduced_state = [my_round(f) for f in subset]

    return reduced_state

# %%
# The policy is determined by Q, the table of state-action values. In this
# state space, Q's dimensions are 8 x 25. The action chosen for a state
# is the argmax of the Q-values of that state, with ties broken randomly.

def policy(Q, state):
    # ravel_multi_index maps the state vectors to the integers from 0 to 7
    row = Q[np.ravel_multi_index(state, dims=(2,)*3), :]
    max_indices = np.flatnonzero(row == row.max())
    return np.random.choice(max_indices)

# This function generates an episode where the first action is given and then the
# policy determined by Q is followed after that. 

def ep_gen_red_state(Q, first_action):
    selected_features = ['qsofa_sysbp_score', 'qsofa_gcs_score', 'qsofa_resprate_score']
    
    env.reset()
    ep_log = pd.DataFrame(columns = selected_features + ['action', 'reward'])

    red_state = reduce_state(env.s)
    new_state, reward, done, prob = env.step(first_action)
    ep_log.loc[0] = np.append(red_state, [first_action, reward])
    
    state_num = 1
    while not done:
        red_state = reduce_state(env.s)
        action = policy(Q, red_state)
        new_state, reward, done, prob = env.step(action)
        ep_log.loc[state_num] = np.append(red_state, [action, reward])
        state_num += 1
    
    return ep_log

# %%
# Example
Q = np.zeros((8, 25))
ep_gen_red_state(Q, 2)

# %%
# This function is used to update Q (which is a mean) using its old value, 
# the new value, and the number of observations over which it is a mean.

def update_mean(new_value, old_mean, old_n):
    return old_mean + (1 / (old_n + 1)) * (new_value - old_mean)

# %%
# This is the Monte Carlo Exploring Starts implementation, using 1000 episodes.

Q = np.zeros((8, 25))
N = np.full((8, 25), 0)

# %%

for _ in range(1000):
    env.reset()
    first_action = np.random.choice(range(25))
    ep = ep_gen_red_state(Q, first_action)
    reward = ep['reward'][len(ep) - 1]

    sas_so_far = []
    for i in range(len(ep)):
        curr_sa = tuple(ep.iloc[i, 0:4])
        if curr_sa not in sas_so_far:
            sas_so_far.append(curr_sa)
            multi_index = np.ravel_multi_index(curr_sa[0:3], dims=(2,)*3)
            q_value = Q[multi_index, curr_sa[3]]
            old_n = N[multi_index, curr_sa[3]]
            Q[multi_index, curr_sa[3]] = update_mean(reward, q_value, old_n)
            N[multi_index, curr_sa[3]] += 1


# %%
# This code evaluates the performance of the policy corresponding to the Q we
# have achieved with MCES.

n = 1000
results = [0] * n

for i in range(n):
    done = False
    env.reset()
    state = env.s
    while not done:
        reduced_state = reduce_state(state)
        state, reward, done, prob = env.step(policy(Q, reduced_state))
    results[i] = reward

results.count(15)

# Successes with 3 features: 952 / 1000

# %%
# This code evaluates the performance of the random policy, which corresponds
# to a Q matrix of all 0s (thus there is always a 25-way tie).

Q = np.zeros((8, 25))

n = 100
results = [0] * n

for i in range(n):
    done = False
    env.reset()
    state = env.s
    while not done:
        reduced_state = reduce_state(state)
        state, reward, done, prob = env.step(policy(Q, reduced_state))
    results[i] = reward

results.count(15)
