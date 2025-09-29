# %%
# Initialization

from gym_sepsis.envs.sepsis_env import SepsisEnv
import numpy as np
import pandas as pd

env = SepsisEnv()

# %%
# This code empirically determines the success rate of the constant policies, 
# defined as taking the same action regardless of state.
# Takes a few minutes to run

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

# Successes over 100 trials each of the 25 constant policies (from 0 to 24):
# [96, 91, 91, 91, 96, 95, 91, 88, 89, 93, 94, 91, 95, 93, 94, 90, 91, 94, 91, 97, 95, 98, 93, 98, 95]
# Successes over 100 trials of random treatment (np.random.randint(0, 25)): 96

# %%
# A way to neatly print a state

def state_series(env):
    features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
                'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
                'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
                'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
                'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
                'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
                'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
                'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
                'blood_culture_positive']
    return pd.Series(env.s.flatten(), index=features)

# %%
# This code investigates the number of values features can take.
features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
                'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
                'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
                'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
                'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
                'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
                'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
                'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
                'blood_culture_positive']

sample = pd.DataFrame(columns=features)

for i in range(100):
    env.reset()
    sample.loc[i] = env.s.flatten()

unique_vals = pd.DataFrame({'feature': features, 'count': np.nan})

unique_vals['count'] = [len(sample[feature].unique()) for feature in unique_vals['feature']]

# The binary variables are:
# is_male: -1.169 and 0.855
# race_white: -1.647 and 0.607
# race_black: -0.283 and 3.533
# race_hispanic: -0.177 and 5.661
# race_other: -0.445 and 2.250
# vent: -1.171 and 0.854
# qsfoa_sysbp_score: -1.852 and 0.540
# qsofa_gcs_score: -0.579 and 1.727
# qsofa_resprate_score: -2.560 and 0.391 
# blood_culture_positive: -0.684 and 1.462

# %%
# This function generates an episode following a given policy. A policy must be a function
# from states to actions (0 to 24). ep_following_pol returns a log of the states, actions,
# and rewards of the episode.

def ep_following_pol(policy):
    features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
                'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
                'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
                'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
                'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
                'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
                'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
                'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
                'blood_culture_positive']
    
    env.reset()
    done = False
    ep_log = pd.DataFrame(columns = features + ['action', 'reward'])
    state_num = 0

    while not done:
        state = env.s.flatten()
        action = policy(state)
        new_state, reward, done, prob = env.step(action)
        ep_log.loc[state_num] = np.append(state, [action, reward])
        state_num += 1
    
    return ep_log


# %%
# Example of use of ep_following_pol with the random policy
def random_policy(state):
    return np.random.randint(0, 25)

ep_following_pol(random_policy)
# %%
