# %%
# Initialization
from gym_sepsis.envs.sepsis_env import SepsisEnv

env = SepsisEnv()

# %%
# Determine success rate of constant policies, defined as taking the same action
# regardless of state

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

# Successes over 100 trials each of the 25 constant policies (from 0 to 24):
# [96, 91, 91, 91, 96, 95, 91, 88, 89, 93, 94, 91, 95, 93, 94, 90, 91, 94, 91, 97, 95, 98, 93, 98, 95]
# Successes over 100 trials of random treatment (np.random.randint(0, 25)): 96

# %%
# A way to neatly print a state
import pandas as pd

features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
                'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
                'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
                'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
                'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
                'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
                'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
                'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
                'blood_culture_positive']

def state_series(env):
    return pd.Series(env.s.flatten(), index=features)

# %%
# Ran this cell several times to see possible values of binary variables
env.reset()
state_series(env)[['is_male', 'race_white', 'race_black', 'race_hispanic', 'race_other']]

# values for is_male: -1.169 and 0.855
# values for race_white: -1.647 and 0.607
# values for race_black: -0.283 and 3.533
# values for race_hispanic: -0.177 and 5.661
# values for race_other: -0.445 and 2.250
# %%
# Investigate number of values features can take
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
# %%
