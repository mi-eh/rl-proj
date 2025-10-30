import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from gym_sepsis.envs.sepsis_env import SepsisEnv

# ----------------------------
# 1. Environment setup
# ----------------------------
env = SepsisEnv()
state_shape = env.reset().shape
n_actions = 25

state_size = np.prod(state_shape)

# ----------------------------
# 2. DQN Agent definition
# ----------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Experience replay memory
        self.memory = deque(maxlen=20000)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        # Q-Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=46, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, state_size))
        targets = np.zeros((self.batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.model.predict(state, verbose=0)[0]
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            states[i] = state
            targets[i] = target

        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ----------------------------
# 3. Training loop
# ----------------------------
agent = DQNAgent(state_size, n_actions)
episodes = 200
target_update_freq = 10

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.replay()

    if e % target_update_freq == 0:
        agent.update_target_model()

    print("Episode: {}/{} | Reward: {:.2f} | Epsilon: {:.2f}".format(
        e+1, episodes, total_reward, agent.epsilon
    ))

env.close()
print("Training complete.")




def evaluate_agent_survival(env, agent, n_episodes=100):
    """Evaluate the agent by the percentage of episodes ending in survival."""
    n_survived = 0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = np.reshape(state, [1, 46])  # match model input format
            action = np.argmax(agent.model.predict(state, verbose=0))
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

        # Positive reward means survival, negative means death
        if total_reward > 0:
            n_survived += 1

    survival_rate = 100.0 * n_survived / n_episodes
    print(f"Survival rate over {n_episodes} evaluation episodes: {survival_rate:.1f}%")
    return survival_rate

evaluate_agent_survival(env, agent, n_episodes=500)


import numpy as np

def evaluate_random_policy(env, n_episodes=100):
    """Evaluate a random policy by the percentage of episodes ending in survival."""
    n_survived = 0
    n_actions = env.action_space.n  # Should be 25

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = np.random.randint(n_actions)  # choose random action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

        # Positive total reward = survival
        if total_reward > 0:
            n_survived += 1

    survival_rate = 100.0 * n_survived / n_episodes
    print(f"Random policy survival rate over {n_episodes} episodes: {survival_rate:.1f}%")
    return survival_rate

evaluate_random_policy(env, n_episodes=500)