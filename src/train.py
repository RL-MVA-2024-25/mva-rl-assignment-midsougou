from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from sklearn.ensemble import ExtraTreesRegressor
import joblib
import numpy as np

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# Define a simple Neural Network for the Q-value approximation
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.buffer_size= 10000
        self.n_estimators= 50

        # Initialize ensemble decision trees for each action
        self.models = [
            ExtraTreesRegressor(n_estimators=self.n_estimators, random_state=42)
            for _ in range(action_dim)
        ]
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.is_trained = [False] * action_dim  # Track whether models are trained
    #built-in method
    def act(self, observation, use_random=False):
        epsilon = 0.1
        if np.random.rand() < epsilon or not all(self.is_trained):
            return np.random.randint(self.action_dim)

        # Predict Q-values for all actions
        q_values = [model.predict([observation])[0] if self.is_trained[a] else 0
                    for a, model in enumerate(self.models)]
        return np.argmax(q_values)
    
    # built-in method
    def save(self, path):
        for a, model in enumerate(self.models):
            joblib.dump(model, f"{path}_action_{a}.joblib")

    def train(self):
        """Perform Fitted Q-Iteration."""
        if len(self.replay_buffer) < 1000:  # Ensure enough data before training
            return

        # Decompose the replay buffer into separate arrays
        states = np.array([transition[0] for transition in self.replay_buffer])
        actions = np.array([transition[1] for transition in self.replay_buffer])
        rewards = np.array([transition[2] for transition in self.replay_buffer])
        next_states = np.array([transition[3] for transition in self.replay_buffer])
        dones = np.array([transition[4] for transition in self.replay_buffer])

        # Compute the target Q-values
        q_targets = np.zeros(len(states))
        for a in range(self.action_dim):
            if self.is_trained[a]:
                q_next = self.models[a].predict(next_states)
                q_targets = np.maximum(q_targets, q_next)

        targets = rewards + self.gamma * q_targets * ~dones

        # Train separate regressors for each action
        for a in range(self.action_dim):
            mask = actions == a
            if mask.sum() > 0:
                X_train = states[mask]
                y_train = targets[mask]
                self.models[a].fit(X_train, y_train)
                self.is_trained[a] = True
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transitions in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    #built-in method
    def load(self):
        path="trained_fqi_agent"
        for a in range(self.action_dim):
            self.models[a] = joblib.load(f"{path}_action_{a}.joblib")
            self.is_trained[a] = True

