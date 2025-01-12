from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

import joblib
import numpy as np
import os

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
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.model = None

    #built-in method
    def act(self, observation, use_random=False):
        """
        Epsilon-greedy action selection using a single model.
        """
        if use_random or self.model is None:
            return env.action_space.sample()

        # Predict Q-values for all actions
        state_batch = np.tile(observation, (self.action_dim, 1))  # Repeat state for each action
        actions = np.arange(self.action_dim).reshape(-1, 1)  # Action indices
        state_action_batch = np.hstack((state_batch, actions))  # Combine state and actions
        q_values = self.model.predict(state_action_batch)

        return np.argmax(q_values)  # Select action with maximum Q-value
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    # built-in method
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def train(self, S, A, R, S2, D, gamma=0.99, iterations=100):
        """
        Train the Q-function model using Fitted Q Iteration.
        """
        nb_samples = S.shape[0]
        self.state_dim = S.shape[1]
        self.action_dim = len(np.unique(A))

        # Append actions to states for (s, a)
        SA = np.hstack((S, A))

        for iteration in range(iterations):
            # Initialize targets
            if self.model is None:
                q_targets = R.copy()
            else:
                # Predict Q-values for next states and all actions
                next_state_batch = np.repeat(S2, self.action_dim, axis=0)
                next_actions = np.tile(np.arange(self.action_dim), len(S2)).reshape(-1, 1)
                next_state_action_batch = np.hstack((next_state_batch, next_actions))
                q_values_next = self.model.predict(next_state_action_batch)
                q_values_next = q_values_next.reshape(len(S2), self.action_dim)

                # Max Q(s', a') for each state
                max_q_values_next = np.max(q_values_next, axis=1)
                q_targets = R + gamma * max_q_values_next * (1 - D)

            # Train the model on (s, a) -> Q(s, a)
            model = HistGradientBoostingRegressor() if self.model is None else self.model
            model.fit(SA, q_targets)
            self.model = model

            print(f"Iteration {iteration + 1}/{iterations} complete. Model updated.")

    #built-in method
    def load(self):
        path="second_trained_fqi_agent.joblib"
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
