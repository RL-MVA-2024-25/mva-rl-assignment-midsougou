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
        if use_random or self.model is None:
            return env.action_space.sample()

        # Predict Q-values for all actions
        q_values = [
            self.model.predict(np.append(observation, a).reshape(1, -1))[0]
            for a in range(self.action_dim)
        ]
        return np.argmax(q_values)  # Choose the action with the highest Q-value
    
    # built-in method
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def train(self, S, A, R, S2, D, gamma=0.98, iterations=100):
        """
        Train the Q-function model using Fitted Q Iteration.
        """
        nb_samples = S.shape[0]
        self.state_dim = S.shape[1]
        self.action_dim = len(np.unique(A))

        # Combine states and actions for training
        SA = np.hstack((S, A))

        for iteration in range(iterations):
            # Initialize target values
            if self.model is None:
                q_targets = R.copy()
            else:
                # Predict Q(s', a') for all actions and find max Q(s', a')
                q_values_next = np.zeros((nb_samples, self.action_dim))
                for a in range(self.action_dim):
                    next_sa = np.hstack((S2, np.full((nb_samples, 1), a)))
                    q_values_next[:, a] = self.model.predict(next_sa)
                max_q_values_next = np.max(q_values_next, axis=1)
                q_targets = R + gamma * max_q_values_next * (1 - D)

            # Train the model on (s, a) -> Q(s, a)
            model = HistGradientBoostingRegressor() if self.model is None else self.model
            model.fit(SA, q_targets)
            self.model = model

            print(f"Iteration {iteration + 1}/{iterations} complete. Model updated.")

    #built-in method
    def load(self):
        path="final_fqi_model.joblib"
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
