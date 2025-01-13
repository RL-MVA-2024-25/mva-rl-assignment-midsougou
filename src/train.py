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
            for a in range(env.action_space.n)  # Dynamically access the action space
        ]
        return np.argmax(q_values)  # Choose the action with the highest Q-value
    
    # built-in method
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def sample(agent, env, num_samples, eps=0.1):
        S, A, R, S2, D = [], [], [], [], []
        cumulative_rewards = []
        current_R = []
        state, _ = env.reset()

        for _ in range(num_samples):
            # Epsilon-greedy action selection
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            next_state, reward, done, trunc, _ = env.step(action)

            # Append to dataset
            S.append(state)
            A.append(action)
            R.append(reward)
            S2.append(next_state)
            D.append(done)

            current_R.append(reward)

            if done or trunc:
                cumulative_rewards.append(np.sum(current_R))
                current_R = []
                state, _ = env.reset()
            else:
                state = next_state

        return (
            np.array(S),
            np.array(A).reshape(-1, 1),
            np.array(R),
            np.array(S2),
            np.array(D),
            np.mean(cumulative_rewards),
        )

    def train(self, states, actions, rewards, next_states, done_flags, discount_factor, iterations=100, num_actions=4):
        """
        Train the Q-function using Fitted Q Iteration logic.
        """
        num_samples = states.shape[0]
        state_action_pairs = np.append(states, actions, axis=1)  # Combine states and actions as input

        current_model = self.model  # Start with the current Q-function, if available

        for iteration in range(iterations):
            # Compute Bellman targets
            if current_model is None:
                targets = rewards.copy()  # If no model exists, use immediate rewards
            else:
                # Compute Q-values for all possible actions in next states
                q_values_next = np.zeros((num_samples, num_actions))
                for action_index in range(num_actions):
                    action_column = np.full((num_samples, 1), action_index)
                    next_state_action_pairs = np.append(next_states, action_column, axis=1)
                    q_values_next[:, action_index] = current_model.predict(next_state_action_pairs)

                # Take the maximum Q-value for each next state
                max_q_values_next = np.max(q_values_next, axis=1)

                # Compute updated targets using the Bellman equation
                targets = rewards + discount_factor * max_q_values_next * (1 - done_flags)

            # Train the Q-function approximator with updated targets
            regressor = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1)
            regressor.fit(state_action_pairs, targets)
            current_model = regressor  # Update the Q-function

        # Save the trained model
        self.model = current_model

    #built-in method
    def load(self):
        path="final_fqi_model.joblib"
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
