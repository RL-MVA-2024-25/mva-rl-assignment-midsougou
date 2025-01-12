from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

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
        self.lr = 1e-3
        self.batch_size = 64
        self.replay_buffer = deque(maxlen=10000)

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
    #built-in method
    def act(self, observation, use_random=False):
        epsilon = 0.1
        if use_random or random.random() < epsilon:
            return env.action_space.sample()
        observation = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(observation)
        return torch.argmax(q_values).item()
    
    # built-in method
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    #built-in method
    def load(self):
        path="trained_hiv_agent.pth"
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
