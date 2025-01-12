from gymnasium.wrappers import TimeLimit
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env_hiv import HIVPatient
from train import ProjectAgent
import numpy as np

env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
agent = ProjectAgent()

# Experiment parameters
n_patients = 20
n_stages = 20
total_samples = n_patients * 200
gamma = 0.98
fqi_iterations = 100
eps = 0.1  # Epsilon for exploration
shuffle_stages = [3, 5, 7, 11, 13, 15]  # Stages for domain randomization

# Initialize dataset
S, A, R, S2, D = [], [], [], [], []

# training loop
for stage in range(n_stages):
    print(f"Stage {stage + 1}/{n_stages}")

    # Enable domain randomization for certain stages
    if stage in shuffle_stages:
        env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
    else:
        env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

    # Collect samples
    state, _ = env.reset()
    cumulative_rewards = []
    for _ in range(total_samples // n_stages):
        # Epsilon-greedy action selection
        if agent.model is None or np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            action = agent.act(state)

        # Step in the environment
        next_state, reward, done, trunc, _ = env.step(action)

        # Store the transition
        S.append(state)
        A.append(action)
        R.append(reward)
        S2.append(next_state)
        D.append(done)
        cumulative_rewards.append(reward)

        # Reset environment if the episode is done or truncated
        if done or trunc:
            state, _ = env.reset()
        else:
            state = next_state

    # Convert data to NumPy arrays for training
    S_np = np.array(S)
    A_np = np.array(A).reshape(-1, 1)
    R_np = np.array(R)
    S2_np = np.array(S2)
    D_np = np.array(D)

    # Train the agent with the collected samples
    agent.train(S_np, A_np, R_np, S2_np, D_np, gamma=gamma, iterations=fqi_iterations)

    # log progress
    avg_reward = np.mean(cumulative_rewards)
    print(f"Stage {stage + 1}/{n_stages} complete. Avg Reward: {avg_reward:.2f}")

# Save the trained models
agent.save("final_fqi_model.joblib")
print("Agent saved.")
