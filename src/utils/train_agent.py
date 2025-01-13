from gymnasium.wrappers import TimeLimit
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env_hiv import HIVPatient
from train import ProjectAgent
import numpy as np

env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
agent = ProjectAgent()

# Experiment hyperparameters
num_stages = 20
num_episodes_per_stage = 20
randomization_stages = [3, 5, 7, 11, 13, 15]
episode_steps = 200  # Maximum steps per episode
training_iterations = 100
discount_factor = 0.98
total_samples_per_stage = num_episodes_per_stage * episode_steps


# Stage 0: Initial sampling
print("Starting initial sample collection")
state_buffer, action_buffer, reward_buffer, next_state_buffer, terminal_buffer, avg_cumulative_reward = agent.sample(
    env, total_samples_per_stage, eps=0.1
)

initial_dataset = [state_buffer, action_buffer, reward_buffer, next_state_buffer, terminal_buffer]
print(f"Stage: 0 \t Average Reward: {avg_cumulative_reward:.2e}")

# Train the initial Q-function
agent.train(
    state_buffer,
    action_buffer,
    reward_buffer,
    next_state_buffer,
    terminal_buffer,
    discount_factor,
    iterations=training_iterations,
    num_actions=4)

# Iterative training process
for stage_index in range(1, num_stages):
    print(f"Processing Stage {stage_index}/{num_stages}")

    # Enable domain randomization for specific stages
    if stage_index in randomization_stages:
        env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=episode_steps)
    else:
        env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=episode_steps)

    # Collect samples using the current Q-function
    new_states, new_actions, new_rewards, new_next_states, new_terminals, new_avg_reward = agent.sample(
        env, total_samples_per_stage, eps=0.1
    )
    print(f"Stage: {stage_index} \t Average Reward: {new_avg_reward:.2e}")

    # Append new samples to the dataset
    state_buffer = np.vstack([state_buffer, new_states])
    action_buffer = np.vstack([action_buffer, new_actions])
    reward_buffer = np.hstack([reward_buffer, new_rewards])
    next_state_buffer = np.vstack([next_state_buffer, new_next_states])
    terminal_buffer = np.hstack([terminal_buffer, new_terminals])

    # Save the updated dataset
    updated_dataset = [state_buffer, action_buffer, reward_buffer, next_state_buffer, terminal_buffer]

    # Retrain the Q-function using the entire dataset
    agent.train(
        state_buffer,
        action_buffer,
        reward_buffer,
        next_state_buffer,
        terminal_buffer,
        discount_factor,
        iterations=training_iterations,
        num_actions=4,
    )

# Save the trained models
agent.save("final_fqi_model.joblib")
print("Agent saved.")
