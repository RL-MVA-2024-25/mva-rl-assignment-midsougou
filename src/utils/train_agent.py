from gymnasium.wrappers import TimeLimit
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env_hiv import HIVPatient
from train import ProjectAgent

env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
agent = ProjectAgent()

num_episodes = 500

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(env._max_episode_steps):
        # Select an action
        action = agent.act(state)
        next_state, reward, _, _, _ = env.step(action)
        total_reward += reward

        # Store the transition and train
        agent.store_transition(state, action, reward, next_state, False)

        state = next_state

    # Train the agent after each episode
    agent.train()

    print(f"Episode {episode}, Total Reward: {total_reward}")

# Save the trained models
agent.save("trained_fqi_agent")
print("Agent saved.")
