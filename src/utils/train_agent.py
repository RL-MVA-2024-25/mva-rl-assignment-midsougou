from gymnasium.wrappers import TimeLimit
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env_hiv import HIVPatient
from train import ProjectAgent

env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

# Training parameters
num_episodes = 500
update_target_every = 10

agent = ProjectAgent()

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(env._max_episode_steps):
        # Take an action
        action = agent.act(state, use_random=True)
        next_state, reward, _, _, _ = env.step(action)  # Correct unpacking
        total_reward += reward

        # Store the transition in the replay buffer
        agent.replay_buffer.append((state, action, reward, next_state, False))
        agent.train_step()

        state = next_state

    
    if episode % update_target_every == 0:
        agent.update_target_network()

    print(f"Episode {episode}, Total Reward: {total_reward}")

save_path = "trained_hiv_agent.pth"
agent.save(save_path)
print(f"Agent saved to {save_path}")
