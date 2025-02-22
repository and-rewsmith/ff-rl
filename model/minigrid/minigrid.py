from typing import Any, Callable
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import FlattenObservation
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchviz import make_dot
torch.manual_seed(0)

# Parameters
NUM_ENVS = 4
HIDDEN_SIZE = 192
NUM_ACTIONS = 3  # MiniGrid has 3 actions: left, right, forward
ACTOR_LR = 0.001
CRITIC_LR = 0.001
NUM_STEPS = 1000
EVAL_FREQ = 100
WINDOW_SIZE = 100
RENDER_EVAL = True

env_name = 'MiniGrid-Empty-8x8-v0'

def make_env() -> gym.Env:
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)
    return env

# Get observation size from environment
test_env = make_env()
OBS_SIZE = test_env.observation_space.shape[0]
test_env.close()

# Networks
class Actor(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def main() -> None:
    # Initialize networks with correct dimensions
    actor = Actor(OBS_SIZE, HIDDEN_SIZE, NUM_ACTIONS)
    critic = Critic(OBS_SIZE, HIDDEN_SIZE)
    optimizer_actor = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    env = AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    obs, _ = env.reset()
    
    # Ensure observations are properly shaped
    assert obs.shape[1] == OBS_SIZE, f"Expected observation size {OBS_SIZE}, got {obs.shape[1]}"
    obs = torch.tensor(obs, dtype=torch.float32)

    # Performance tracking
    episode_rewards = deque(maxlen=WINDOW_SIZE)
    episode_lengths = deque(maxlen=WINDOW_SIZE)
    success_rate = deque(maxlen=WINDOW_SIZE)  # Track task completion rate
    actor_losses = []
    critic_losses = []
    total_steps = 0
    current_rewards = np.zeros(NUM_ENVS)
    episode_steps = np.zeros(NUM_ENVS)

    # Create separate eval env
    eval_env = make_env()
    
    for step in range(NUM_STEPS):
        # Actor logic
        probs = actor(obs)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Critic logic
        values = critic(obs).squeeze()

        # Environment step
        next_obs, rewards, dones, truncs, infos = env.step(actions.cpu().numpy())
        
        # Update episode rewards and steps
        current_rewards += rewards
        episode_steps += 1
        
        # Handle episode completion
        for i in range(NUM_ENVS):
            if dones[i] or truncs[i]:
                episode_rewards.append(current_rewards[i])
                episode_lengths.append(episode_steps[i])
                # In MiniGrid, reward of 1.0 means task completion
                success_rate.append(1.0 if current_rewards[i] > 0.0 else 0.0)
                current_rewards[i] = 0
                episode_steps[i] = 0
                print(f"Env {i} completed episode: {'Success' if rewards[i] > 0.0 else 'Failed'} "
                      f"after {episode_steps[i]:.0f} steps")

        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)

        # Loss calculations
        critic_loss = F.mse_loss(values, rewards)
        advantages = rewards - values.detach()
        actor_loss = -(log_probs * advantages).mean()

        # Update networks
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        # Store losses
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

        # Evaluation
        if step % EVAL_FREQ == 0:
            eval_obs, _ = eval_env.reset()
            eval_obs = torch.tensor(eval_obs, dtype=torch.float32).unsqueeze(0)
            eval_reward = 0
            eval_steps = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    probs = actor(eval_obs)
                    action = torch.argmax(probs).item()
                
                eval_obs, reward, done, trunc, _ = eval_env.step(action)
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32).unsqueeze(0)
                eval_reward += reward
                eval_steps += 1
                done = done or trunc

            success = eval_reward > 0.0
            print(f"\nEval Step {step}")
            print(f"Task {'Completed' if success else 'Failed'} in {eval_steps} steps")
            print(f"Eval Reward: {eval_reward:.2f}")
            print(f"Training Stats:")
            print(f"- Avg Reward: {np.mean(list(episode_rewards)):.2f}")
            print(f"- Avg Episode Length: {np.mean(list(episode_lengths)):.1f}")
            print(f"- Success Rate: {np.mean(list(success_rate)):.1%}")

        # Update observation
        obs = next_obs
        total_steps += NUM_ENVS

    # Plot results
    plt.figure(figsize=(15, 4))
    
    plt.subplot(141)
    plt.plot(actor_losses)
    plt.title("Actor Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    
    plt.subplot(142)
    plt.plot(critic_losses)
    plt.title("Critic Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    
    plt.subplot(143)
    plt.plot(list(episode_rewards))
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    plt.subplot(144)
    plt.plot(list(success_rate))
    plt.title("Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    
    plt.tight_layout()
    plt.show()

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()