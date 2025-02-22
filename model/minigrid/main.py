from typing import Any, Callable, Tuple, Optional
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
import pygame
from tqdm import tqdm
from torchviz import make_dot
import numpy.typing as npt


HIDDEN_SIZE = 256
NUM_ENVS = 10
NUM_ACTIONS = 3  # MiniGrid has 3 actions: left, right, forward
ACTOR_LR = 1e-6
CRITIC_LR = 1e-6
NUM_STEPS = 16000
EVAL_FREQ = 15900
WINDOW_SIZE = 20
RENDER_EVAL = False

env_name = 'MiniGrid-Empty-5x5-v0'


def make_env(render_mode: str = "rgb_array", max_steps: int = 100) -> gym.Env:
    env: gym.Env = gym.make(env_name, render_mode=render_mode, max_episode_steps=max_steps)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env: FlattenObservation = FlattenObservation(env)
    return env


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
        result: torch.Tensor = F.softmax(self.net(x), dim=-1)
        return result


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
        out: torch.Tensor = self.net(x)
        return out


def main() -> None:
    torch.manual_seed(0)

    # Get observation size from environment
    test_env = make_env()
    obs_shape: Optional[Tuple[int, ...]] = test_env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape cannot be None")
    OBS_SIZE = obs_shape[0]
    test_env.close()

    # Initialize pygame for visualization
    pygame.init()

    # Initialize networks and environments
    actor = Actor(OBS_SIZE, HIDDEN_SIZE, NUM_ACTIONS)
    critic = Critic(OBS_SIZE, HIDDEN_SIZE)
    optimizer_actor = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    env = AsyncVectorEnv([lambda: make_env("rgb_array", max_steps=100) for _ in range(NUM_ENVS)])
    eval_env = make_env("human", max_steps=100)  # Use human render mode for evaluation

    obs: torch.Tensor
    obs, _ = env.reset(seed=0)

    # Ensure observations are properly shaped
    assert obs.shape[1] == OBS_SIZE, f"Expected observation size {OBS_SIZE}, got {obs.shape[1]}"
    assert obs.shape == (NUM_ENVS, OBS_SIZE)
    obs: torch.Tensor = torch.tensor(obs, dtype=torch.float32)

    # Performance tracking
    episode_rewards = []  # Remove maxlen to track all episodes
    success_rate = []    # Remove maxlen to track all episodes
    actor_losses = []
    critic_losses = []
    total_steps = 0
    current_rewards = np.zeros(NUM_ENVS)
    episode_steps = np.zeros(NUM_ENVS)

    # NOTE: uncomment for dense reward
    # last_agent_pos = [None for _ in range(NUM_ENVS)]
    for step in tqdm(range(NUM_STEPS)):
        # Actor logic
        probs = actor(obs)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Critic logic
        prev_values_ = critic(obs).squeeze()
        assert prev_values_.shape == (NUM_ENVS,)

        # Environment step
        next_obs: torch.Tensor
        rewards: npt.NDArray[np.float32]
        dones: npt.NDArray[np.bool_]
        truncs: npt.NDArray[np.bool_]
        infos: dict[str, Any]
        next_obs, rewards, dones, truncs, infos = env.step(actions.cpu().numpy())

        # NOTE: dense reward
        # agent_pos = env.get_attr("agent_pos")
        # for i in range(NUM_ENVS):
        #     if 8 in next_obs[i]:
        #         rewards[i] += 0.005
        #     if last_agent_pos[i] is not None:
        #         if agent_pos[i] == last_agent_pos[i]:
        #             rewards[i] -= 0.01
        #     last_agent_pos[i] = agent_pos[i]

        # Update episode rewards and steps
        current_rewards += rewards
        episode_steps += 1

        # Handle episode completion
        for i in range(NUM_ENVS):
            if dones[i] or truncs[i]:  # type: ignore
                episode_rewards.append(current_rewards[i])
                success_rate.append(1.0 if current_rewards[i] > 0.0 else 0.0)
                current_rewards[i] = 0
                episode_steps[i] = 0

        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)

        # Loss calculations
        assert rewards.shape == (NUM_ENVS,)
        assert dones.shape == (NUM_ENVS,)  # type: ignore
        assert actions.shape == (NUM_ENVS,)

        # calc next values
        next_values_ = critic(next_obs).squeeze()
        assert next_values_.shape == (NUM_ENVS,)
        td_error = rewards + (0.99 * next_values_ * (1 - torch.tensor(dones,
                              dtype=torch.float32))) - prev_values_
        # td_error = rewards + (0.99 * prev_values_ * (1 - torch.tensor(dones, dtype=torch.float32))) - prev_values
        assert td_error.shape == (NUM_ENVS,)

        critic_loss = td_error.pow(2).mean()
        actor_loss = (-dist.log_prob(actions) * td_error.detach()).mean()

        # NOTE: actor comp graph
        # dot = make_dot(actor_loss, params=dict(actor.named_parameters()))
        # dot.render('model_graph_actor', format='png')
        # input()

        # NOTE: critic comp graph
        # dot = make_dot(critic_loss, params=dict(critic.named_parameters()))
        # dot.render('model_graph_critic', format='png')
        # input()

        # Update networks
        optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer_critic.step()

        optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer_actor.step()

        # Store losses
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

        # Evaluation
        if (step + 1) % EVAL_FREQ == 0 and RENDER_EVAL:
            print("---------------------------")
            eval_obs, _ = eval_env.reset(seed=0)
            eval_obs = torch.tensor(eval_obs, dtype=torch.float32).unsqueeze(0)
            eval_reward = 0
            eval_steps = 0
            done = False

            count = 0
            while not done:
                count += 1
                if count > 200:
                    break
                eval_env.render()  # This will handle the visualization automatically

                with torch.no_grad():
                    probs = actor(eval_obs)
                    dist = Categorical(probs)
                    # action = torch.argmax(probs).item()

                    # sample action
                    action = dist.sample().item()

                eval_obs, reward, done, trunc, _ = eval_env.step(action)
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32).unsqueeze(0)
                eval_reward += reward  # type: ignore
                eval_steps += 1
                done = done or trunc

                pygame.time.wait(25)  # Add delay to make visualization visible

        # Update observation
        obs = next_obs
        total_steps += NUM_ENVS

    # Plot results
    plt.figure(figsize=(15, 3))

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
    if len(episode_rewards) >= WINDOW_SIZE:
        smoothed_rewards = np.convolve(episode_rewards,
                                       np.ones(WINDOW_SIZE)/WINDOW_SIZE,
                                       mode='valid')
        plt.plot(smoothed_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
