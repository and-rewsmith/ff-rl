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
import wandb

from model.ff.ff import FFReservoir

# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    raise RuntimeError("No GPU (CUDA or MPS) available")

NUM_ACTIONS = 3  # MiniGrid has 3 actions: left, right, forward
NUM_ENVS = 10
NUM_STEPS = 15000
EVAL_FREQ = 14000
RENDER_EVAL = False
WINDOW_SIZE = 50

HIDDEN_SIZE = 128
ACTOR_LR = 1e-6
CRITIC_LR = 2e-6

RESERVOIR_SIZE = 1000
STATE_LR = 1e-4
STATE_LOSS_THRESHOLD = 1.5

env_name = 'MiniGrid-Empty-5x5-v0'


def make_env(render_mode: str | None = "rgb_array", max_steps: int = 100) -> gym.Env:
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


def one_hot_encode_observation(obs: np.ndarray) -> torch.Tensor:
    """
    Convert raw observation to one-hot encoded tensor.
    Each 3 elements in obs represent (object, color, state).
    object: one-hot size 11
    color: one-hot size 6
    state: one-hot size 3
    """
    assert len(obs) % 3 == 0, f"Observation length {len(obs)} not divisible by 3"

    num_tuples = len(obs) // 3
    encoded = []

    for i in range(num_tuples):
        obj = obs[i * 3]
        color = obs[i * 3 + 1]
        state = obs[i * 3 + 2]

        obj_onehot = F.one_hot(torch.tensor(obj, dtype=torch.int64), num_classes=11)
        color_onehot = F.one_hot(torch.tensor(color, dtype=torch.int64), num_classes=6)
        # bug in docs where wrong class cardinality is documented
        state_onehot = F.one_hot(torch.tensor(state, dtype=torch.int64), num_classes=4)

        encoded.extend([obj_onehot, color_onehot, state_onehot])

    return torch.cat(encoded).float()

def init_wandb() -> None:
    wandb.init(
        project="ff-rl",
        config={
            "architecture": "Initial",
            "dataset": "Minigrid-untagged",
            "num_envs": NUM_ENVS,
            "num_steps": NUM_STEPS,
            "eval_freq": EVAL_FREQ,
            "render_eval": RENDER_EVAL,
            "hidden_size": HIDDEN_SIZE,
            "actor_lr": ACTOR_LR,
            "critic_lr": CRITIC_LR,
            "reservoir_size": RESERVOIR_SIZE,
            "state_lr": STATE_LR,
            "state_loss_threshold": STATE_LOSS_THRESHOLD,
            "env_name": env_name,
            "window_size": WINDOW_SIZE,
            "device": DEVICE,
            "num_actions": NUM_ACTIONS,
        }
    )



def main() -> None:
    print("USING GPU:", DEVICE)
    torch.manual_seed(0)

    # Initialize wandb
    init_wandb()

    # Get observation size from environment
    test_env = make_env()
    obs_shape: Optional[Tuple[int, ...]] = test_env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape cannot be None")
    RAW_OBS_SIZE = obs_shape[0]
    OBS_SIZE = (RAW_OBS_SIZE // 3) * (11 + 6 + 4)
    test_env.close()

    # Initialize pygame for visualization
    pygame.init()

    env = None
    eval_env = None
    try:
        # Initialize networks and environments
        actor = Actor(RESERVOIR_SIZE, HIDDEN_SIZE, NUM_ACTIONS).to(DEVICE)
        critic = Critic(RESERVOIR_SIZE, HIDDEN_SIZE).to(DEVICE)
        state = FFReservoir(reservoir_size=RESERVOIR_SIZE, batch_size=NUM_ENVS, input_dim=OBS_SIZE+3+1,
                            learning_rate=STATE_LR, loss_threshold=STATE_LOSS_THRESHOLD, device=DEVICE).to(DEVICE)
        optimizer_actor = optim.Adam(actor.parameters(), lr=ACTOR_LR)
        optimizer_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR)
        optimizer_state = optim.Adam(state.parameters(), lr=STATE_LR)

        env = AsyncVectorEnv([lambda: make_env(None, max_steps=100) for _ in range(NUM_ENVS)])
        eval_env = make_env("human", max_steps=100)

        obs_array: np.ndarray
        obs_array, _ = env.reset(seed=0)

        # Process observations
        obs_array = np.stack([one_hot_encode_observation(o) for o in obs_array])

        # Performance tracking
        episode_rewards = []
        actor_losses = []
        critic_losses = []
        total_steps = 0
        current_rewards = np.zeros(NUM_ENVS)
        episode_steps = np.zeros(NUM_ENVS)

        # NOTE: uncomment for dense reward
        last_agent_pos = [None for _ in range(NUM_ENVS)]
        for step in tqdm(range(NUM_STEPS)):
            # NOTE: lower learning rate by 10x (dynamic)
            # if step == 5000:
            #     optimizer_actor.param_groups[0]['lr'] = optimizer_actor.param_groups[0]['lr'] * 0.1
            #     optimizer_critic.param_groups[0]['lr'] = optimizer_critic.param_groups[0]['lr'] * 0.1

            # Actor logic
            probs = actor(state.activations.detach())
            dist = Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            obs = torch.tensor(obs_array, dtype=torch.float32).to(DEVICE)  # type: ignore

            # pass through ff
            if step != 0:
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE)  # Add dimension
            else:
                rewards = torch.zeros(NUM_ENVS, 1, device=DEVICE)

            # handle pos
            actions_onehot = F.one_hot(actions, num_classes=NUM_ACTIONS)
            sensory_input_pos = torch.cat((obs, actions_onehot, rewards), dim=1)

            # handle neg
            negative_actions = torch.randint(0, NUM_ACTIONS, (NUM_ENVS,), device=DEVICE)
            while (negative_actions == actions).any():
                negative_actions = torch.randint(0, NUM_ACTIONS, (NUM_ENVS,), device=DEVICE)
            negative_actions_onehot = F.one_hot(negative_actions, num_classes=NUM_ACTIONS)
            sensory_input_neg = torch.cat((obs, negative_actions_onehot, rewards), dim=1)

            state.process_timestep(sensory_input_pos, sensory_input_neg, training=True)
            state_activations_prev = state.activations.detach().clone()

            # add to existing rewards a state reward that is magnitude of activations in EBM
            from model.ff.ff import layer_activations_to_badness
            state_rewards = torch.zeros_like(layer_activations_to_badness(state_activations_prev).detach().unsqueeze(1))
            assert state_rewards.shape == (NUM_ENVS, 1)
            assert rewards.shape == (NUM_ENVS, 1)
            rewards += state_rewards

            # make this append the taken action
            # append the action to the obs -- for neg action, sample random action but it cannot be same as pos action
            # pos_obs = torch.cat((obs, actions.unsqueeze(1)), dim=1)
            # neg_action = torch.randint(0, NUM_ACTIONS, (NUM_ENVS, 1)).to(DEVICE)
            # while (neg_action == actions.unsqueeze(1)).any():
            #     neg_action = torch.randint(0, NUM_ACTIONS, (NUM_ENVS, 1)).to(DEVICE)
            # assert neg_action.shape == (NUM_ENVS, 1)
            # neg_obs = torch.cat((obs, neg_action), dim=1)
            # state.process_timestep(pos_obs, neg_obs, training=True)

            # Critic logic
            prev_values_ = critic(state_activations_prev).squeeze()
            assert prev_values_.shape == (NUM_ENVS,)

            # Environment step
            next_obs_array: np.ndarray
            rewards: np.ndarray
            dones: np.ndarray
            truncs: np.ndarray
            infos: dict
            next_obs_array, rewards, dones, truncs, infos = env.step(actions.cpu().numpy())

            if True:
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE)  # Add dimension
                # log the raw reward (take into account the batch dimension)
                wandb.log({
                    "raw_reward": rewards.squeeze().mean().item()
                })

                # handle pos
                actions_onehot = F.one_hot(actions, num_classes=NUM_ACTIONS)
                sensory_input_pos = torch.cat((obs, actions_onehot, rewards), dim=1)

                # handle neg
                negative_actions = torch.randint(0, NUM_ACTIONS, (NUM_ENVS,), device=DEVICE)
                while (negative_actions == actions).any():
                    negative_actions = torch.randint(0, NUM_ACTIONS, (NUM_ENVS,), device=DEVICE)
                negative_actions_onehot = F.one_hot(negative_actions, num_classes=NUM_ACTIONS)
                sensory_input_neg = torch.cat((obs, negative_actions_onehot, rewards), dim=1)

                state.process_timestep(sensory_input_pos, sensory_input_neg, training=True)
                state_activations_prev = state.activations.detach().clone()

                # add to existing rewards a state reward that is magnitude of activations in EBM
                from model.ff.ff import layer_activations_to_badness
                state_rewards = torch.zeros_like(layer_activations_to_badness(state_activations_prev).detach().unsqueeze(1))
                assert state_rewards.shape == (NUM_ENVS, 1)
                assert rewards.shape == (NUM_ENVS, 1)
                rewards += state_rewards
                rewards = rewards.squeeze()

                state_activations_next = state.activations.detach().clone()


            # NOTE: dense reward
            # agent_pos = env.get_attr("agent_pos")
            # for i in range(NUM_ENVS):
            #     if 8 in next_obs[i]:
            #         rewards[i] += 0.005
            #     if last_agent_pos[i] is not None:
            #         if agent_pos[i] == last_agent_pos[i]:
            #             rewards[i] -= 0.01
            #     last_agent_pos[i] = agent_pos[i]


            # Process the new observations
            next_obs_array = np.stack([one_hot_encode_observation(o) for o in next_obs_array])
            next_obs = torch.tensor(next_obs_array, dtype=torch.float32).to(DEVICE)  # type: ignore

            # Update episode rewards and steps
            current_rewards += rewards.squeeze().cpu().numpy()
            episode_steps += 1

            # Handle episode completion
            for i in range(NUM_ENVS):
                if dones[i] or truncs[i]:  # type: ignore
                    episode_rewards.append(current_rewards[i])
                    current_rewards[i] = 0
                    episode_steps[i] = 0
                    state.reset_activations_for_batches(torch.tensor([i], dtype=torch.int))

            rewards: torch.Tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)  # type: ignore

            # Loss calculations
            assert rewards.shape == (NUM_ENVS,)
            assert dones.shape == (NUM_ENVS,)  # type: ignore
            assert actions.shape == (NUM_ENVS,)

            # calc next values
            next_values_ = critic(state_activations_next).squeeze()
            assert next_values_.shape == (NUM_ENVS,)
            td_error = rewards + (0.99 * next_values_ * (1 - torch.tensor(dones,
                                  dtype=torch.float32).to(DEVICE))) - prev_values_
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

            # Log to wandb
            wandb.log({
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "ff_loss": state.last_loss,  # Assuming FFReservoir stores its last loss
                "episode_reward": np.mean(episode_rewards[-NUM_ENVS:]) if episode_rewards else 0,
                "step": step
            })

            # Store losses
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            # Evaluation
            if (step + 1) % EVAL_FREQ == 0 and RENDER_EVAL:
                input("Ready for eval?")
                eval_obs, _ = eval_env.reset(seed=0)
                # Process eval observation
                eval_obs = one_hot_encode_observation(eval_obs)
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
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

                        # sample action
                        action = dist.sample().item()

                    eval_obs, reward, done, trunc, _ = eval_env.step(action)
                    # Process eval observation
                    eval_obs = one_hot_encode_observation(eval_obs)
                    eval_obs = torch.tensor(eval_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
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
        plt.savefig("results.png")

    finally:
        # Ensure proper cleanup
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
        pygame.quit()


if __name__ == '__main__':
    main()
