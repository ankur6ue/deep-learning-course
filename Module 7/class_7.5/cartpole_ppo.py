from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class RolloutData:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    completed_episode_returns: list[float]
    next_observation: np.ndarray
    ongoing_episode_return: float


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = len(rewards)

    advantages = np.zeros(num_steps, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(num_steps)):
        # A true termination has no future value. A time-limit truncation does
        # bootstrap from its final observation, but both events reset the
        # environment, so GAE must not flow into the next episode in the rollout.
        bootstrap_mask = 0.0 if terminated[t] else 1.0
        episode_continues = 0.0 if terminated[t] or truncated[t] else 1.0

        delta = rewards[t] + gamma * next_values[t] * bootstrap_mask - values[t]
        gae = delta + gamma * gae_lambda * episode_continues * gae

        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


@torch.no_grad()
def collect_rollout(
    env: gym.Env,
    model: ActorCritic,
    observation: np.ndarray,
    ongoing_episode_return: float,
    rollout_length: int,
    gamma: float,
    gae_lambda: float,
) -> RolloutData:
    """Collect exactly ``rollout_length`` transitions.

    Episode boundaries do not determine the rollout boundary. When an episode
    ends, collection resets the environment and continues until the fixed-size
    batch is full. If the batch ends first, its episode continues next update.
    """
    if rollout_length <= 0:
        raise ValueError("rollout_length must be positive")

    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminated_flags = []
    truncated_flags = []
    old_log_probs = []
    old_values = []
    completed_episode_returns = []

    obs = observation

    for _ in range(rollout_length):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        logits, value = model(obs_tensor)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        action_id = int(action.item())

        observations.append(obs.copy())
        actions.append(action_id)
        old_log_probs.append(float(log_prob.item()))
        old_values.append(float(value.item()))

        next_obs, reward, terminated, truncated, _ = env.step(action_id)

        next_observations.append(next_obs.copy())
        rewards.append(float(reward))
        terminated_flags.append(terminated)
        truncated_flags.append(truncated)
        ongoing_episode_return += float(reward)

        if terminated or truncated:
            completed_episode_returns.append(ongoing_episode_return)
            obs, _ = env.reset()
            ongoing_episode_return = 0.0
        else:
            obs = next_obs

    rewards_array = np.asarray(rewards, dtype=np.float32)
    values_array = np.asarray(old_values, dtype=np.float32)
    terminated_array = np.asarray(terminated_flags, dtype=np.bool_)
    truncated_array = np.asarray(truncated_flags, dtype=np.bool_)

    # Value every post-step observation in one batch. Values at true terminal
    # states are masked out by compute_gae; truncated and unfinished rollouts use
    # these predictions for bootstrapping.
    next_observations_array = np.asarray(next_observations, dtype=np.float32)
    _, next_values = model(torch.tensor(next_observations_array, dtype=torch.float32))
    next_values_array = next_values.numpy().astype(np.float32, copy=False)

    advantages, returns = compute_gae(
        rewards=rewards_array,
        values=values_array,
        next_values=next_values_array,
        terminated=terminated_array,
        truncated=truncated_array,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    return RolloutData(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=rewards_array,
        terminated=terminated_array,
        truncated=truncated_array,
        old_log_probs=np.asarray(old_log_probs, dtype=np.float32),
        old_values=values_array,
        advantages=advantages,
        returns=returns,
        completed_episode_returns=completed_episode_returns,
        next_observation=np.asarray(obs, dtype=np.float32),
        ongoing_episode_return=ongoing_episode_return,
    )


def train_ppo(
    total_updates: int = 200,
    rollout_length: int = 2048,
    update_epochs: int = 10,
    minibatch_size: int = 256,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    learning_rate: float = 3e-4,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float | None = 0.02,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    observation, _ = env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(obs_dim, action_dim)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    episode_return_history = []
    ongoing_episode_return = 0.0

    for update_idx in range(total_updates):
        # --------------------------------------------------
        # Rollout phase
        # --------------------------------------------------

        rollout = collect_rollout(
            env=env,
            model=model,
            observation=observation,
            ongoing_episode_return=ongoing_episode_return,
            rollout_length=rollout_length,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        observation = rollout.next_observation
        ongoing_episode_return = rollout.ongoing_episode_return
        episode_return_history.extend(rollout.completed_episode_returns)

        observations = torch.from_numpy(rollout.observations)
        actions = torch.from_numpy(rollout.actions)
        old_log_probs = torch.from_numpy(rollout.old_log_probs)
        advantages = torch.from_numpy(rollout.advantages)
        returns = torch.from_numpy(rollout.returns)

        # Normalize over the fixed rollout, not independently by episode.
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = observations.shape[0]

        # --------------------------------------------------
        # Optimization phase
        # --------------------------------------------------

        stop_early = False

        for epoch in range(update_epochs):
            permutation = torch.randperm(batch_size)

            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start:start + minibatch_size]

                mb_obs = observations[indices]
                mb_actions = actions[indices]
                mb_old_log_probs = old_log_probs[indices]
                mb_advantages = advantages[indices]
                mb_returns = returns[indices]

                logits, new_values = model(mb_obs)
                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                unclipped_objective = ratio * mb_advantages

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon,
                )

                clipped_objective = (
                    clipped_ratio * mb_advantages
                )

                policy_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()

                value_loss = 0.5 * (
                    new_values - mb_returns
                ).pow(2).mean()

                loss = (
                    policy_loss
                    + value_coef * value_loss
                    - entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

                optimizer.step()

                with torch.no_grad():
                    # Common approximate KL estimate.
                    approximate_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean()

                if (
                    target_kl is not None
                    and approximate_kl > target_kl
                ):
                    stop_early = True
                    break

            if stop_early:
                break

        recent_return = np.mean(episode_return_history[-20:])

        print(
            f"update={update_idx + 1:3d} "
            f"steps={batch_size:4d} "
            f"average_return={recent_return:7.1f} "
            f"approx_kl={float(approximate_kl):.5f}"
        )

    env.close()
    return model, episode_return_history


def main() -> None:
    train_ppo()


if __name__ == "__main__":
    main()
