from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class Episode:
    log_probs: list[torch.Tensor]
    rewards: list[float]


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def action_distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.forward(obs)
        return Categorical(logits=logits)


def compute_returns(
    rewards: list[float],
    gamma: float,
) -> torch.Tensor:
    returns = []
    G = 0.0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.append(G)

    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def collect_episode(
    env: gym.Env,
    policy: PolicyNetwork,
    device: torch.device,
) -> Episode:
    obs, _ = env.reset()

    log_probs = []
    rewards = []

    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        dist = policy.action_distribution(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(
            int(action.item())
        )

        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(float(reward))

        obs = next_obs

    return Episode(log_probs=log_probs, rewards=rewards)


def train_reinforce(
    episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-2,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    episode_returns_history = []

    for episode_idx in range(episodes):
        episode = collect_episode(env, policy, device)

        returns = compute_returns(episode.rewards, gamma).to(device)

        # Normalize returns for variance reduction.
        # This is not required by the theory, but helps training.
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(episode.log_probs)

        loss = -(returns * log_probs).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_return = sum(episode.rewards)
        episode_returns_history.append(episode_return)

        if (episode_idx + 1) % 50 == 0:
            avg_return = np.mean(episode_returns_history[-50:])
            print(
                f"Episode {episode_idx + 1:4d} | "
                f"avg return over last 50 = {avg_return:.1f}"
            )

    env.close()
    return policy, episode_returns_history


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def save_returns(returns: list[float], output_path: Path) -> Path:
    output_path = resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(returns, dtype=np.float32))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train REINFORCE on CartPole without a baseline.",
    )
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("without_baseline.npy"),
        help=(
            "Where to save per-episode returns. Relative paths are resolved "
            "relative to this script."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, returns = train_reinforce(
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        seed=args.seed,
    )
    output_path = save_returns(returns, args.output)

    print("Final average return over last 100 episodes:")
    print(np.mean(returns[-100:]))
    print(f"Saved episode returns to {output_path}")


if __name__ == "__main__":
    main()
