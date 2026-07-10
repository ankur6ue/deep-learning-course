from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class Episode:
    observations: list[np.ndarray]
    actions: list[int]
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


class ValueNetwork(nn.Module):
    """
    Separate critic network used only as a baseline.

    Keeping the critic separate prevents value-regression updates from moving
    the policy's hidden features. The policy still sees the critic through the
    detached advantage, but the critic loss no longer directly changes policy
    parameters.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
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

    observations = []
    actions = []
    rewards = []

    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        # The baseline version recomputes logits for the whole episode during
        # the update below. We only need actions while collecting data, so do
        # not retain an unused autograd graph for every environment step.
        with torch.no_grad():
            dist = policy.action_distribution(obs_tensor)
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(
            int(action.item())
        )

        done = terminated or truncated

        observations.append(obs.copy())
        actions.append(int(action.item()))
        rewards.append(float(reward))

        obs = next_obs

    return Episode(
        observations=observations,
        actions=actions,
        rewards=rewards,
    )


def train_reinforce_with_baseline(
    episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-2,
    policy_lr: float | None = None,
    critic_lr: float | None = None,
    value_loss_coef: float = 0.1,
    max_grad_norm: float | None = 1.0,
    normalize_advantages: bool = False,
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
    critic = ValueNetwork(obs_dim).to(device)
    effective_policy_lr = lr if policy_lr is None else policy_lr
    effective_critic_lr = lr if critic_lr is None else critic_lr
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=effective_policy_lr,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=effective_critic_lr,
    )

    episode_returns_history = []

    for episode_idx in range(episodes):
        episode = collect_episode(env, policy, device)

        obs_tensor = torch.tensor(
            np.array(episode.observations),
            dtype=torch.float32,
            device=device,
        )

        returns = compute_returns(episode.rewards, gamma).to(device)

        logits = policy(obs_tensor)
        values = critic(obs_tensor)
        dist = Categorical(logits=logits)

        actions = torch.tensor(
            episode.actions,
            dtype=torch.int64,
            device=device,
        )

        log_probs = dist.log_prob(actions)

        # Advantage = return - learned baseline.
        # values.detach() because we do not want the policy gradient to
        # backprop through the value network. The value loss is calculated
        # separately below.
        advantages = returns - values.detach()

        # Optional advantage normalization. This can further reduce scale
        # variation, but it also changes the relative weighting of timesteps in
        # short episodes, so keep it explicit rather than always on.
        if normalize_advantages and advantages.numel() > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)

        # Keep the policy-gradient term as a sum over timesteps, matching the
        # no-baseline REINFORCE objective. Longer successful trajectories then
        # contribute more gradient signal, which is what we want for CartPole.
        policy_loss = -(advantages * log_probs).sum()

        # The critic is a regression problem over visited states. Average its
        # MSE so the value update does not grow just because the episode got
        # longer. Since the critic is separate, this only controls critic scale;
        # it no longer directly moves the policy's hidden representation.
        value_loss = F.mse_loss(values, returns)

        # Use separate backward/optimizer steps. This keeps the critic's large
        # regression gradients from changing the scale of the policy update via
        # global gradient clipping or shared optimizer state.
        policy_optimizer.zero_grad()
        policy_loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        policy_optimizer.step()

        critic_optimizer.zero_grad()
        (value_loss_coef * value_loss).backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_optimizer.step()

        episode_return = sum(episode.rewards)
        episode_returns_history.append(episode_return)

        if (episode_idx + 1) % 50 == 0:
            avg_return = np.mean(episode_returns_history[-50:])
            print(
                f"Episode {episode_idx + 1:4d} | "
                f"avg return over last 50 = {avg_return:.1f}"
            )

    env.close()
    return (policy, critic), episode_returns_history


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
        description="Train REINFORCE on CartPole with a learned baseline.",
    )
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=None,
        help="Policy learning rate. Defaults to --lr when omitted.",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=None,
        help="Critic learning rate. Defaults to --lr when omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-loss-coef", type=float, default=0.1)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Set to 0 or a negative value to disable gradient clipping.",
    )
    parser.add_argument(
        "--normalize-advantages",
        action="store_true",
        help="Normalize advantages inside each episode before the policy loss.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("with_baseline.npy"),
        help=(
            "Where to save per-episode returns. Relative paths are resolved "
            "relative to this script."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_grad_norm = (
        None if args.max_grad_norm <= 0.0 else args.max_grad_norm
    )
    _, returns = train_reinforce_with_baseline(
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=max_grad_norm,
        normalize_advantages=args.normalize_advantages,
        seed=args.seed,
    )
    output_path = save_returns(returns, args.output)

    print("Final average return over last 100 episodes:")
    print(np.mean(returns[-100:]))
    print(f"Saved episode returns to {output_path}")


if __name__ == "__main__":
    main()
