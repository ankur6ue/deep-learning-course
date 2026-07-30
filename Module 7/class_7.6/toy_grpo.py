from __future__ import annotations

import copy

import torch
from torch import nn
from torch.distributions import Categorical


class TinySequencePolicy(nn.Module):
    """
    Prompt-conditioned policy over fixed-length token sequences.

    logits[prompt, position, token]

    Unlike an LLM, positions are conditionally independent here.
    This keeps attention on the GRPO mechanics.
    """

    def __init__(
        self,
        num_prompts: int,
        sequence_length: int,
        vocab_size: int,
    ):
        super().__init__()

        self.num_prompts = num_prompts
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        self.logits = nn.Parameter(
            torch.zeros(
                num_prompts,
                sequence_length,
                vocab_size,
            )
        )

    def distributions(
        self,
        prompt_ids: torch.Tensor,
    ) -> Categorical:
        # [B, T, V]
        logits = self.logits[prompt_ids]
        return Categorical(logits=logits)

    def log_probs(
        self,
        prompt_ids: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prompt_ids: [B]
            tokens: [B, G, T]

        Returns:
            log probabilities: [B, G, T]
        """

        # [B, T, V]
        log_probs_all = torch.log_softmax(
            self.logits[prompt_ids],
            dim=-1,
        )

        B, G, T = tokens.shape

        expanded = log_probs_all.unsqueeze(1).expand(
            B,
            G,
            T,
            self.vocab_size,
        )

        selected = torch.gather(
            expanded,
            dim=-1,
            index=tokens.unsqueeze(-1),
        )

        return selected.squeeze(-1)


@torch.no_grad()
def sample_groups(
    policy: TinySequencePolicy,
    prompt_ids: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        tokens: [B, G, T]
        old_log_probs: [B, G, T]
    """

    dist = policy.distributions(prompt_ids)

    # Categorical batch shape is [B, T].
    # Sampling G times returns [G, B, T].
    tokens = dist.sample((group_size,))
    tokens = tokens.permute(1, 0, 2).contiguous()

    old_log_probs = policy.log_probs(prompt_ids, tokens)

    return tokens, old_log_probs


def reward_sequences(
    tokens: torch.Tensor,
    target_sums: torch.Tensor,
) -> torch.Tensor:
    """
    Reward sequences for matching a prompt-specific target sum.

    Args:
        tokens: [B, G, T]
        target_sums: [B]

    Returns:
        rewards: [B, G]
    """

    generated_sums = tokens.sum(dim=-1)
    distance = torch.abs(
        generated_sums - target_sums.unsqueeze(1)
    )

    # Dense shaping plus a correctness bonus.
    rewards = -distance.float()
    rewards = rewards + 2.0 * (distance == 0).float()

    return rewards


def group_relative_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(
        dim=1,
        keepdim=True,
        unbiased=False,
    )
    return (rewards - mean) / (std + eps)


def train(
    outer_updates: int = 500,
    group_size: int = 4,
    inner_epochs: int = 4,
    clip_epsilon: float = 0.2,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> TinySequencePolicy:
    torch.manual_seed(seed)

    prompt_ids = torch.tensor([0, 1], dtype=torch.long)
    target_sums = torch.tensor([3, 9], dtype=torch.long)

    policy = TinySequencePolicy(
        num_prompts=2,
        sequence_length=3,
        vocab_size=5,
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
    )

    for update in range(outer_updates):
        # Snapshot the rollout policy.
        old_policy = copy.deepcopy(policy).eval()

        tokens, old_log_probs = sample_groups(
            old_policy,
            prompt_ids,
            group_size,
        )

        rewards = reward_sequences(tokens, target_sums)

        advantages = group_relative_advantages(
            rewards
        ).detach()

        # Several optimization passes over the same sampled groups.
        for _ in range(inner_epochs):
            new_log_probs = policy.log_probs(
                prompt_ids,
                tokens,
            )

            ratio = torch.exp(
                new_log_probs - old_log_probs
            )

            token_advantages = advantages.unsqueeze(-1)

            unclipped = ratio * token_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - clip_epsilon,
                1.0 + clip_epsilon,
            ) * token_advantages

            # Equal-length completions, so a simple mean is enough.
            policy_loss = -torch.minimum(
                unclipped,
                clipped,
            ).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        if (update + 1) % 50 == 0:
            with torch.no_grad():
                accuracy = (
                    tokens.sum(dim=-1)
                    == target_sums.unsqueeze(1)
                ).float().mean()

                zero_std_fraction = (
                    rewards.std(
                        dim=1,
                        unbiased=False,
                    ) == 0
                ).float().mean()

            print(
                f"update={update + 1:4d} "
                f"mean_reward={rewards.mean():7.3f} "
                f"sample_accuracy={accuracy:6.3f} "
                f"zero_std_fraction={zero_std_fraction:5.2f}"
            )

    return policy


def main() -> None:
    policy = train()

    print("\nLearned token probabilities:")
    print(torch.softmax(policy.logits, dim=-1))


if __name__ == "__main__":
    main()