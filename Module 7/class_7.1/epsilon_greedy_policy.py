import numpy as np
import matplotlib.pyplot as plt


class GaussianBandit:
    def __init__(self, k: int, reward_std: float, rng: np.random.Generator):
        self.k = k
        self.reward_std = reward_std
        self.rng = rng

        self.q_star = self.rng.normal(0.0, 1.0, size=k)
        self.best_action = int(np.argmax(self.q_star))

    def step(self, action: int) -> float:
        return float(self.rng.normal(self.q_star[action], self.reward_std))


class EpsilonGreedyAgent:
    def __init__(self, k: int, epsilon: float, rng: np.random.Generator):
        self.k = k
        self.epsilon = epsilon
        self.rng = rng

        self.q = np.zeros(k)
        self.n = np.zeros(k, dtype=np.int64)

    def act(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))

        max_q = np.max(self.q)
        best_actions = np.flatnonzero(self.q == max_q)
        # Randomly selects among equally good actions
        return int(self.rng.choice(best_actions))

    def update(self, action: int, reward: float) -> None:
        self.n[action] += 1
        alpha = 1.0 / self.n[action]
        self.q[action] += alpha * (reward - self.q[action])


def simulate(epsilon: float, runs: int, steps: int, k: int, seed: int):
    rng = np.random.default_rng(seed)

    all_rewards = np.zeros((runs, steps))
    all_optimal = np.zeros((runs, steps))

    for run in range(runs):
        env_rng = np.random.default_rng(rng.integers(1_000_000_000))
        agent_rng = np.random.default_rng(rng.integers(1_000_000_000))

        bandit = GaussianBandit(k=k, reward_std=1.0, rng=env_rng)
        agent = EpsilonGreedyAgent(k=k, epsilon=epsilon, rng=agent_rng)

        for t in range(steps):
            action = agent.act()
            reward = bandit.step(action)
            agent.update(action, reward)

            all_rewards[run, t] = reward
            all_optimal[run, t] = action == bandit.best_action

    return all_rewards.mean(axis=0), all_optimal.mean(axis=0)


def main():
    runs = 500
    steps = 1000
    k = 10
    epsilons = [0.0, 0.01, 0.1, 0.3]

    plt.figure()
    for eps in epsilons:
        avg_reward, _ = simulate(
            epsilon=eps,
            runs=runs,
            steps=steps,
            k=k,
            seed=123,
        )
        plt.plot(avg_reward, label=f"epsilon={eps}")

    plt.xlabel("Step")
    plt.ylabel("Average reward")
    plt.title(f"Average reward over {runs} bandit problems")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for eps in epsilons:
        _, optimal = simulate(
            epsilon=eps,
            runs=runs,
            steps=steps,
            k=k,
            seed=123,
        )
        plt.plot(optimal, label=f"epsilon={eps}")

    plt.xlabel("Step")
    plt.ylabel("% optimal action")
    plt.title(f"Optimal action frequency over {runs} bandit problems")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()