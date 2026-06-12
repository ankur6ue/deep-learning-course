from __future__ import annotations

from typing import Dict

from gridworld import Action, Gridworld, State


Policy = Dict[State, Dict[Action, float]]


def make_random_policy(env: Gridworld) -> Policy:
    policy: Policy = {}

    for state in env.states:
        if env.is_terminal(state):
            policy[state] = {}
        else:
            action_prob = 1.0 / len(env.actions)
            policy[state] = {
                action: action_prob
                for action in env.actions
            }

    return policy


def iterative_policy_evaluation(
    env: Gridworld,
    policy: Policy,
    gamma: float = 0.99,
    theta: float = 1e-8,
) -> tuple[Dict[State, float], int]:
    """
    Evaluate a fixed policy using the Bellman expectation equation.

    V_{k+1}(s) =
        sum_a pi(a|s) [r + gamma V_k(s')]

    This environment is deterministic, so each action leads to one next state.
    """

    values = {state: 0.0 for state in env.states}
    iteration = 0

    while True:
        delta = 0.0
        new_values = values.copy()

        for state in env.states:
            if env.is_terminal(state):
                new_values[state] = 0.0
                continue

            new_value = 0.0

            for action, action_prob in policy[state].items():
                next_state, reward = env.transition(state, action)
                new_value += action_prob * (
                    reward + gamma * values[next_state]
                )

            delta = max(delta, abs(new_value - values[state]))
            new_values[state] = new_value

        values = new_values
        iteration += 1

        if delta < theta:
            break

    return values, iteration


def main() -> None:
    env = Gridworld(step_reward=-0.01)

    random_policy = make_random_policy(env)

    values, iterations = iterative_policy_evaluation(
        env=env,
        policy=random_policy,
        gamma=0.99,
    )

    print(f"Random policy evaluation converged in {iterations} iterations")
    env.print_values(values)


if __name__ == "__main__":
    main()