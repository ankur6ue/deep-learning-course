from __future__ import annotations

from typing import Dict

from gridworld import Action, Gridworld, State


def value_iteration(
    env: Gridworld,
    gamma: float = 0.99,
    theta: float = 1e-8,
) -> tuple[Dict[State, float], int]:
    """
    Compute optimal value function using value iteration.

    V_{k+1}(s) =
        max_a [r + gamma V_k(s')]
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

            action_values = []

            for action in env.actions:
                next_state, reward = env.transition(state, action)
                action_value = reward + gamma * values[next_state]
                action_values.append(action_value)

            best_value = max(action_values)

            delta = max(delta, abs(best_value - values[state]))
            new_values[state] = best_value

        values = new_values
        iteration += 1

        if delta < theta:
            break

    return values, iteration


def extract_greedy_policy(
    env: Gridworld,
    values: Dict[State, float],
    gamma: float = 0.99,
) -> Dict[State, Action]:
    """
    Extract greedy policy from a value function.

    pi(s) = argmax_a [r + gamma V(s')]
    """

    policy: Dict[State, Action] = {}

    for state in env.states:
        if env.is_terminal(state):
            policy[state] = "T"
            continue

        best_action = None
        best_action_value = -float("inf")

        for action in env.actions:
            next_state, reward = env.transition(state, action)
            action_value = reward + gamma * values[next_state]

            if action_value > best_action_value:
                best_action_value = action_value
                best_action = action

        if best_action is None:
            raise RuntimeError(f"No best action found for state {state}")

        policy[state] = best_action

    return policy


def main() -> None:
    env = Gridworld(step_reward=-0.01)
    gamma = 0.99
    values, iterations = value_iteration(
        env=env,
        gamma=gamma,
    )

    print(f"Value iteration converged in {iterations} iterations")
    print("Optimal values:")
    env.print_values(values)

    policy = extract_greedy_policy(
        env=env,
        values=values,
        gamma=gamma,
    )

    print("Greedy policy:")
    env.print_policy(policy)


if __name__ == "__main__":
    main()