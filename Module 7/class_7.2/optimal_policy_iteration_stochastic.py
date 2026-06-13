from __future__ import annotations

from typing import Dict

from gridworld import Action, Gridworld, State

def slip_distribution(action: str) -> list[tuple[str, float]]:
    if action == "U":
        return [("U", 0.8), ("L", 0.1), ("R", 0.1)]
    if action == "D":
        return [("D", 0.8), ("L", 0.1), ("R", 0.1)]
    if action == "L":
        return [("L", 0.8), ("U", 0.1), ("D", 0.1)]
    if action == "R":
        return [("R", 0.8), ("U", 0.1), ("D", 0.1)]

    raise ValueError(f"Unknown action: {action}")


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

            # for the stochastic version, we calculate the expected value of each action by considering all possible
            # state transitions and their probabilities
            for intended_action in env.actions:
                expected_action_value = 0.0

                for actual_action, prob in slip_distribution(intended_action):
                    next_state, reward = env.transition(state, actual_action)
                    expected_action_value += prob * (
                            reward + gamma * values[next_state]
                    )

                action_values.append(expected_action_value)

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

        for intended_action in env.actions:
            expected_action_value = 0.0

            for actual_action, prob in slip_distribution(intended_action):
                next_state, reward = env.transition(state, actual_action)
                expected_action_value += prob * (
                        reward + gamma * values[next_state]
                )
            if expected_action_value > best_action_value:
                best_action_value = expected_action_value
                best_action = intended_action


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