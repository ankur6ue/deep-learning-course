from __future__ import annotations

from typing import Dict, List, Tuple


State = Tuple[int, int]
Action = str


class Gridworld:
    """
    Small deterministic Gridworld.

    Layout:

        S   .   .   +1
        .   X   .   -1
        .   .   .    .

    The agent can move up, down, left, or right.

    If it hits a wall or boundary, it stays in place.

    Terminal states:
        (0, 3): +1
        (1, 3): -1

    The value of a terminal state is treated as 0 after termination.
    The terminal reward is received when entering the terminal state.
    """

    def __init__(self, step_reward: float = -0.01):
        self.height = 3
        self.width = 4

        self.wall: State = (1, 1)

        self.terminal_rewards: Dict[State, float] = {
            (0, 3): 1.0,
            (1, 3): -1,
        }

        self.step_reward = step_reward

        self.actions: List[Action] = ["U", "D", "L", "R"]

        self.action_to_delta = {
            "U": (-1, 0),
            "D": (1, 0),
            "L": (0, -1),
            "R": (0, 1),
        }

        self.states: List[State] = []
        for r in range(self.height):
            for c in range(self.width):
                state = (r, c)
                if state != self.wall:
                    self.states.append(state)

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_rewards

    def transition(self, state: State, action: Action) -> Tuple[State, float]:
        """
        Deterministic transition.

        Returns:
            next_state, reward
        """

        if self.is_terminal(state):
            return state, 0.0

        dr, dc = self.action_to_delta[action]
        r, c = state

        next_state = (r + dr, c + dc)

        out_of_bounds = (
            next_state[0] < 0
            or next_state[0] >= self.height
            or next_state[1] < 0
            or next_state[1] >= self.width
        )

        if out_of_bounds or next_state == self.wall:
            next_state = state

        if next_state in self.terminal_rewards:
            reward = self.terminal_rewards[next_state]
        else:
            reward = self.step_reward

        return next_state, reward

    def print_values(self, values: Dict[State, float]) -> None:
        for r in range(self.height):
            row = []
            for c in range(self.width):
                state = (r, c)

                if state == self.wall:
                    row.append("  WALL ")
                else:
                    row.append(f"{values[state]:7.3f}")

            print(" ".join(row))
        print()

    def print_policy(self, policy: Dict[State, Action]) -> None:
        for r in range(self.height):
            row = []
            for c in range(self.width):
                state = (r, c)

                if state == self.wall:
                    row.append("  X ")
                elif self.is_terminal(state):
                    row.append("  T ")
                else:
                    row.append(f"  {policy[state]} ")

            print(" ".join(row))
        print()