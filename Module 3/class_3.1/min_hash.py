# Copyright 2025 Ankur Mohan
import random
from collections import Counter

def jaccard(A, B):
    A, B = set(A), set(B)
    return len(A & B) / len(A | B)


def random_permutation(universe):
    """Return a random permutation of the universe as a list."""
    perm = list(universe)
    random.shuffle(perm)
    return perm


def min_under_permutation(S, perm, pos):
    """
    Given a set S and a permutation perm of the universe,
    return the element of S that appears first in perm.
    `pos` is a dict: element -> index in perm.
    """
    return min(S, key=lambda x: pos[x])


def experiment(A, B, universe, num_trials=100):
    """
    Empirically estimate P[min_pi(A) == min_pi(B)] over random permutations pi.
    """
    A, B = set(A), set(B)
    universe = list(universe)

    true_j = jaccard(A, B)
    collisions = 0

    for _ in range(num_trials):
        perm = random_permutation(universe)
        # Precompute positions of each element in the permutation
        pos = {elem: i for i, elem in enumerate(perm)}
        minA = min_under_permutation(A, perm, pos)
        minB = min_under_permutation(B, perm, pos)
        if minA == minB:
            collisions += 1

    est_prob = collisions / num_trials
    return true_j, est_prob


if __name__ == "__main__":
    random.seed(0)

    # Define a small universe and two sets
    U = set(range(1, 21))  # {1,2,...,20}

    A = {1, 2, 3, 4, 5, 6}
    B = {4, 5, 6, 7, 8}

    # True Jaccard
    true_J = jaccard(A, B)
    print("Set A:", A)
    print("Set B:", B)
    print("Intersection:", A & B)
    print("Union:", A | B)
    print("True Jaccard(A,B) =", true_J)

    # Run experiment
    for trials in [10, 50, 100, 1000, 5000]:
        true_j, est = experiment(A, B, U, num_trials=trials)
        print(f"\nTrials = {trials:6d}  Estimated P[min match] = {est:.4f}")

    # You can also try very different sets
    A2 = {1, 2, 3}
    B2 = {100, 200, 300} & U  # empty in this universe
    if B2:
        print("\nA2, B2 Jaccard:", jaccard(A2, B2))
        _, est2 = experiment(A2, B2, U, num_trials=10_000)
        print("Estimated P[min match] (disjoint-ish) =", est2)
