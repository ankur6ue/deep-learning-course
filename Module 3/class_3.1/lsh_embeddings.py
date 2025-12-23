# Copyright 2025 Ankur Mohan
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Dict, List, Iterable, Tuple, Optional


class CosineLSHIndex:
    """
    Locality-Sensitive Hashing index for cosine similarity over dense embeddings
    using random hyperplane LSH.

    Parameters
    ----------
    dim : int
        Dimensionality of embeddings.
    num_tables : int
        Number of independent hash tables (L).
    num_bits_per_table : int
        Number of hash bits per table (k). More bits → tighter buckets.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, dim: int, num_tables: int = 10,
                 num_bits_per_table: int = 16, seed: int = 42) -> None:
        self.dim = dim
        self.L = num_tables
        self.k = num_bits_per_table

        rng = np.random.default_rng(seed)
        # Hyperplanes shape: (L, k, dim)
        self.hyperplanes = rng.normal(size=(self.L, self.k, self.dim))

        # Hash tables: list of dicts, one per table
        # table[i]: {bucket_key (int): [item_ids]}
        self.tables: List[Dict[int, List[str]]] = [
            defaultdict(list) for _ in range(self.L)
        ]

        # Optional: store vectors to re-rank with exact cosine similarity
        self.vectors: Dict[str, np.ndarray] = {}

    # -------------------------
    # Internal helpers
    # -------------------------

    def _hash_vector(self, v: np.ndarray) -> List[int]:
        """
        Compute L integer bucket keys for vector v, one per table.
        Each key encodes k bits (hyperplane signs) as an integer.
        """
        # v: (dim,)
        # projections: (L, k)
        projections = np.tensordot(self.hyperplanes, v, axes=([2], [0]))
        bits = (projections >= 0).astype(np.uint8)  # (L, k)

        # Pack bits into ints for each table
        keys: List[int] = []
        for i in range(self.L):
            key = 0
            for b in bits[i]:
                key = (key << 1) | int(b)
            keys.append(key)
        return keys

    # -------------------------
    # Public API
    # -------------------------

    def add(self, item_id: str, vector: np.ndarray) -> None:
        """
        Add an item to the index.

        item_id : str
            Identifier for this item (e.g., document ID).
        vector : np.ndarray
            Embedding of shape (dim,).
        """
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected vector of shape {(self.dim,)}, got {vector.shape}")

        self.vectors[item_id] = vector
        keys = self._hash_vector(vector)
        for t_idx, key in enumerate(keys):
            self.tables[t_idx][key].append(item_id)

    def add_many(self, items: Iterable[Tuple[str, np.ndarray]]) -> None:
        """Convenience method to add multiple (item_id, vector) pairs."""
        for item_id, vec in items:
            self.add(item_id, vec)

    def query(self,
              vector: np.ndarray,
              max_candidates: Optional[int] = None,
              top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Query the index with a vector.

        Parameters
        ----------
        vector : np.ndarray
            Query embedding of shape (dim,).
        max_candidates : int, optional
            Limit the number of unique candidates before scoring.
        top_k : int, optional
            If provided, returns the top_k items by cosine similarity.

        Returns
        -------
        List of (item_id, cosine_similarity) sorted by decreasing similarity.
        """
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected vector of shape {(self.dim,)}, got {vector.shape}")

        keys = self._hash_vector(vector)

        # Collect candidates from all tables
        candidates = set()
        for t_idx, key in enumerate(keys):
            bucket = self.tables[t_idx].get(key, [])
            for item_id in bucket:
                candidates.add(item_id)
                if max_candidates is not None and len(candidates) >= max_candidates:
                    break
            if max_candidates is not None and len(candidates) >= max_candidates:
                break

        if not candidates:
            return []

        # Convert to list for indexing
        candidates = list(candidates)
        mat = np.stack([self.vectors[c] for c in candidates], axis=0)  # (N, dim)

        # Cosine similarity with the query
        v_norm = np.linalg.norm(vector) + 1e-9
        mat_norms = np.linalg.norm(mat, axis=1) + 1e-9
        sims = (mat @ vector) / (mat_norms * v_norm)  # (N,)

        order = np.argsort(-sims)  # descending
        if top_k is not None:
            order = order[:top_k]

        return [(candidates[i], float(sims[i])) for i in order]


# -------------------------
# Demo
# -------------------------

if __name__ == "__main__":
    dim = 64
    num_items = 1000

    rng = np.random.default_rng(0)

    # Create some random embeddings
    # Cluster around two centers to make neighbors meaningful
    center1 = rng.normal(size=dim)
    center2 = rng.normal(size=dim)

    items = {}
    for i in range(num_items):
        if i < num_items // 2:
            vec = center1 + 0.1 * rng.normal(size=dim)
        else:
            vec = center2 + 0.1 * rng.normal(size=dim)
        items[f"item_{i}"] = vec

    # Build LSH index
    index = CosineLSHIndex(dim=dim, num_tables=12, num_bits_per_table=18, seed=123)
    index.add_many(items.items())

    # Pick a query near center1
    query_vec = center1 + 0.1 * rng.normal(size=dim)

    results = index.query(query_vec, max_candidates=100, top_k=10)

    print("Top 10 neighbors:")
    for item_id, score in results:
        print(f"{item_id:10s}  cosine={score:.3f}")
