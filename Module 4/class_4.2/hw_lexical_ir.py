"""
HW2: Lexical Retrieval – Inverted Index, TF-IDF, BM25

Implement:
- Tokenization & normalization
- Inverted index
- TF, DF, IDF
- TF-IDF scoring with cosine similarity
- BM25 scoring

Run: python hw2_lexical_ir.py
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set
import math
import re


# ----------------------------
# Toy corpus & queries
# ----------------------------

documents: Dict[str, str] = {
    "d1": "Best hikes near San Francisco bay area",
    "d2": "Hiking safety tips and trail etiquette",
    "d3": "Top restaurants in San Francisco",
    "d4": "Deep learning for natural language processing",
    "d5": "Neural networks and deep learning basics",
}

queries: Dict[str, str] = {
    "q1": "hikes near san francisco",
    "q2": "deep learning",
}


# ----------------------------
# 1. Tokenization / normalization
# ----------------------------

def tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer:
    - lowercase
    - remove non-alphabetic characters (keep spaces)
    - split on whitespace

    Example:
      "Best hikes near San Francisco bay area" ->
      ["best", "hikes", "near", "san", "francisco", "bay", "area"]
    """
    # TODO: implement
    # Hint: use re.sub to replace non-letters with space, then split.
    raise NotImplementedError


# ----------------------------
# 2. Build inverted index and term statistics
# ----------------------------

def build_inverted_index(
    docs: Dict[str, str]
) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, int], Dict[str, int]]:
    """
    Build inverted index and term statistics.

    Returns:
      inverted_index: term -> list of (doc_id, tf(term, doc))
      doc_lengths: doc_id -> number of tokens in that doc
      df: term -> document frequency (in how many docs the term appears)

    Implementation steps:
      - For each doc_id, text in docs:
          * tokenize text
          * count term frequencies in that doc
          * update inverted_index and df
          * store doc length
    """
    inverted_index: Dict[str, List[Tuple[str, int]]] = {}
    doc_lengths: Dict[str, int] = {}
    df: Dict[str, int] = {}

    # TODO: implement
    raise NotImplementedError


def compute_idf(df: Dict[str, int], N: int) -> Dict[str, float]:
    """
    Compute IDF for each term using a smoothed formula:

      idf(t) = log( (N + 1) / (df(t) + 1) )

    where N is the total number of documents.
    """
    idf: Dict[str, float] = {}

    # TODO: implement
    raise NotImplementedError


# ----------------------------
# 3. TF-IDF vectors and cosine similarity
# ----------------------------

def tf_idf_vector(
    tokens: List[str],
    idf: Dict[str, float],
) -> Dict[str, float]:
    """
    Build a *sparse representation* of the TF-IDF vector for a document/query.

    Conceptual view:
      - There is a full vocabulary V.
      - The TF-IDF vector v ∈ R^|V| has one coordinate per term in V.
      - For any term t not in the document/query, tf(t) = 0 ⇒ tf-idf(t) = 0.

    Implementation view:
      - We do NOT build a full |V|-dim dense vector.
      - Instead, we return a dict: term -> tf-idf weight for terms that appear
        in `tokens`. All other terms are implicitly weight 0.

    Steps:
      - Count term frequencies in `tokens`.
      - For each term, compute: tf-idf = tf * idf(term).
      - Ignore terms that are not in the idf dict or treat their idf as 0.

    Returns:
      A dict mapping term -> tf-idf weight (sparse vector).
    """
    # TODO: implement
    raise NotImplementedError


def cosine_similarity(
    v1: Dict[str, float],
    v2: Dict[str, float],
) -> float:
    """
      Compute cosine similarity between two sparse TF-IDF vectors.

      v1, v2: dict term -> weight for NON-ZERO coordinates only.
              Terms that do not appear as keys are implicitly assumed to have
              weight 0 in that vector.

      cos(v1, v2) = (sum_t v1[t] * v2[t]) / (||v1|| * ||v2||)

      Implementation hint:
        - Dot product: iterate over the intersection of keys, or over keys of one
          dict and look up v2.get(term, 0.0).
        - Norm: sqrt(sum_t (v1[t]^2)) over keys of v1 (similar for v2).
        - If either norm is 0 (vector has all zeros), return 0.0.
      """
    # TODO: implement
    raise NotImplementedError


def rank_with_tfidf(
    query: str,
    docs: Dict[str, str],
    idf: Dict[str, float],
) -> List[Tuple[str, float]]:
    """
    Rank documents for a query using TF-IDF cosine similarity.

    Returns:
      list of (doc_id, score) sorted by decreasing score.
    """
    # TODO: implement
    # Steps:
    #   - tokenize query, build TF-IDF vector for query
    #   - for each doc, build TF-IDF vector and compute cosine similarity to query
    #   - sort doc_ids by score descending
    raise NotImplementedError


# ----------------------------
# 4. BM25 scoring
# ----------------------------

def bm25_scores(
    query: str,
    inverted_index: Dict[str, List[Tuple[str, int]]],
    doc_lengths: Dict[str, int],
    idf_bm25: Dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, float]:
    """
    Compute BM25 scores for all documents for a given query.

    BM25(q, d) = sum over terms t in q:
       idf_bm25(t) * [ tf(t,d) * (k1 + 1) ] /
                     [ tf(t,d) + k1 * (1 - b + b * |d| / avgdl) ]

    Implementation steps:
      - tokenize query, de-duplicate query terms if you like
      - precompute avgdl from doc_lengths
      - for each term t in the query:
          look up posting list inverted_index[t] -> list of (doc_id, tf)
          for each (doc_id, tf_td) in posting list:
             compute contribution to that doc's score
      - documents that never appear in any posting list for query terms get score 0.
    """
    # TODO: implement
    raise NotImplementedError


def compute_bm25_idf(df: Dict[str, int], N: int) -> Dict[str, float]:
    """
    BM25-style IDF:

      idf_bm25(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) )

    We'll clip negative values at 0 for safety: max(idf, 0.0)
    """
    idf_bm25: Dict[str, float] = {}

    # TODO: implement
    raise NotImplementedError


# ----------------------------
# Main: simple demo
# ----------------------------

def main():
    # 1) Build index & stats
    inverted_index, doc_lengths, df = build_inverted_index(documents)
    N = len(documents)
    idf = compute_idf(df, N)
    idf_bm25 = compute_bm25_idf(df, N)

    print("Documents:")
    for doc_id, text in documents.items():
        print(f"{doc_id}: {text}")
    print()

    # 2) TF-IDF ranking
    print("=== TF-IDF ranking ===")
    for qid, qtext in queries.items():
        rankings = rank_with_tfidf(qtext, documents, idf)
        print(f"Query {qid!r}: {qtext!r}")
        for doc_id, score in rankings:
            print(f"  {doc_id:>2}  score={score:.4f}")
        print()

    # 3) BM25 ranking
    print("=== BM25 ranking ===")
    for qid, qtext in queries.items():
        scores = bm25_scores(qtext, inverted_index, doc_lengths, idf_bm25)
        # sort by decreasing score
        rankings = sorted(scores.items(), key=lambda x: -x[1])
        print(f"Query {qid!r}: {qtext!r}")
        for doc_id, score in rankings:
            print(f"  {doc_id:>2}  BM25={score:.4f}")
        print()


if __name__ == "__main__":
    main()
