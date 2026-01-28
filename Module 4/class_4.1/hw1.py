"""
HW1: Evaluation Metrics for Information Retrieval

Implement basic IR metrics:
- Precision, Recall, F1
- Precision@k, Recall@k
- Average Precision (AP), mean Average Precision (mAP)
- Reciprocal Rank (RR), Mean Reciprocal Rank (MRR)
- NDCG@k

Run: python hw1_ir_metrics.py
"""

from typing import List, Dict, Set, Tuple
import math


# ----------------------------
# Toy data
# ----------------------------

qrels_binary: Dict[str, Dict[str, int]] = {
    "q1": {"d1": 1, "d3": 1},
    "q2": {"d2": 1},
    "q3": {"d2": 1, "d4": 1, "d5": 1},
}

run: Dict[str, List[str]] = {
    "q1": ["d1", "d2", "d3", "d4"],
    "q2": ["d3", "d2", "d5"],
    "q3": ["d5", "d4", "d2", "d1"],
}

qrels_graded: Dict[str, Dict[str, int]] = {
    "q1": {"d1": 2, "d3": 1},
    "q2": {"d2": 3},
    "q3": {"d2": 2, "d4": 1, "d5": 1},
}


# ----------------------------
# Metric implementations
# ----------------------------

def precision_recall_f1(
    retrieved: List[str],
    relevant: Set[str],
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F1 for a single query (binary relevance).

    retrieved: list of doc IDs returned by the system (order does not matter here)
    relevant: set of relevant doc IDs

    Returns: (precision, recall, f1)
    """
    # TODO: implement
    # Hint:
    #   tp = number of retrieved docs that are in relevant
    #   fp = retrieved but not relevant
    #   fn = relevant but not retrieved
    # Be careful with division by zero.
    raise NotImplementedError


def precision_at_k(
    ranked_docs: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """
    Precision@k for a single query.

    ranked_docs: ranked list of docs from the system
    relevant: set of relevant doc IDs
    k: cutoff

    Returns: precision@k
    """
    # TODO: implement
    raise NotImplementedError


def recall_at_k(
    ranked_docs: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """
    Recall@k for a single query.
    """
    # TODO: implement
    raise NotImplementedError


def average_precision(
    ranked_docs: List[str],
    relevant: Set[str],
) -> float:
    """
    Average Precision (AP) for a single query (binary relevance).

    Definition:
      Let the ranks where relevant docs appear be k1, k2, ..., km.
      AP = (1/m) * sum_j P@kj

    If there are no relevant docs, return 0.0.
    """
    # TODO: implement
    # Steps:
    #   1. iterate through ranked_docs
    #   2. every time you see a relevant doc at position i (0-based or 1-based),
    #      compute P@(i+1) and accumulate
    #   3. divide by number of relevant docs
    raise NotImplementedError


def mean_average_precision(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
) -> float:
    """
    Mean Average Precision (mAP) over multiple queries.

    run: dict query_id -> ranked list of docs
    qrels: dict query_id -> {doc_id: relevance (1 or 0)}

    Only queries present in run should be evaluated.
    If a query has no relevant docs, its AP is defined as 0.
    """
    # TODO: implement
    raise NotImplementedError


def reciprocal_rank(
    ranked_docs: List[str],
    relevant: Set[str],
) -> float:
    """
    Reciprocal Rank (RR) for a single query.

    RR = 1 / rank_of_first_relevant_doc
    If there is no relevant doc in ranked_docs, return 0.0.
    """
    # TODO: implement
    raise NotImplementedError


def mean_reciprocal_rank(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
) -> float:
    """
    Mean Reciprocal Rank (MRR) over multiple queries.
    """
    # TODO: implement
    raise NotImplementedError


def ndcg_at_k(
    ranked_docs: List[str],
    rel_grades: Dict[str, int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k (NDCG@k) for graded relevance.

    ranked_docs: ranked list of docs
    rel_grades: dict doc_id -> relevance grade (0,1,2,...)
    k: cutoff

    DCG@k = sum_{i=1..k} (2^rel_i - 1) / log2(i+1)
    IDCG@k = DCG@k of ideal ranking (docs sorted by rel grade)

    If IDCG@k is 0, return 0.
    """
    # TODO: implement
    # Hints:
    #   - For DCG, iterate over ranked_docs[:k], look up each doc's grade (default 0).
    #   - For IDCG, sort all docs in rel_grades by grade (descending), then compute DCG on that ideal list.
    raise NotImplementedError


# ----------------------------
# Main: simple tests
# ----------------------------

def main():
    # Convert binary qrels to sets for convenience
    relevant_sets = {
        qid: {doc_id for doc_id, rel in rels.items() if rel > 0}
        for qid, rels in qrels_binary.items()
    }

    print("=== Single-query metrics (q1) ===")
    qid = "q1"
    ranked = run[qid]
    rel = relevant_sets[qid]

    p, r, f1 = precision_recall_f1(ranked, rel)
    print(f"Precision(q1): {p:.3f}")
    print(f"Recall(q1):    {r:.3f}")
    print(f"F1(q1):        {f1:.3f}")

    for k in [1, 2, 3, 4]:
        pk = precision_at_k(ranked, rel, k)
        rk = recall_at_k(ranked, rel, k)
        print(f"P@{k}(q1): {pk:.3f}, R@{k}(q1): {rk:.3f}")

    ap_q1 = average_precision(ranked, rel)
    rr_q1 = reciprocal_rank(ranked, rel)
    print(f"AP(q1):  {ap_q1:.3f}")
    print(f"RR(q1):  {rr_q1:.3f}")

    print("\n=== Multi-query metrics ===")
    map_all = mean_average_precision(run, qrels_binary)
    mrr_all = mean_reciprocal_rank(run, qrels_binary)
    print(f"mAP (all queries): {map_all:.3f}")
    print(f"MRR (all queries): {mrr_all:.3f}")

    print("\n=== NDCG examples ===")
    for qid in ["q1", "q2", "q3"]:
        ranked = run[qid]
        grades = qrels_graded[qid]
        for k in [1, 2, 3]:
            ndcg = ndcg_at_k(ranked, grades, k)
            print(f"NDCG@{k}({qid}): {ndcg:.3f}")
        print("")


if __name__ == "__main__":
    main()
