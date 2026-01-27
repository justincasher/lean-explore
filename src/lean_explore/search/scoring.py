"""Score normalization and fusion algorithms for search ranking.

This module provides utilities for combining multiple retrieval signals into
a unified ranking using techniques like Reciprocal Rank Fusion (RRF) and
weighted score fusion.
"""

import difflib
import math

EPSILON = 1e-9


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1] range.

    Args:
        scores: List of raw scores.

    Returns:
        List of normalized scores.
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range < EPSILON:
        if max_score > EPSILON:
            return [1.0] * len(scores)
        return [0.0] * len(scores)

    return [(s - min_score) / score_range for s in scores]


def normalize_dependency_counts(counts: list[int]) -> list[float]:
    """Log-scale normalization for dependency counts.

    Uses log(1 + count) / log(1 + max_count) to compress the range
    and give more credit to items with moderate dependency counts.

    Args:
        counts: List of dependency counts.

    Returns:
        List of normalized scores in [0, 1] range.
    """
    if not counts:
        return []

    max_count = max(counts)
    if max_count == 0:
        return [0.0] * len(counts)

    log_max = math.log(1 + max_count)
    return [math.log(1 + c) / log_max for c in counts]


def compute_ranks(scores: list[float]) -> list[int]:
    """Compute ranks for a list of scores (1-indexed, higher score = lower rank).

    Candidates with score 0 get rank len(scores)+1 (worst possible).

    Args:
        scores: List of raw scores.

    Returns:
        List of ranks (1 = best).
    """
    n = len(scores)
    indexed = [(i, s) for i, s in enumerate(scores)]
    indexed.sort(key=lambda x: x[1], reverse=True)

    ranks = [0] * n
    for rank, (idx, score) in enumerate(indexed, 1):
        if score > 0:
            ranks[idx] = rank
        else:
            ranks[idx] = n + 1

    return ranks


def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 0) -> list[float]:
    """Compute RRF scores from multiple rank lists.

    RRF(d) = sum(1 / (k + rank_i(d)) for each signal i)

    Args:
        rank_lists: List of rank lists, one per signal.
        k: Constant to prevent top rank from dominating. Default 0 means 1/rank.

    Returns:
        List of RRF scores for each candidate.
    """
    n = len(rank_lists[0])
    rrf_scores = []

    for i in range(n):
        score = sum(1.0 / (k + ranks[i]) for ranks in rank_lists)
        rrf_scores.append(score)

    return rrf_scores


def weighted_score_fusion(
    score_lists: list[list[float]],
    weights: list[float],
) -> list[float]:
    """Combine multiple score lists using weighted normalized scores.

    Each score list is normalized to [0, 1] using min-max scaling,
    then combined with the given weights.

    Args:
        score_lists: List of score lists, one per signal.
        weights: Weight for each signal (should sum to 1.0 for interpretability).

    Returns:
        List of combined scores for each candidate.
    """
    if not score_lists:
        return []

    n = len(score_lists[0])
    if n == 0:
        return []

    normalized_lists = [normalize_scores(scores) for scores in score_lists]

    combined = []
    for i in range(n):
        score = sum(w * normalized_lists[j][i] for j, w in enumerate(weights))
        combined.append(score)

    return combined


def fuzzy_name_score(query: str, name: str) -> float:
    """Compute fuzzy match score between query and declaration name.

    Normalizes both strings (dots/underscores -> spaces) and uses
    SequenceMatcher ratio for character-level similarity.

    Args:
        query: Search query string.
        name: Declaration name to match against.

    Returns:
        Similarity score between 0 and 1.
    """
    normalized_query = query.lower().replace(".", " ").replace("_", " ")
    normalized_name = name.lower().replace(".", " ").replace("_", " ")
    return difflib.SequenceMatcher(None, normalized_query, normalized_name).ratio()
