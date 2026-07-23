"""Reciprocal Rank Fusion (RRF).

RRF combines several ranked lists (e.g. a semantic ranking and a keyword ranking)
into one, using only each item's *rank* — not its raw score — so lists with
incomparable score scales fuse cleanly. Each item accrues 1 / (k + rank).

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual rank learning methods" (SIGIR 2009).
"""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse ranked lists of ids into a single {id: fused_score} mapping.

    Args:
        rankings: each inner list is item ids in descending relevance order.
        k: smoothing constant; larger k flattens the contribution of top ranks.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] += 1.0 / (k + rank + 1)
    return dict(scores)
