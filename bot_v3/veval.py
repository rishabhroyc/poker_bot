"""Vectorised poker hand evaluator (numpy).

Scores many 5-, 6- or 7-card hands in one call. Scores use the same packed integer
as evaluator.eval_cards (higher is better), which was verified against engine.py.

Method: every 5-card subset is looked up in one of two small tables, built at import
time from evaluator.eval_cards:
  * non-flush table, indexed by the multiset of 5 ranks   (C(17,5) = 6,188 entries)
  * flush table, indexed by the set of 5 distinct ranks    (C(13,5) = 1,287 entries)
A hand's score is the maximum over its 5-card subsets (21 for 7 cards).
Card encoding: card = rank * 4 + suit, rank 0..12 (2..A), suit 0..3 (s, h, d, c).
"""
from itertools import combinations, combinations_with_replacement
from math import comb

import numpy as np

from evaluator import eval_cards

# binomial table: BIN[n, k] for n <= 17, k <= 5
BIN = np.array([[comb(n, k) for k in range(6)] for n in range(18)], dtype=np.int64)


def _nf_index(ranks):
    return sum(comb(r + i, i + 1) for i, r in enumerate(sorted(ranks)))


def _fl_index(ranks):
    return sum(comb(r, i + 1) for i, r in enumerate(sorted(ranks)))


def _build_tables():
    nf = np.zeros(comb(17, 5), dtype=np.int32)
    for ms in combinations_with_replacement(range(13), 5):
        if ms.count(ms[0]) == 5:
            continue  # five of one rank cannot exist
        seen = {}
        cards = []
        for r in ms:  # give copies of a rank different suits; singletons alternate
            k = seen.get(r, 0)
            seen[r] = k + 1
            cards.append(r * 4 + k)
        if len({c & 3 for c in cards}) == 1:  # accidental flush: change one suit
            cards[0] = (cards[0] & ~3) | ((cards[0] + 1) & 3)
        nf[_nf_index(ms)] = eval_cards(cards).score
    fl = np.zeros(comb(13, 5), dtype=np.int32)
    for rs in combinations(range(13), 5):
        fl[_fl_index(rs)] = eval_cards([r * 4 for r in rs]).score
    return nf, fl


NF_TABLE, FL_TABLE = _build_tables()
SUBSETS = {k: np.array(list(combinations(range(k), 5)), dtype=np.int64) for k in (5, 6, 7)}


def veval(cards):
    """cards: int array (N, k), k in {5, 6, 7}. Returns int32 scores (N,)."""
    cards = np.asarray(cards, dtype=np.int64)
    sub = cards[:, SUBSETS[cards.shape[1]]]            # (N, S, 5)
    ranks = np.sort(sub >> 2, axis=2)
    suits = sub & 3
    flush = (suits == suits[:, :, :1]).all(axis=2)
    nf_idx = (BIN[ranks[..., 0], 1] + BIN[ranks[..., 1] + 1, 2] + BIN[ranks[..., 2] + 2, 3]
              + BIN[ranks[..., 3] + 3, 4] + BIN[ranks[..., 4] + 4, 5])
    fl_idx = (BIN[ranks[..., 0], 1] + BIN[ranks[..., 1], 2] + BIN[ranks[..., 2], 3]
              + BIN[ranks[..., 3], 4] + BIN[ranks[..., 4], 5])
    score = np.where(flush, FL_TABLE[np.where(flush, fl_idx, 0)], NF_TABLE[nf_idx])
    return score.max(axis=1)
