#!/usr/bin/env python3.11
"""10_overbet_shover: Big jams/overbets with value and some bluffs; smart equity discard."""
from collections import Counter
from itertools import combinations
import json
import os
import random
import sys

RANKS = "23456789TJQKA"
SUITS = "shdc"
C2I = {r + s: i * 4 + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}
_SEED = os.environ.get("OPP_SEED")
RNG = random.Random(int(_SEED)) if _SEED is not None else random.Random(1010)
BLUFF_PROB = float(os.environ.get("SHOVER_BLUFF_PROB", "0.18"))


def _eval5(cards):
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = [c & 3 for c in cards]
    flush = len(set(suits)) == 1
    rc = Counter(ranks)
    groups = sorted(rc.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = [g[1] for g in groups]
    granks = [g[0] for g in groups]
    straight = False
    hi = 0
    if len(rc) == 5:
        if ranks[0] - ranks[4] == 4:
            straight, hi = True, ranks[0]
        elif ranks == [12, 3, 2, 1, 0]:
            straight, hi = True, 3
    if flush and straight:
        return (8, hi)
    if counts[0] == 4:
        return (7, granks[0], granks[1])
    if counts[:2] == [3, 2]:
        return (6, granks[0], granks[1])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, hi)
    if counts[0] == 3:
        return (3, *granks)
    if counts[:2] == [2, 2]:
        return (2, *granks)
    if counts[0] == 2:
        return (1, *granks)
    return (0, *ranks)


def eval_best(cards):
    return max(_eval5(sub) for sub in combinations(cards, 5))


def preflop_strength(hole_str):
    rs = sorted((RANKS.index(c[0]) for c in hole_str), reverse=True)
    ss = [c[1] for c in hole_str]
    pair = rs[0] == rs[1] or rs[1] == rs[2]
    trips = rs[0] == rs[2]
    suited = len(set(ss)) <= 2
    s = 0.34 + 0.017 * rs[0] + 0.010 * rs[1]
    if trips:
        s += 0.24
    elif pair:
        s += 0.15 + 0.012 * rs[1]
    if suited:
        s += 0.05
    return min(0.95, s)


def mc_equity(hole, board, n=45):
    if len(board) < 3:
        return preflop_strength(hole)
    h = [C2I[c] for c in hole]
    b = [C2I[c] for c in board]
    dead = set(h) | set(b)
    deck = [c for c in range(52) if c not in dead]
    need_b = 5 - len(b)
    wins = 0.0
    for _ in range(n):
        samp = RNG.sample(deck, 2 + need_b)
        opp, fb = samp[:2], b + samp[2:]
        my_s = max(eval_best(list(k) + fb) for k in combinations(h, 2)) if len(h) == 3 else eval_best(h + fb)
        op_s = eval_best(opp + fb)
        wins += 1.0 if my_s > op_s else (0.5 if my_s == op_s else 0.0)
    return wins / n


def choose_discard(req):
    hole, board = req["hole_cards"], req["board"]
    best_keep, best_eq = None, -1.0
    for i, j in ((0, 1), (0, 2), (1, 2)):
        eq = mc_equity([hole[i], hole[j]], board, n=35)
        if eq > best_eq:
            best_eq, best_keep = eq, {hole[i], hole[j]}
    return {"type": "discard", "card": next(c for c in hole if c not in best_keep)}


def choose_action(req, hole):
    legal = req["legal_actions"]
    tc, pot = req["to_call"], req["pot"]
    eq = mc_equity(hole, req["board"], n=45)
    if "raise" in legal and (eq >= 0.72 or (eq < 0.38 and RNG.random() < BLUFF_PROB)):
        amt = req["max_raise"] if eq >= 0.80 or RNG.random() < 0.5 else max(
            req["min_raise"], min(req["max_raise"], req["opp_bet"] + max(4, int(2.0 * max(2, pot + tc))))
        )
        return {"type": "action", "action": "raise", "amount": int(amt)}
    if tc == 0:
        return {"type": "action", "action": "check" if "check" in legal else legal[0]}
    po = tc / float(pot + tc) if (pot + tc) > 0 else 0.5
    if "call" in legal and eq >= po + 0.05:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else legal[0]}


def main():
    hole = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        t = msg.get("type")
        if t == "new_hand":
            hole = list(msg["hole_cards"])
            continue
        if t == "discard_result":
            hole = list(msg["hole_cards"])
            continue
        if t == "action_request":
            reply = choose_action(msg, hole)
        elif t == "discard_request":
            reply = choose_discard(msg)
        elif t == "match_end":
            break
        else:
            continue
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()

