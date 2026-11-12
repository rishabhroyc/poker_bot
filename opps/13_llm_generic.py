#!/usr/bin/env python3.11
"""13_llm_generic: Pot-odds + heuristic hand-strength thresholds with 10% random noise; random discard."""
from collections import Counter
import json
import os
import random
import sys

RANKS = "23456789TJQKA"
_SEED = os.environ.get("OPP_SEED")
RNG = random.Random(int(_SEED)) if _SEED is not None else random.Random(1313)
NOISE_PROB = float(os.environ.get("LLM_NOISE", "0.10"))


def heuristic_strength(hole, board):
    hr = sorted((RANKS.index(c[0]) for c in hole), reverse=True)
    hs = [c[1] for c in hole]
    if len(board) < 3:
        s = 0.35 + 0.015 * hr[0] + 0.008 * hr[1]
        if len(set(hr)) < len(hr):
            s += 0.18
        if len(set(hs)) <= 2:
            s += 0.05
        return min(0.95, s)
    br = [RANKS.index(c[0]) for c in board]
    bs = [c[1] for c in board]
    rc = Counter(hr + br)
    sc = Counter(hs + bs)
    best_cnt = max(rc.values())
    pairs = sum(1 for v in rc.values() if v >= 2)
    hit = any(r in br for r in hr)
    if best_cnt >= 4 or max(sc.values()) >= 5:
        return 0.90
    if best_cnt >= 3 and hit:
        return 0.80
    if pairs >= 2 and hit:
        return 0.72
    if hit or len(set(hr)) < len(hr):
        return 0.54 + 0.008 * hr[0]
    if max(sc.values()) == 4:
        return 0.45
    return 0.25 + 0.008 * hr[0]


def choose_action(req, hole):
    legal = req["legal_actions"]
    tc, pot = req["to_call"], req["pot"]
    if RNG.random() < NOISE_PROB:
        act = RNG.choice(legal)
        if act == "raise":
            amt = max(req["min_raise"], min(req["max_raise"], req["opp_bet"] + max(2, int(0.5 * max(2, pot)))))
            return {"type": "action", "action": "raise", "amount": int(amt)}
        return {"type": "action", "action": act}
    s = heuristic_strength(hole, req["board"])
    if tc == 0:
        if "raise" in legal and s >= 0.60:
            amt = max(req["min_raise"], min(req["max_raise"], req["opp_bet"] + max(2, int(0.6 * pot))))
            return {"type": "action", "action": "raise", "amount": int(amt)}
        return {"type": "action", "action": "check" if "check" in legal else legal[0]}
    po = tc / float(pot + tc) if (pot + tc) > 0 else 0.5
    if "raise" in legal and s >= 0.78:
        amt = max(req["min_raise"], min(req["max_raise"], req["opp_bet"] + max(2, int(0.75 * (pot + tc)))))
        return {"type": "action", "action": "raise", "amount": int(amt)}
    if "call" in legal and s >= po + 0.05:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else legal[0]}


def choose_discard(req):
    return {"type": "discard", "card": RNG.choice(req["hole_cards"])}


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

