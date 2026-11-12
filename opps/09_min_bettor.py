#!/usr/bin/env python3.11
"""09_min_bettor: Limps preflop, min-bets postflop, calls small bets; discards lowest rank."""
import json
import os
import random
import sys

RANKS = "23456789TJQKA"
_SEED = os.environ.get("OPP_SEED")
RNG = random.Random(int(_SEED)) if _SEED is not None else random.Random(909)


def choose_action(req):
    legal = req["legal_actions"]
    st, tc, pot = req["street"], req["to_call"], req["pot"]
    if st == "preflop":
        if "call" in legal and tc <= 4:
            return {"type": "action", "action": "call"}
        if "check" in legal:
            return {"type": "action", "action": "check"}
        if "call" in legal and RNG.random() < 0.4:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold" if "fold" in legal else legal[0]}
    if tc == 0:
        if "raise" in legal and RNG.random() < 0.65:
            return {"type": "action", "action": "raise", "amount": int(req["min_raise"])}
        return {"type": "action", "action": "check" if "check" in legal else legal[0]}
    if "call" in legal and (tc <= max(4, 0.35 * pot) or RNG.random() < 0.3):
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else legal[0]}


def choose_discard(req):
    cards = req["hole_cards"]
    worst = min(cards, key=lambda c: RANKS.index(c[0]))
    return {"type": "discard", "card": worst}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        t = msg.get("type")
        if t == "action_request":
            reply = choose_action(msg)
        elif t == "discard_request":
            reply = choose_discard(msg)
        elif t == "match_end":
            break
        else:
            continue
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()

