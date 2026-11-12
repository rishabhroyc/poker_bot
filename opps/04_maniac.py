#!/usr/bin/env python3.11
"""04_maniac: Raises or shoves almost always; discards lowest rank."""
import json
import os
import random
import sys

RANKS = "23456789TJQKA"
_SEED = os.environ.get("OPP_SEED")
RNG = random.Random(int(_SEED)) if _SEED is not None else random.Random(404)
RAISE_PROB = float(os.environ.get("MANIAC_RAISE_PROB", "0.88"))
SHOVE_PROB = float(os.environ.get("MANIAC_SHOVE_PROB", "0.25"))


def choose_action(req):
    legal = req["legal_actions"]
    if "raise" in legal and RNG.random() < RAISE_PROB:
        lo, hi = req["min_raise"], req["max_raise"]
        if RNG.random() < SHOVE_PROB:
            amt = hi
        else:
            mult = RNG.choice((0.75, 1.0, 1.5, 2.5))
            target = req["opp_bet"] + max(2, int(mult * max(2, req["pot"] + req["to_call"])))
            amt = max(lo, min(hi, target))
        return {"type": "action", "action": "raise", "amount": int(amt)}
    if "call" in legal:
        return {"type": "action", "action": "call"}
    if "check" in legal:
        return {"type": "action", "action": "check"}
    return {"type": "action", "action": legal[0]}


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

