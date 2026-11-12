#!/usr/bin/env python3.11
"""11_over_folder: Folds to any bet above 1/3 of pot; discards lowest rank."""
import json
import os
import random
import sys

RANKS = "23456789TJQKA"
_SEED = os.environ.get("OPP_SEED")
RNG = random.Random(int(_SEED)) if _SEED is not None else random.Random(1111)
FOLD_FRAC = float(os.environ.get("OVERFOLDER_FRAC", "0.333333"))


def choose_action(req):
    legal = req["legal_actions"]
    tc, pot = req["to_call"], req["pot"]
    if tc == 0:
        if "raise" in legal and RNG.random() < 0.25:
            amt = max(req["min_raise"], min(req["max_raise"], req["opp_bet"] + max(2, int(0.5 * pot))))
            return {"type": "action", "action": "raise", "amount": int(amt)}
        return {"type": "action", "action": "check" if "check" in legal else legal[0]}
    if "fold" in legal and tc > FOLD_FRAC * max(1, pot):
        return {"type": "action", "action": "fold"}
    if "call" in legal:
        return {"type": "action", "action": "call"}
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

