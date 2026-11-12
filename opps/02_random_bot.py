#!/usr/bin/env python3.11
"""02_random_bot: Organiser random bot with optional OPP_SEED."""
import json
import os
import random
import sys

_SEED = os.environ.get("OPP_SEED")
RNG = random.Random(int(_SEED)) if _SEED is not None else random.Random()


def choose_action(req):
    legal = req["legal_actions"]
    roll = RNG.random()
    if "raise" in legal and roll < 0.2:
        size = RNG.choice((0.5, 0.75, 1.0, 2.0))
        target = req["opp_bet"] + int(size * (req["pot"] + req["to_call"]))
        amount = max(req["min_raise"], min(req["max_raise"], target))
        return {"type": "action", "action": "raise", "amount": int(amount)}
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "call" in legal and roll < 0.8:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else legal[0]}


def choose_discard(req):
    return {"type": "discard", "card": RNG.choice(req["hole_cards"])}


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

