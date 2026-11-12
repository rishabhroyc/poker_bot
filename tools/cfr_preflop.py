#!/usr/bin/env python3.11
"""Offline CFR+ solver for Discard Hold'em preflop abstraction over 1755 hand classes."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time
import zlib

import numpy as np


def build_equity_matrix(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pre = np.load(npz_path)
    cls_arr = pre["combo_class"].astype(np.int64)
    eq = pre["eq"].astype(np.float64)        # (1755,)
    eq_dec = pre["eq_dec"].astype(np.float64)  # (1755, 10)
    counts = np.bincount(cls_arr, minlength=1755).astype(np.float64)
    prior = counts / counts.sum()

    # Map each class to its equity decile 0..9
    order = np.argsort(eq, kind="stable")
    dec_of = np.empty(1755, dtype=np.int64)
    dec_of[order] = np.clip((np.arange(1755) * 10) // 1755, 0, 9)

    # Zero-sum pairwise equity matrix E[i, j] = P(class i beats class j)
    e_ij = eq_dec[:, dec_of]  # (1755, 1755)
    E = 0.5 * (e_ij + (1.0 - e_ij.T))
    return E, prior


# Tree definition:
# Each info-state node: (player 0=BTN/1=BB, invested_btn, invested_bb, actions=[(name, next_node_or_terminal), ...])
# Terminal: ("FOLD", folder_player, inv_btn, inv_bb) or ("SHOW", inv_btn, inv_bb, is_allin)
NODES = {
    0: (0, 1, 2, [("fold", ("FOLD", 0, 1, 2)), ("call", 1), ("raise7", 2), ("shove", 6)]),
    1: (1, 2, 2, [("check", ("SHOW", 2, 2, False)), ("raise7", 3), ("shove", 7)]),
    2: (1, 7, 2, [("fold", ("FOLD", 1, 7, 2)), ("call", ("SHOW", 7, 7, False)), ("raise21", 4), ("shove", 6)]),
    3: (0, 2, 7, [("fold", ("FOLD", 0, 2, 7)), ("call", ("SHOW", 2, 7, False)), ("raise21", 5), ("shove", 7)]),  # wait: call makes it (7,7)
    4: (0, 7, 21, [("fold", ("FOLD", 0, 7, 21)), ("call", ("SHOW", 21, 21, False)), ("shove", 6)]),
    5: (1, 21, 7, [("fold", ("FOLD", 1, 21, 7)), ("call", ("SHOW", 21, 21, False)), ("shove", 7)]),
    6: (1, 200, 7, [("fold", ("FOLD", 1, 200, 7)), ("call", ("SHOW", 200, 200, True))]),
    7: (0, 2, 200, [("fold", ("FOLD", 0, 2, 200)), ("call", ("SHOW", 200, 200, True))]),
}
# Fix node 3 call pot: both invested 7
NODES[3] = (0, 2, 7, [("fold", ("FOLD", 0, 2, 7)), ("call", ("SHOW", 7, 7, False)), ("raise21", 5), ("shove", 7)])
# Note: in node 2 & 3, shove by BB goes to node 7 (BTN facing 200) -> fix node 2 shove -> 7, node 3 shove -> 6, node 4 shove -> 6 (BB facing 200), node 5 shove -> 7 (BTN facing 200)
NODES[2] = (1, 7, 2, [("fold", ("FOLD", 1, 7, 2)), ("call", ("SHOW", 7, 7, False)), ("raise21", 4), ("shove", 7)])
NODES[3] = (0, 2, 7, [("fold", ("FOLD", 0, 2, 7)), ("call", ("SHOW", 7, 7, False)), ("raise21", 5), ("shove", 6)])
# Fix node 6 & 7: if BTN shoves (node 6), BB faces 200 (inv_btn=200, inv_bb=7 or 21, wait: node 6 can be reached from 7 or 21! Split 6a/6b or use general inv_bb)
# Even cleaner: store invested_opp in terminal FOLD directly, and split node 6 into 6 (vs open/3bet, BB inv=2/7) or 60 (BB inv=2), 61 (BB inv=7), 62 (BB inv=21), 70 (BTN inv=2), 71 (BTN inv=7)
NODES = {
    0: (0, [("fold", ("FOLD", 0, 1, 2)), ("call", 1), ("raise7", 2), ("shove", 60)]),
    1: (1, [("check", ("SHOW", 2, 2, False)), ("raise7", 3), ("shove", 70)]),
    2: (1, [("fold", ("FOLD", 1, 7, 2)), ("call", ("SHOW", 7, 7, False)), ("raise21", 4), ("shove", 71)]),
    3: (0, [("fold", ("FOLD", 0, 2, 7)), ("call", ("SHOW", 7, 7, False)), ("raise21", 5), ("shove", 61)]),
    4: (0, [("fold", ("FOLD", 0, 7, 21)), ("call", ("SHOW", 21, 21, False)), ("shove", 62)]),
    5: (1, [("fold", ("FOLD", 1, 21, 7)), ("call", ("SHOW", 21, 21, False)), ("shove", 72)]),
    60: (1, [("fold", ("FOLD", 1, 200, 2)), ("call", ("SHOW", 200, 200, True))]),
    61: (1, [("fold", ("FOLD", 1, 200, 7)), ("call", ("SHOW", 200, 200, True))]),
    62: (1, [("fold", ("FOLD", 1, 200, 21)), ("call", ("SHOW", 200, 200, True))]),
    70: (0, [("fold", ("FOLD", 0, 2, 200)), ("call", ("SHOW", 200, 200, True))]),
    71: (0, [("fold", ("FOLD", 0, 7, 200)), ("call", ("SHOW", 200, 200, True))]),
    72: (0, [("fold", ("FOLD", 0, 21, 200)), ("call", ("SHOW", 200, 200, True))]),
}


def cfr_eval_node(node, r0, r1, strat, E, real_non_allin, update_player=None, regrets=None, avg_strat=None, weight=1.0):
    """Vectorized CFR+ traversal returning BTN's counterfactual value vector (1755,) or BB's if sign=-1."""
    if isinstance(node, tuple):
        kind = node[0]
        if kind == "FOLD":
            _, folder, inv0, inv1 = node
            # Net to BTN per opp reach: if folder==0 (BTN folds), BTN loses inv0; if folder==1, BTN wins inv1
            pay = -float(inv0) if folder == 0 else float(inv1)
            opp_reach = r1 if update_player == 0 else r0
            return np.full(1755, pay * float(opp_reach.sum()), dtype=np.float64)
        else:
            _, inv, _, is_allin = node
            pot = 2.0 * inv
            rf = 1.0 if is_allin else real_non_allin
            # BTN payoff = inv * (2 * (0.5 + rf*(E - 0.5)) - 1) = pot * rf * (E - 0.5)
            if update_player == 0:
                return (pot * rf) * ((E - 0.5) @ r1)
            else:
                # Return BTN's value from BB's perspective: E.T @ r0
                return (pot * rf) * (((E.T - 0.5) @ r0) * -1.0)

    player, actions = NODES[node]
    sig = strat[node]  # (1755, A)
    A = len(actions)
    vals = np.empty((A, 1755), dtype=np.float64)
    for a_idx, (_, nxt) in enumerate(actions):
        if player == 0:
            vals[a_idx] = cfr_eval_node(nxt, r0 * sig[:, a_idx], r1, strat, E, real_non_allin, update_player, regrets, avg_strat, weight)
        else:
            vals[a_idx] = cfr_eval_node(nxt, r0, r1 * sig[:, a_idx], strat, E, real_non_allin, update_player, regrets, avg_strat, weight)

    node_val = (sig * vals.T).sum(axis=1)
    if regrets is not None and player == update_player:
        # For player 1, values returned are BB's utility (since we negated in SHOW and need to negate FOLD for BB!)
        # Wait: in FOLD above, pay was BTN's payoff! So for update_player == 1, FOLD payoff to BB is -pay.
        reg = vals - node_val[None, :]
        regrets[node] = np.maximum(0.0, regrets[node] + reg.T)
        reach = r0 if player == 0 else r1
        avg_strat[node] += weight * reach[:, None] * sig
    return node_val


def eval_node_signed(node, r0, r1, strat, E, real_non_allin, br_player=None):
    """Returns value to `br_player` (0=BTN, 1=BB); if br_player is set, `br_player` takes best response."""
    if isinstance(node, tuple):
        kind = node[0]
        if kind == "FOLD":
            _, folder, inv0, inv1 = node
            pay0 = -float(inv0) if folder == 0 else float(inv1)
            pay = pay0 if br_player == 0 else -pay0
            opp_reach = r1 if br_player == 0 else r0
            return np.full(1755, pay * float(opp_reach.sum()), dtype=np.float64)
        _, inv, _, is_allin = node
        pot = 2.0 * inv
        rf = 1.0 if is_allin else real_non_allin
        if br_player == 0:
            return (pot * rf) * ((E - 0.5) @ r1)
        return -(pot * rf) * ((E.T - 0.5) @ r0)

    player, actions = NODES[node]
    sig = strat[node]
    A = len(actions)
    vals = np.empty((A, 1755), dtype=np.float64)
    for a_idx, (_, nxt) in enumerate(actions):
        if player == 0:
            vals[a_idx] = eval_node_signed(nxt, r0 * (1.0 if br_player == 0 else sig[:, a_idx]), r1, strat, E, real_non_allin, br_player)
        else:
            vals[a_idx] = eval_node_signed(nxt, r0, r1 * (1.0 if br_player == 1 else sig[:, a_idx]), strat, E, real_non_allin, br_player)
    if player == br_player:
        return vals.max(axis=0)
    return (sig * vals.T).sum(axis=1)


def cfr_step_signed(node, r0, r1, strat, E, real_non_allin, update_player, regrets, avg_strat, weight):
    if isinstance(node, tuple):
        kind = node[0]
        if kind == "FOLD":
            _, folder, inv0, inv1 = node
            pay0 = -float(inv0) if folder == 0 else float(inv1)
            pay = pay0 if update_player == 0 else -pay0
            opp_reach = r1 if update_player == 0 else r0
            return np.full(1755, pay * float(opp_reach.sum()), dtype=np.float64)
        _, inv, _, is_allin = node
        pot = 2.0 * inv
        rf = 1.0 if is_allin else real_non_allin
        if update_player == 0:
            return (pot * rf) * ((E - 0.5) @ r1)
        return -(pot * rf) * ((E.T - 0.5) @ r0)

    player, actions = NODES[node]
    sig = strat[node]
    A = len(actions)
    vals = np.empty((A, 1755), dtype=np.float64)
    for a_idx, (_, nxt) in enumerate(actions):
        if player == 0:
            vals[a_idx] = cfr_step_signed(nxt, r0 * sig[:, a_idx], r1, strat, E, real_non_allin, update_player, regrets, avg_strat, weight)
        else:
            vals[a_idx] = cfr_step_signed(nxt, r0, r1 * sig[:, a_idx], strat, E, real_non_allin, update_player, regrets, avg_strat, weight)
    node_val = (sig * vals.T).sum(axis=1)
    if player == update_player:
        regrets[node] = np.maximum(0.0, regrets[node] + (vals - node_val[None, :]).T)
        reach = r0 if player == 0 else r1
        avg_strat[node] += weight * reach[:, None] * sig
    return node_val


def normalize_strat(arr: np.ndarray) -> np.ndarray:
    s = arr.sum(axis=1, keepdims=True)
    A = arr.shape[1]
    return np.where(s > 1e-12, arr / np.maximum(s, 1e-12), 1.0 / A)


def main():
    parser = argparse.ArgumentParser(description="CFR+ Preflop Blueprint Solver")
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--real", type=float, default=0.95)
    parser.add_argument("--out", type=Path, default=Path("results/preflop_blueprint.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    E, prior = build_equity_matrix(Path("dev/preflop.npz"))
    regrets = {nid: np.zeros((1755, len(acts)), dtype=np.float64) for nid, (_, acts) in NODES.items()}
    avg_strat = {nid: np.zeros((1755, len(acts)), dtype=np.float64) for nid, (_, acts) in NODES.items()}
    cur_strat = {nid: np.full((1755, len(acts)), 1.0 / len(acts), dtype=np.float64) for nid, (_, acts) in NODES.items()}

    expl_hist = []
    for it in range(1, args.iters + 1):
        w = float(it)
        for p in (0, 1):
            cfr_step_signed(0, prior, prior, cur_strat, E, args.real, p, regrets, avg_strat, w)
            for nid, (pl, _) in NODES.items():
                if pl == p:
                    cur_strat[nid] = normalize_strat(regrets[nid])
        if it % 50 == 0 or it == args.iters:
            avg_norm = {nid: normalize_strat(avg_strat[nid]) for nid in NODES}
            br0 = float((prior * eval_node_signed(0, prior, prior, avg_norm, E, args.real, br_player=0)).sum())
            br1 = float((prior * eval_node_signed(0, prior, prior, avg_norm, E, args.real, br_player=1)).sum())
            expl = 0.5 * (br0 + br1)
            expl_hist.append((it, expl, br0, br1))
            print(f"Iter {it:>4d}: exploitability = {expl:.4f} chips/hand (BR_BTN={br0:+.4f}, BR_BB={br1:+.4f}) [{time.perf_counter()-t0:.1f}s]", flush=True)

    # Quantize to uint8 (0..255 summing to 255) and zlib+b64 pack
    avg_norm = {nid: normalize_strat(avg_strat[nid]) for nid in NODES}
    packed = {}
    total_raw_bytes = 0
    for nid, mat in avg_norm.items():
        q = np.floor(mat * 255.0 + 0.5).astype(np.uint8)
        total_raw_bytes += q.nbytes
        comp = zlib.compress(q.tobytes(), level=9)
        packed[str(nid)] = {
            "actions": [a[0] for a in NODES[nid][1]],
            "shape": list(q.shape),
            "b64": base64.b64encode(comp).decode("ascii"),
        }

    out_obj = {
        "iters": args.iters,
        "real_non_allin": args.real,
        "raw_uint8_bytes": total_raw_bytes,
        "exploitability_history": expl_hist,
        "final_exploitability_cph": expl_hist[-1][1],
        "nodes": packed,
    }
    args.out.write_text(json.dumps(out_obj), encoding="utf-8")
    print(f"Saved {args.out} (raw uint8={total_raw_bytes:,} bytes, json={args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

