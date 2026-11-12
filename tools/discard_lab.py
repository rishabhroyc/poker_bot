#!/usr/bin/env python3.11
"""Discard Lab:
  A. 50,000-situation ground-truth equity (uniform + range-aware) & v3 agreement/loss.
  B. Profile _choose_discard & train cheap numpy scoring prior on 45k situations, test on 5k.
  D. Opponent discard inference Total-Variation distance audit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path("dev").resolve()))
from veval import veval  # noqa: E402
import bot as bot_mod  # noqa: E402


KEEPS = ((0, 1), (0, 2), (1, 2))


def extract_keep_features(keep2: np.ndarray, flop3: np.ndarray, bi: bot_mod.BoardInfo) -> np.ndarray:
    """keep2: (M, 2), flop3: (3,) -> (M, F) float64 features."""
    M = len(keep2)
    r0 = keep2[:, 0] >> 2
    r1 = keep2[:, 1] >> 2
    s0 = keep2[:, 0] & 3
    s1 = keep2[:, 1] & 3
    hi = np.maximum(r0, r1) / 12.0
    lo = np.minimum(r0, r1) / 12.0
    pp = (r0 == r1).astype(np.float64)
    suited = (s0 == s1).astype(np.float64)

    br = flop3 >> 2
    bs = flop3 & 3
    # Pair with board
    hit0 = ((r0[:, None] == br[None, :]).any(axis=1)).astype(np.float64)
    hit1 = ((r1[:, None] == br[None, :]).any(axis=1)).astype(np.float64)
    top_b = br.max()
    top_pair = (((r0 == top_b) | (r1 == top_b))).astype(np.float64)
    overpair = (pp * (r0 > top_b)).astype(np.float64)

    # Flush draw / backdoor flush draw
    suit_cnt = np.zeros((M, 4), dtype=np.int64)
    for sc in range(4):
        suit_cnt[:, sc] = (s0 == sc).astype(np.int64) + (s1 == sc).astype(np.int64) + int((bs == sc).sum())
    max_sc = suit_cnt.max(axis=1)
    fd = (max_sc >= 4).astype(np.float64)
    bdfd = (max_sc == 3).astype(np.float64)

    # Made strength from BoardInfo
    ps = bi.pstrength(keep2)
    made = bi.strength(keep2)

    return np.column_stack([
        np.ones(M),
        ps,
        made,
        hi,
        lo,
        pp,
        suited,
        hit0 + hit1,
        top_pair,
        overpair,
        fd,
        bdfd,
    ])


def _worker_chunk(args):
    n_sit, n_gt, seed = args
    rng = np.random.default_rng(seed)
    X_list = []
    y_uni_list = []
    y_ra_list = []
    # Also evaluate n=900 noisy sample on first 500 situations per chunk
    v3_agree_uni = 0
    v3_loss_uni = []
    v3_agree_ra = 0
    v3_loss_ra = []
    v3_checked = 0

    for idx in range(n_sit):
        deal6 = rng.choice(52, size=6, replace=False)
        h3 = np.sort(deal6[:3])
        f3 = deal6[3:]
        rem = np.array([c for c in range(52) if c not in set(deal6.tolist())], dtype=np.int64)

        bi = bot_mod.BoardInfo(f3.tolist())
        keeps = np.array([[h3[i], h3[j]] for i, j in KEEPS], dtype=np.int64)
        feats = extract_keep_features(keeps, f3, bi)  # (3, F)

        # Sample n_gt rollouts (opp2 + turn + river = 4 cards from 46)
        opp_idx = rng.integers(0, len(rem), size=(n_gt, 4))
        # Ensure distinct 4 cards per row by fast random permutation on small batch or rejection
        # Rejection on 4 draws from 46: collision prob ~ 1 - (46*45*44*43)/(46^4) = 12.5%
        good = (
            (opp_idx[:, 0] != opp_idx[:, 1]) & (opp_idx[:, 0] != opp_idx[:, 2]) & (opp_idx[:, 0] != opp_idx[:, 3])
            & (opp_idx[:, 1] != opp_idx[:, 2]) & (opp_idx[:, 1] != opp_idx[:, 3]) & (opp_idx[:, 2] != opp_idx[:, 3])
        )
        while not good.all():
            bad = ~good
            opp_idx[bad] = rng.integers(0, len(rem), size=(int(bad.sum()), 4))
            good[bad] = (
                (opp_idx[bad, 0] != opp_idx[bad, 1]) & (opp_idx[bad, 0] != opp_idx[bad, 2]) & (opp_idx[bad, 0] != opp_idx[bad, 3])
                & (opp_idx[bad, 1] != opp_idx[bad, 2]) & (opp_idx[bad, 1] != opp_idx[bad, 3]) & (opp_idx[bad, 2] != opp_idx[bad, 3])
            )
        cards4 = rem[opp_idx]
        opp2 = cards4[:, :2]
        tr = cards4[:, 2:4]
        fb = np.concatenate([np.tile(f3, (n_gt, 1)), tr], axis=1)  # (n_gt, 5)

        o_sc = veval(np.concatenate([opp2, fb], axis=1))
        eq_uni = np.empty(3, dtype=np.float64)
        # Range-aware weights: opp flop bet/call (~top 50% pstrength)
        opp_ps = bi.pstrength(opp2)
        med = np.median(opp_ps)
        w_ra = 1.0 / (1.0 + np.exp(-np.clip((opp_ps - med) / 0.10, -20.0, 20.0)))
        w_ra /= w_ra.sum()
        eq_ra = np.empty(3, dtype=np.float64)

        for k in range(3):
            mk = np.tile(keeps[k], (n_gt, 1))
            m_sc = veval(np.concatenate([mk, fb], axis=1))
            out = (m_sc > o_sc).astype(np.float64) + 0.5 * (m_sc == o_sc).astype(np.float64)
            eq_uni[k] = out.mean()
            eq_ra[k] = float((out * w_ra).sum())

        X_list.append(feats)
        y_uni_list.append(eq_uni)
        y_ra_list.append(eq_ra)

        # Compare v3's 900-sample budget (or independent 900-sample estimate) on 500 situations per chunk
        if idx < 500:
            v3_checked += 1
            n_v3 = 900
            oi2 = rng.integers(0, len(rem), size=(n_v3, 4))
            g2 = (
                (oi2[:, 0] != oi2[:, 1]) & (oi2[:, 0] != oi2[:, 2]) & (oi2[:, 0] != oi2[:, 3])
                & (oi2[:, 1] != oi2[:, 2]) & (oi2[:, 1] != oi2[:, 3]) & (oi2[:, 2] != oi2[:, 3])
            )
            while not g2.all():
                b2 = ~g2
                oi2[b2] = rng.integers(0, len(rem), size=(int(b2.sum()), 4))
                g2[b2] = (
                    (oi2[b2, 0] != oi2[b2, 1]) & (oi2[b2, 0] != oi2[b2, 2]) & (oi2[b2, 0] != oi2[b2, 3])
                    & (oi2[b2, 1] != oi2[b2, 2]) & (oi2[b2, 1] != oi2[b2, 3]) & (oi2[b2, 2] != oi2[b2, 3])
                )
            c4 = rem[oi2]
            fb2 = np.concatenate([np.tile(f3, (n_v3, 1)), c4[:, 2:4]], axis=1)
            osc2 = veval(np.concatenate([c4[:, :2], fb2], axis=1))
            v3_eq = np.empty(3, dtype=np.float64)
            for k in range(3):
                msc2 = veval(np.concatenate([np.tile(keeps[k], (n_v3, 1)), fb2], axis=1))
                v3_eq[k] = ((msc2 > osc2).astype(np.float64) + 0.5 * (msc2 == osc2).astype(np.float64)).mean()

            k_v3 = int(np.argmax(v3_eq))
            k_gt_u = int(np.argmax(eq_uni))
            k_gt_r = int(np.argmax(eq_ra))
            if k_v3 == k_gt_u:
                v3_agree_uni += 1
            else:
                v3_loss_uni.append(float(eq_uni[k_gt_u] - eq_uni[k_v3]))
            if k_v3 == k_gt_r:
                v3_agree_ra += 1
            else:
                v3_loss_ra.append(float(eq_ra[k_gt_r] - eq_ra[k_v3]))

    return (
        np.array(X_list, dtype=np.float64),
        np.array(y_uni_list, dtype=np.float64),
        np.array(y_ra_list, dtype=np.float64),
        v3_agree_uni,
        v3_loss_uni,
        v3_agree_ra,
        v3_loss_ra,
        v3_checked,
    )


def part_d_inference_audit(n_trials: int = 400) -> dict:
    """Measure Total-Variation distance between true posterior over opp kept pairs and Bot.apply_opp_discard."""
    rng = np.random.default_rng(777)
    results = {}
    for policy in ("best_pstrength", "softmax_0.023", "lowest_rank", "random"):
        tv_cur = []
        tv_soft = []
        for _ in range(n_trials):
            deal = rng.choice(52, size=6, replace=False)
            my_h3 = sorted(deal[:3].tolist())
            flop3 = deal[3:].tolist()
            rem = sorted(set(range(52)) - set(deal.tolist()))
            opp_h3 = sorted(rng.choice(rem, size=3, replace=False).tolist())

            bi = bot_mod.BoardInfo(flop3)
            # True policy probabilities for opponent's 3 discards
            opp_keeps = np.array([[opp_h3[i], opp_h3[j]] for i, j in KEEPS], dtype=np.int64)
            s_opp = bi.pstrength(opp_keeps)
            if policy == "best_pstrength":
                k_choose = int(np.argmax(s_opp))
            elif policy == "softmax_0.023":
                pr = np.exp((s_opp - s_opp.max()) / 0.023)
                pr /= pr.sum()
                k_choose = int(rng.choice(3, p=pr))
            elif policy == "lowest_rank":
                d_card = min(opp_h3)
                k_choose = next(k for k in range(3) if opp_h3[bot_mod.DROP[k]] == d_card)
            else:
                k_choose = int(rng.integers(0, 3))
            opp_disc = opp_h3[bot_mod.DROP[k_choose]]

            # Build Bot and run apply_opp_discard
            b = bot_mod.Bot()
            b.new_hand({"hand_id": 1, "seat": "button", "hole_cards": [bot_mod.I2C[c] for c in my_h3]})
            b.board = list(flop3)
            b.init_range3()
            # Pin opp's disc_logw to the true policy for clean posterior comparison
            if policy in ("best_pstrength", "softmax_0.023"):
                b.opp.disc_logw = np.array([10.0, -10.0, -10.0])
            elif policy == "lowest_rank":
                b.opp.disc_logw = np.array([-10.0, 10.0, -10.0])
            else:
                b.opp.disc_logw = np.array([-10.0, -10.0, 10.0])

            # True posterior over pairs containingopp_disc removed from combos containing opp_disc
            combos, w_prior = b.range3
            has_d = (combos == opp_disc).any(axis=1)
            c_sub = combos[has_d]
            w_sub = w_prior[has_d]
            s0 = bi.pstrength(c_sub[:, [0, 1]])
            s1 = bi.pstrength(c_sub[:, [0, 2]])
            s2 = bi.pstrength(c_sub[:, [1, 2]])
            s_all = np.stack([s0, s1, s2], axis=1)

            if policy == "best_pstrength":
                bk = np.argmax(s_all, axis=1)
                dc = c_sub[np.arange(len(c_sub)), np.array([bot_mod.DROP[k] for k in bk])]
                true_lk = (dc == opp_disc).astype(np.float64)
            elif policy == "softmax_0.023":
                pr_all = np.exp((s_all - s_all.max(axis=1, keepdims=True)) / 0.023)
                pr_all /= pr_all.sum(axis=1, keepdims=True)
                # Probability of discarding opp_disc is pr_all[:, k] where DROP[k] holds opp_disc
                d_pos = np.argmax(c_sub == opp_disc, axis=1)  # 0, 1, or 2
                # DROP = (2, 1, 0): keep 0 -> drop 2, keep 1 -> drop 1, keep 2 -> drop 0
                keep_idx = 2 - d_pos
                true_lk = pr_all[np.arange(len(c_sub)), keep_idx]
            elif policy == "lowest_rank":
                true_lk = (c_sub.min(axis=1) == opp_disc).astype(np.float64)
            else:
                true_lk = np.full(len(c_sub), 1.0 / 3.0)

            p_true = w_sub * true_lk
            if p_true.sum() > 0:
                p_true /= p_true.sum()
            else:
                p_true = w_sub / w_sub.sum()

            b.apply_opp_discard(opp_disc)
            _, w_bot = b.range2
            tv_cur.append(0.5 * float(np.abs(p_true - w_bot).sum()))

            # Softmax-corrected likelihood
            pr_soft = np.exp((s_all - s_all.max(axis=1, keepdims=True)) / 0.023)
            pr_soft /= pr_soft.sum(axis=1, keepdims=True)
            d_pos = np.argmax(c_sub == opp_disc, axis=1)
            p1_soft = pr_soft[np.arange(len(c_sub)), 2 - d_pos]
            p2 = (c_sub.min(axis=1) == opp_disc).astype(np.float64)
            p3 = np.full(len(c_sub), 1.0 / 3.0)
            mix = b.opp.disc_mix()
            lk_soft = (p1_soft * mix[0] + p2 * mix[1] + p3 * mix[2]) * 0.90 + 0.10 / 3.0
            w_soft = w_sub * lk_soft
            w_soft /= w_soft.sum()
            tv_soft.append(0.5 * float(np.abs(p_true - w_soft).sum()))

        results[policy] = {
            "tv_current_mean": float(np.mean(tv_cur)),
            "tv_softmax_mean": float(np.mean(tv_soft)),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Discard Lab (Parts A, B, D)")
    parser.add_argument("--situations", type=int, default=50_000)
    parser.add_argument("--gt-samples", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()

    t0 = time.perf_counter()
    per_job = args.situations // args.jobs
    tasks = [(per_job, args.gt_samples, 1000 + i) for i in range(args.jobs)]
    print(f"Part A/B: simulating {per_job * args.jobs:,} situations ({args.gt_samples} rollouts each) on {args.jobs} workers...")

    Xs, Yus, Yrs = [], [], []
    ag_u, ls_u, ag_r, ls_r, chk = 0, [], 0, [], 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_worker_chunk, t) for t in tasks]
        for fut in as_completed(futs):
            X, yu, yr, au, lu, ar, lr, c = fut.result()
            Xs.append(X)
            Yus.append(yu)
            Yrs.append(yr)
            ag_u += au
            ls_u.extend(lu)
            ag_r += ar
            ls_r.extend(lr)
            chk += c

    X_all = np.concatenate(Xs, axis=0)  # (N, 3, F)
    Yu_all = np.concatenate(Yus, axis=0)  # (N, 3)
    Yr_all = np.concatenate(Yrs, axis=0)  # (N, 3)
    N = len(X_all)

    # Train linear scoring prior on first 90% (45k), test on last 10% (5k)
    n_tr = int(0.9 * N)
    X_tr = X_all[:n_tr].reshape(-1, X_all.shape[2])
    # Predict centered equity (keep - mean_3_keeps) so ranking is sharp
    y_tr = (Yu_all[:n_tr] - Yu_all[:n_tr].mean(axis=1, keepdims=True)).reshape(-1)
    # Ridge regression
    lam = 1.0
    XtX = X_tr.T @ X_tr + lam * np.eye(X_tr.shape[1])
    w_prior = np.linalg.solve(XtX, X_tr.T @ y_tr)

    X_te = X_all[n_tr:]  # (N_te, 3, F)
    Yu_te = Yu_all[n_tr:]
    pred_te = X_te @ w_prior  # (N_te, 3)
    pred_order = np.argsort(-pred_te, axis=1)
    gt_best = np.argmax(Yu_te, axis=1)
    prior_agree = float((pred_order[:, 0] == gt_best).mean())

    # Margin calibration: when top1 - top2 > margin, how often does prior agree?
    gaps = pred_te[np.arange(len(pred_te)), pred_order[:, 0]] - pred_te[np.arange(len(pred_te)), pred_order[:, 1]]
    margin_stats = []
    for thr in (0.01, 0.02, 0.03, 0.04, 0.05):
        m = gaps >= thr
        if m.sum() > 0:
            acc = float((pred_order[m, 0] == gt_best[m]).mean())
            loss = float((Yu_te[m, gt_best[m]] - Yu_te[m, pred_order[m, 0]]).mean())
            margin_stats.append({"threshold": thr, "skip_frac": float(m.mean()), "agree": acc, "mean_eq_loss": loss})

    # Part D: inference audit
    inf_res = part_d_inference_audit()

    out = {
        "situations": N,
        "gt_samples": args.gt_samples,
        "wall_s": round(time.perf_counter() - t0, 1),
        "part_a": {
            "v3_checked": chk,
            "uniform_agree_pct": 100.0 * ag_u / max(1, chk),
            "uniform_disagree_mean_loss": float(np.mean(ls_u)) if ls_u else 0.0,
            "range_aware_agree_pct": 100.0 * ag_r / max(1, chk),
            "range_aware_disagree_mean_loss": float(np.mean(ls_r)) if ls_r else 0.0,
        },
        "part_b": {
            "prior_weights": w_prior.tolist(),
            "prior_test_agree_pct": 100.0 * prior_agree,
            "margin_table": margin_stats,
        },
        "part_d": inf_res,
    }
    Path("results/discard_lab.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

