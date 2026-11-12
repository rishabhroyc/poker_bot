#!/usr/bin/env python3
"""Autonomous Discard Hold'em Competition Bot (v2 Strategic Upgrade).
Combines Bayesian range tracking, one-step EV decisions, discard style learning,
strict protocol channel isolation, and robust time budgeting.

v2 Enhancements:
  1. Multi-street calling discipline & risk margin against large bets/shoves.
  2. Geometric value-bet sizing (no premature flop/turn all-in jams).
  3. Preflop 3-bet/4-bet gating (no bluff-jamming with weak hands).
  4. Turn barreling discipline with marginal showdown value.
    5. Enhanced aggression & bet-sizing informativeness priors in OppModel.

v3 additions:
  * Time left is read from time_left_ms (or time_bank_ms) AND tracked from our own
    measured spend, taking the smaller; hands left counted from hand_result.
  * Opponent statistics fade old evidence (half-life scales with leg length).
  * hand_result board read from several key names, falling back to the board
    seen during the hand; showdown learning is skipped if no full board exists.
  * Exception fallback uses pot odds and a rough hand-strength estimate.
"""

import os
import sys
import json
import random
import time
from itertools import combinations
from math import comb
import numpy as np

# Thread limitation for NumPy/OpenBLAS to keep latency predictable
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from veval import veval  # noqa: E402

# Swap stdout to protect protocol channel from accidental prints
_PROTO_OUT = sys.stdout
sys.stdout = sys.stderr

RANKS = "23456789TJQKA"
SUITS = "shdc"
C2I = {r + s: i * 4 + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}
I2C = {v: k for k, v in C2I.items()}
STACK = 200

_LOG_BYTES_WRITTEN = 0
_MAX_LOG_BYTES = 32768  # 32 KB hard cap

def log(msg):
    global _LOG_BYTES_WRITTEN
    if _LOG_BYTES_WRITTEN >= _MAX_LOG_BYTES:
        return
    text = str(msg) + "\n"
    b_len = len(text.encode("utf-8", "replace"))
    if _LOG_BYTES_WRITTEN + b_len > _MAX_LOG_BYTES:
        sys.stderr.write("[LOG LIMIT REACHED]\n")
        _LOG_BYTES_WRITTEN = _MAX_LOG_BYTES
        return
    sys.stderr.write(text)
    sys.stderr.flush()
    _LOG_BYTES_WRITTEN += b_len

def send_proto(obj: dict) -> None:
    data = json.dumps(obj, separators=(",", ":"))
    _PROTO_OUT.write(data + "\n")
    _PROTO_OUT.flush()

# ---------------------------------------------------------------- Clock manager
class Clock:
    def __init__(self, bank_ms: float = 30000.0, total_hands: int = 200):
        self.bank_ms = float(bank_ms)
        self.total_hands = int(total_hands)

    def update_init(self, bank_ms: float, total_hands: int) -> None:
        self.bank_ms = float(bank_ms)
        self.total_hands = int(total_hands)

    def budget_ms(self, time_left_ms: float, hand_id: int, weight: float = 1.0) -> float:
        hands_left = max(1, self.total_hands - hand_id + 1)
        reserve = max(2500.0, 0.12 * self.bank_ms)
        usable = max(0.0, float(time_left_ms) - reserve)
        per_hand = usable / hands_left
        return min(per_hand * weight, 1000.0)

# ---------------------------------------------------------------- params loader
def load_params(path=None):
    if path is None:
        path = os.path.join(HERE, "params.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

DEFAULT_PARAMS = load_params()

# ---------------------------------------------------------------- static tables
POP13 = np.array([bin(i).count("1") for i in range(1 << 13)], dtype=np.int64)
WINDOWS = [((1 << 5) - 1) << lo for lo in range(9)] + [(1 << 12) | 0b1111]
ALL3 = np.array(list(combinations(range(52), 3)), dtype=np.int64)          # 22100
ALL2 = np.array(list(combinations(range(52), 2)), dtype=np.int64)          # 1326
KEEPS = ((0, 1), (0, 2), (1, 2))
DROP = (2, 1, 0)  # card dropped by each keep

def colex3(a):
    return a[:, 0] + a[:, 1] * (a[:, 1] - 1) // 2 + a[:, 2] * (a[:, 2] - 1) * (a[:, 2] - 2) // 6

PRE = np.load(os.path.join(HERE, "preflop.npz"))
PF_CLASS = PRE["combo_class"].astype(np.int64)
PF_EQ = PRE["eq"].astype(np.float64)
PF_EQDEC = PRE["eq_dec"].astype(np.float64)
_eq_combo = PF_EQ[PF_CLASS]
_order = np.argsort(_eq_combo, kind="stable")
PF_PCT = np.empty(len(_eq_combo), dtype=np.float64)
PF_PCT[_order] = np.arange(len(_eq_combo), dtype=np.float64) / (len(_eq_combo) - 1)
DEC_MID = np.linspace(0.05, 0.95, 10)

def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def wquantile(values, weights, q):
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cum = np.cumsum(weights_sorted)
    target = q * cum[-1]
    idx = np.searchsorted(cum, target)
    return values_sorted[min(idx, len(values_sorted) - 1)]

# ---------------------------------------------------------------- hand strength
class BoardInfo:
    def __init__(self, board, params=None):
        self.params = params or DEFAULT_PARAMS
        self.board = list(board)
        rest = np.array([c for c in range(52) if c not in board], dtype=np.int64)
        pairs = np.array(list(combinations(rest, 2)), dtype=np.int64)
        bt = np.tile(np.array(board, dtype=np.int64), (len(pairs), 1))
        self.ref = np.sort(veval(np.concatenate([pairs, bt], axis=1)))
        self.pairs_all = pairs
        self._lut = None

    def strength(self, pairs):
        b = self.board
        n = len(pairs)
        cards = np.concatenate([pairs, np.tile(np.array(b, dtype=np.int64), (n, 1))], axis=1)
        made = veval(cards)
        lo = np.searchsorted(self.ref, made, side="left")
        hi = np.searchsorted(self.ref, made, side="right")
        pct = (lo + hi) / (2.0 * len(self.ref))
        if len(b) >= 5:
            return pct
        suits = cards & 3
        suit_max = np.max(np.stack([(suits == s).sum(axis=1) for s in range(4)]), axis=0)
        cat = made >> 20
        fd = (suit_max == 4) & (cat < 5)
        mask = np.bitwise_or.reduce(np.left_shift(1, cards >> 2), axis=1)
        outs = np.zeros(n, dtype=np.int64)
        for w in WINDOWS:
            have = POP13[mask & w]
            outs |= np.where(have == 4, w & ~mask, 0)
        nout = POP13[outs & 0x1FFF]
        oesd = (nout >= 2) & (cat < 4)
        gut = (nout == 1) & (cat < 4)
        sp = self.params.get("strength", {})
        k = 1.0 if len(b) == 3 else sp.get("turn_factor", 0.55)
        bonus = k * (sp.get("draw_bonus_fd", 0.30) * fd + sp.get("draw_bonus_oesd", 0.20) * oesd + sp.get("draw_bonus_gut", 0.07) * gut)
        return np.minimum(1.0, pct + (1.0 - pct) * bonus)

    def lut(self):
        """52x52 table of strength(pair) for every pair that avoids the board."""
        if self._lut is None:
            st = self.strength(self.pairs_all)
            L = np.zeros((52, 52))
            L[self.pairs_all[:, 0], self.pairs_all[:, 1]] = st
            L[self.pairs_all[:, 1], self.pairs_all[:, 0]] = st
            self._lut = L
        return self._lut

    def pstrength(self, pairs):
        L = self.lut()
        return L[pairs[:, 0], pairs[:, 1]]

# ---------------------------------------------------------------- opponent model
class Stat:
    def __init__(self, p0, n0=6.0):
        self.p0 = float(p0)
        self.n0 = float(n0)
        self.a = self.p0 * self.n0
        self.n = self.n0
        self.decay = 1.0   # <1: old evidence fades toward the prior

    def add(self, hit, w=1.0):
        if self.decay < 1.0:
            b = self.p0 * self.n0
            self.a = b + (self.a - b) * self.decay
            self.n = self.n0 + (self.n - self.n0) * self.decay
        self.a += float(hit) * float(w)
        self.n += float(w)

    @property
    def p(self):
        return self.a / self.n

class OppModel:
    def __init__(self, params=None):
        p = params or DEFAULT_PARAMS
        opp_p = p.get("opponent_priors", {})
        n0 = opp_p.get("n0", 6.0)
        info_p = p.get("informativeness", {})

        self.sb_open_raise = Stat(opp_p.get("sb_open_raise", 0.55), n0)
        self.sb_open_fold = Stat(opp_p.get("sb_open_fold", 0.10), n0)
        self.bb_raise_vs_limp = Stat(opp_p.get("bb_raise_vs_limp", 0.30), n0)
        self.fold_vs_raise_pf = Stat(opp_p.get("fold_vs_raise_pf", 0.35), n0)
        self.reraise_vs_raise_pf = Stat(opp_p.get("reraise_vs_raise_pf", 0.20), n0)
        self.bet_when_checked = Stat(opp_p.get("bet_when_checked", 0.45), n0)
        self.fold_vs_bet = Stat(opp_p.get("fold_vs_bet", 0.40), n0)
        self.raise_vs_bet = Stat(opp_p.get("raise_vs_bet", 0.12), n0)
        self.bluff_share = Stat(opp_p.get("bluff_share", 0.20), n0)

        # Enhanced prior on raise informativeness: opponent raises represent strong hands
        sd_r = info_p.get("sd_raise_prior", [6.0, 2.0])
        self.sd_raise = [float(sd_r[0]), float(sd_r[1])]
        sd_p = info_p.get("sd_pass_prior", [2.0, 3.0])
        self.sd_pass = [float(sd_p[0]), float(sd_p[1])]
        self.gap_scale = float(info_p.get("gap_scale", 0.18))

        disc_w = np.array(info_p.get("disc_logw", [0.6, 0.2, 0.2]), dtype=np.float64)
        self.disc_logw = np.log(disc_w)
        self.hands = 0
        self.net = 0
        # bet-size -> showdown-strength relationship: higher prior so big bets narrow range
        sz_p = p.get("size_model", {})
        self.size_prior = float(sz_p.get("info_prior", 0.40))
        self.size_gap_scale = float(sz_p.get("gap_scale", 0.20))
        self.size_split = float(sz_p.get("split_frac", 0.55))
        self.sz_small = [0.55 * 3.0, 3.0]
        self.sz_large = [0.75 * 3.0, 3.0]
        # recency weighting: evidence fades with a half-life tied to leg length
        self.rec_p = p.get("recency", {})
        self.sd_raise_prior = list(self.sd_raise)
        self.sd_pass_prior = list(self.sd_pass)
        self.sz_small_prior = list(self.sz_small)
        self.sz_large_prior = list(self.sz_large)
        self.sd_decay = 1.0
        self.set_horizon(200)

    def set_horizon(self, total_hands):
        r = self.rec_p
        hl = max(float(r.get("half_life_min", 30.0)), float(r.get("half_life_frac", 0.25)) * total_hands)
        hl_sd = max(float(r.get("sd_half_life_min", 15.0)), float(r.get("sd_half_life_frac", 0.12)) * total_hands)
        d = 0.5 ** (1.0 / hl)
        self.sd_decay = 0.5 ** (1.0 / hl_sd)
        for v in vars(self).values():
            if isinstance(v, Stat):
                v.decay = d

    def fade_add(self, lst, prior, x, w=1.0):
        d = self.sd_decay
        lst[0] = prior[0] + (lst[0] - prior[0]) * d + x
        lst[1] = prior[1] + (lst[1] - prior[1]) * d + w

    @property
    def aggr_info(self):
        gap = self.sd_raise[0] / self.sd_raise[1] - self.sd_pass[0] / self.sd_pass[1]
        return min(1.0, max(0.40, gap / self.gap_scale))

    @property
    def size_info(self):
        """0..1: how much opponent bet size reveals hand strength."""
        gap = self.sz_large[0] / self.sz_large[1] - self.sz_small[0] / self.sz_small[1]
        return min(1.0, max(0.0, self.size_prior + gap / self.size_gap_scale))

    def disc_mix(self):
        w = np.exp(self.disc_logw - self.disc_logw.max())
        return w / w.sum()

# ---------------------------------------------------------------- the bot
class Bot:
    def __init__(self, params=None):
        self.params = params or DEFAULT_PARAMS
        self.opp = OppModel(self.params)
        self.clock = Clock()
        self.rng = np.random.default_rng(20261111)
        self.hand_id = 0
        self.hole0 = []
        self.hole = []
        self.board = []
        self.board_full = []
        self.seat = 0
        self.total_hands = 200
        self.time_bank = 30000.0
        self.hands_played = 0
        self.spent_s = 0.0

    def on_init(self, m):
        self.clock.update_init(m.get("time_bank_ms", 30000.0), m.get("total_hands", 200))
        self.time_bank = float(m.get("time_bank_ms", 30000.0))
        self.total_hands = int(m.get("total_hands", 200))
        self.opp.set_horizon(self.total_hands)

    def new_hand(self, m):
        self.hand_id = m["hand_id"]
        self.seat = m["seat"]
        self.hole0 = [C2I[c] for c in m["hole_cards"]]
        self.hole = list(self.hole0)
        self.board = []
        self.board_full = []
        self.my_discard = None
        self.opp_discard = None
        self.hist_seen = 0
        self.opp_pf_acts = []
        self.range3 = None
        self.range2 = None
        self.binfo = {}
        self.street_ctx = {}
        self.we_bet_last = False
        self.pending_aggr = []
        self.opp_sizes = {}
        self.cur_pot = 3

    def hands_left(self):
        return max(1, self.total_hands - self.hands_played)

    def time_left_ms(self, req):
        """Smaller of: what the engine reports (time_left_ms / time_bank_ms) and
        our own estimate (bank minus measured decision time)."""
        vals = [self.time_bank - 1000.0 * self.spent_s]
        for k in ("time_left_ms", "time_bank_ms"):
            v = req.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(float(v))
        return min(vals)

    def budget_scale(self, req):
        left = self.time_left_ms(req)
        reserve = max(2500.0, 0.12 * self.time_bank)
        usable = max(0.0, left - reserve)
        budget = usable / self.hands_left()
        nominal = (self.time_bank - reserve) / max(1, self.total_hands)
        return min(1.5, max(0.2, budget / max(nominal, 1.0)))

    def extra_time_s(self, req, weight):
        left = self.time_left_ms(req)
        reserve = max(2500.0, 0.12 * self.time_bank)
        per_hand = max(0.0, left - reserve) / self.hands_left()
        return min(per_hand * weight, 1000.0) / 1000.0

    def size_mult(self, sz):
        if sz is None:
            return 1.0
        sp = self.params.get("size_model", {})
        ref = sp.get("ref_frac", 0.6)
        pw = sp.get("power", 0.5)
        raw = (ref / max(sz, 0.15)) ** pw
        raw = min(1.8, max(0.35, raw))
        return 1.0 + self.opp.size_info * (raw - 1.0)

    def dead(self):
        d = set(self.hole0) | set(self.board)
        if self.opp_discard is not None:
            d.add(self.opp_discard)
        return d

    def board_info(self, board):
        key = tuple(board)
        bi = self.binfo.get(key)
        if bi is None:
            bi = BoardInfo(board, self.params)
            self.binfo[key] = bi
        return bi

    def init_range3(self):
        dead = self.dead()
        mask = np.isin(ALL3, list(dead)).any(axis=1)
        combos = ALL3[~mask]
        pct = PF_PCT[colex3(combos)]
        rho = np.ones(10)
        for act in self.opp_pf_acts:
            rho *= self.pf_likelihood(act, DEC_MID)
        rho /= rho.sum()
        dec = np.clip(np.floor(pct * 10).astype(int), 0, 9)
        w = rho[dec]
        self.range3 = (combos, w / w.sum())

    def pf_likelihood(self, act, pct):
        a, ctx = act
        lk = self.params.get("likelihood", {})
        w = lk.get("pf_width", 0.08)
        base = self._pf_likelihood(a, ctx, pct, w)
        fl = lk.get("pf_raise_floor", 0.05) if a == "raise" else lk.get("pf_call_floor", 0.04)
        return np.maximum(base, fl)

    def _pf_likelihood(self, a, ctx, pct, w):
        o = self.opp
        lk = self.params.get("likelihood", {})
        if ctx == "open":
            if a == "raise":
                f = o.sb_open_raise.p
                return sig((pct - (1.0 - f)) / w)
            if a == "call":
                b = lk.get("pf_call_open_base", 0.30)
                m = lk.get("pf_call_open_mult", 0.70)
                supp = lk.get("pf_call_open_suppress", 0.60)
                f = min(0.95, max(0.05, b + m * (1.0 - o.sb_open_raise.p - o.sb_open_fold.p)))
                mid = 1.0 - o.sb_open_raise.p - f / 2
                return np.maximum(0.0, 1.0 - supp * np.abs(pct - mid) / max(f, 0.1))
            if a == "fold":
                return sig(((1.0 - o.sb_open_fold.p) - pct) / w)
        if ctx == "vs_raise":
            if a == "raise":
                return sig((pct - (1.0 - o.reraise_vs_raise_pf.p)) / w)
            if a == "fold":
                return sig(((1.0 - o.fold_vs_raise_pf.p) - pct) / w)
            return sig((pct - (1.0 - o.fold_vs_raise_pf.p)) / w) * sig(((1.0 - o.reraise_vs_raise_pf.p) - pct) / w)
        if ctx == "vs_limp":
            if a == "raise":
                return sig((pct - (1.0 - o.bb_raise_vs_limp.p)) / w)
            return 1.0 - lk.get("pf_check_suppress", 0.75) * sig((pct - (1.0 - o.bb_raise_vs_limp.p)) / w)
        return np.ones_like(pct)

    def strengths3(self):
        combos, _ = self.range3
        bi = self.board_info(self.board[:3])
        s0 = bi.pstrength(combos[:, [0, 1]])
        s1 = bi.pstrength(combos[:, [0, 2]])
        s2 = bi.pstrength(combos[:, [1, 2]])
        return np.stack([s0, s1, s2], axis=1)

    def current_range(self):
        if self.range2 is not None:
            pairs, w = self.range2
            if self.board:
                clash = np.isin(pairs, self.board).any(axis=1)
                if clash.any():
                    pairs, w = pairs[~clash], w[~clash]
                    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / max(1, len(w)))
                    self.range2 = (pairs, w)
            s = self.board_info(self.board).pstrength(pairs)
            return pairs, w, s
        combos, w = self.range3
        s = self.strengths3().max(axis=1)
        return combos, w, s

    def action_likelihood(self, action, s, w, facing, size_frac, opp_sz=None):
        lk = self._action_likelihood(action, s, w, facing, size_frac, opp_sz)
        if action in ("raise", "check"):
            rho = self.opp.aggr_info
            lk = (1.0 - rho) * float((lk * w).sum() / max(w.sum(), 1e-12)) + rho * lk
        return lk

    def _action_likelihood(self, action, s, w, facing, size_frac, opp_sz=None):
        o = self.opp
        lk = self.params.get("likelihood", {})
        if action == "raise" and not facing:
            f = min(0.95, max(0.02, o.bet_when_checked.p * self.size_mult(opp_sz)))
            th = wquantile(s, w, 1.0 - f)
            bl_base = lk.get("bluff_floor_base", 0.05)
            bl_share = lk.get("bluff_floor_share", 0.20)
            bl_thresh = lk.get("bluff_strength_threshold", 0.25)
            w_open = lk.get("postflop_open_width", 0.06)
            eps = lk.get("postflop_open_eps", 0.02)
            bl = bl_base + bl_share * o.bluff_share.p
            return bl * (s < bl_thresh) + (1 - bl) * sig((s - th) / w_open) + eps
        if action == "check":
            f = o.bet_when_checked.p
            th = wquantile(s, w, 1.0 - f)
            supp = lk.get("postflop_check_suppress", 0.85)
            w_chk = lk.get("postflop_open_width", 0.06)
            return 1.0 - supp * sig((s - th) / w_chk)
        if action == "raise" and facing:
            f = min(0.95, max(0.02, o.raise_vs_bet.p * self.size_mult(opp_sz)))
            th = wquantile(s, w, 1.0 - f)
            w_raise = lk.get("postflop_raise_width", 0.05)
            base_raise = lk.get("postflop_raise_base", 0.02)
            return base_raise + sig((s - th) / w_raise)
        if action == "call":
            f = o.fold_vs_bet.p * (0.75 + 0.5 * min(size_frac, 1.5) / 1.5)
            th = wquantile(s, w, min(0.95, f))
            w_call = lk.get("postflop_call_width", 0.07)
            base_call = lk.get("postflop_call_base", 0.03)
            return base_call + sig((s - th) / w_call)
        return np.ones_like(s)

    def process_history(self, history, stats_only=False):
        for k, h in enumerate(history):
            if k < self.hist_seen:
                continue
            st, actor, a = h.get("street"), h.get("actor"), h.get("action")
            ctx = self.street_ctx.get(st)
            if ctx is None:
                ctx = {"bet": 2 if st == "preflop" else 0, "opp_bet": 2 if st == "preflop" else 0,
                       "my_bet": 1 if st == "preflop" else 0,
                       "pot": 3 if st == "preflop" else self.cur_pot}
                self.street_ctx[st] = ctx
            if st == "preflop" and "init" not in ctx:
                ctx["init"] = True
                ctx["my_bet"], ctx["opp_bet"] = (1, 2) if self.seat == 0 else (2, 1)
            facing = (ctx["opp_bet"] < ctx["my_bet"]) if actor == "opp" else (ctx["my_bet"] < ctx["opp_bet"])
            amt = h.get("amount")
            opp_sz = None
            if actor == "opp" and a == "raise" and st != "preflop" and amt is not None:
                call_amt = max(0, ctx["my_bet"] - ctx["opp_bet"])
                raise_inc = amt - max(ctx["my_bet"], ctx["opp_bet"])
                opp_sz = max(0.0, raise_inc) / max(1.0, ctx["pot"] + call_amt)
                self.opp_sizes.setdefault(st, []).append(opp_sz)
            if actor == "opp":
                if st == "preflop":
                    n_prev_opp = sum(1 for x in history[:k] if x.get("street") == "preflop" and x.get("actor") == "opp")
                    any_raise_before = any(x.get("street") == "preflop" and x.get("action") == "raise" for x in history[:k])
                    if self.seat == 1 and n_prev_opp == 0 and not any_raise_before:
                        pctx = "open"
                    elif self.seat == 0 and not any_raise_before and n_prev_opp == 0:
                        pctx = "vs_limp"
                    else:
                        pctx = "vs_raise"
                    self.record_pf_stat(a, pctx)
                    if a in ("raise", "call", "check"):
                        self.opp_pf_acts.append((a, pctx))
                elif stats_only:
                    self.record_post_stat(a, facing)
                elif (self.range3 is not None or self.range2 is not None
                      or (len(self.board) >= 3 and self._init_range_now())):
                    cards, w, s = self.current_range() if st == self.cur_street_name() else self.range_for_street(st)
                    size = 0.0
                    if facing and ctx["pot"] > 0:
                        size = (ctx["my_bet"] - ctx["opp_bet"]) / max(1, ctx["pot"] - (ctx["my_bet"] - ctx["opp_bet"]))
                    if a == "raise":
                        f = self.opp.raise_vs_bet.p if facing else self.opp.bet_when_checked.p
                        lk_full = self._action_likelihood(a, s, w, facing, size, opp_sz)
                        wn = w / w.sum()
                        e0 = float((s * wn).sum())
                        wf = wn * lk_full
                        e1 = float((s * wf).sum() / max(wf.sum(), 1e-12))
                        self.pending_aggr.append((st, e0, e1))
                    lk = self.action_likelihood(a, s, w, facing, size, opp_sz) if a != "fold" else np.ones_like(s)
                    self.record_post_stat(a, facing)
                    nw = w * lk
                    if nw.sum() > 1e-12:
                        nw /= nw.sum()
                        if self.range2 is not None and st in ("turn", "river"):
                            self.range2 = (self.range2[0], nw)
                        elif self.range3 is not None and st == "flop":
                            self.range3 = (self.range3[0], nw)
            if a == "call":
                if actor == "opp":
                    ctx["pot"] += ctx["my_bet"] - ctx["opp_bet"]
                    ctx["opp_bet"] = ctx["my_bet"]
                else:
                    ctx["pot"] += ctx["opp_bet"] - ctx["my_bet"]
                    ctx["my_bet"] = ctx["opp_bet"]
            elif a == "raise":
                if actor == "opp":
                    ctx["pot"] += amt - ctx["opp_bet"]
                    ctx["opp_bet"] = amt
                else:
                    ctx["pot"] += amt - ctx["my_bet"]
                    ctx["my_bet"] = amt
            self.cur_pot = ctx["pot"]
        self.hist_seen = len(history)

    def _init_range_now(self):
        if self.range3 is None:
            self.init_range3()
        return True

    def cur_street_name(self):
        b = len(self.board)
        return "preflop" if b == 0 else ("flop" if b == 3 else ("turn" if b == 4 else "river"))

    def range_for_street(self, st):
        if st in ("turn", "river") and self.range2 is not None:
            return self.range2[0], self.range2[1], self.board_info(self.board).pstrength(self.range2[0])
        return self.range3[0], self.range3[1], self.strengths3().max(axis=1)

    def record_pf_stat(self, a, ctx):
        o = self.opp
        if ctx == "open":
            o.sb_open_raise.add(a == "raise")
            o.sb_open_fold.add(a == "fold")
        elif ctx == "vs_raise":
            o.fold_vs_raise_pf.add(a == "fold")
            o.reraise_vs_raise_pf.add(a == "raise")
        elif ctx == "vs_limp":
            o.bb_raise_vs_limp.add(a == "raise")

    def record_post_stat(self, a, facing):
        o = self.opp
        if facing:
            o.fold_vs_bet.add(a == "fold")
            o.raise_vs_bet.add(a == "raise")
        else:
            o.bet_when_checked.add(a == "raise")

    def apply_opp_discard(self, d):
        self.opp_discard = d
        if self.range3 is None:
            self.init_range3()
        combos, w = self.range3
        has_d = (combos == d).any(axis=1)
        combos = combos[has_d]
        w = w[has_d]
        bi = self.board_info(self.board[:3])
        s0 = bi.pstrength(combos[:, [0, 1]])
        s1 = bi.pstrength(combos[:, [0, 2]])
        s2 = bi.pstrength(combos[:, [1, 2]])
        s = np.stack([s0, s1, s2], axis=1)
        best_k = np.argmax(s, axis=1)
        disc_k = np.array([DROP[k] for k in best_k])
        disc_cards = combos[np.arange(len(combos)), disc_k]
        p1 = (disc_cards == d).astype(np.float64)
        p2 = (combos.min(axis=1) == d).astype(np.float64)
        p3 = np.full(len(combos), 1.0 / 3.0)
        mix = self.opp.disc_mix()
        lk = p1 * mix[0] + p2 * mix[1] + p3 * mix[2]
        dp = self.params.get("discard", {})
        blend = dp.get("discard_blend_weight", 0.90)
        lk = lk * blend + (1.0 - blend) / 3.0
        nw = w * lk
        pairs = np.array([[c for c in row if c != d] for row in combos], dtype=np.int64)
        tot = nw.sum()
        if tot <= 1e-12:
            nw = w
            tot = nw.sum()
        self.range2 = (pairs, nw / tot)

    def learn_discard_style(self, opp_two, opp_disc, flop):
        h3 = sorted(opp_two + [opp_disc])
        combos = np.array([h3], dtype=np.int64)
        bi = self.board_info(flop)
        s0 = float(bi.pstrength(combos[:, [0, 1]])[0])
        s1 = float(bi.pstrength(combos[:, [0, 2]])[0])
        s2 = float(bi.pstrength(combos[:, [1, 2]])[0])
        best_k = int(np.argmax([s0, s1, s2]))
        p1 = 1.0 if DROP[best_k] == h3.index(opp_disc) else 0.0
        blend = self.params.get("discard", {}).get("discard_blend_weight", 0.90)
        p2 = 1.0 if min(h3) == opp_disc else 0.0
        p3 = 1.0 / 3.0
        lk = np.array([p1, p2, p3]) * blend + (1.0 - blend) / 3.0
        self.opp.disc_logw = self.opp.disc_logw + np.log(lk)

    def equity_samples(self, my_cards_list, n):
        cards, w, s = self.current_range()
        n = int(n)
        idx = self.rng.choice(len(cards), size=n, p=w)
        opp = cards[idx]
        if opp.shape[1] == 3:
            k = np.argmax(self.strengths3()[idx], axis=1)
            sel = np.array(KEEPS)[k]
            opp = np.take_along_axis(opp, sel, axis=1)
        need = 5 - len(self.board)
        board = np.tile(np.array(self.board, dtype=np.int64), (n, 1))
        if need > 0:
            keys = self.rng.random((n, 52))
            dead = list(self.dead())
            keys[:, dead] = 9.0
            keys[np.arange(n)[:, None], cards[idx]] = 9.0
            run = np.argsort(keys, axis=1)[:, :need]
            board = np.concatenate([board, run], axis=1)
        o_sc = veval(np.concatenate([opp, board], axis=1))
        outs = []
        for mine in my_cards_list:
            m = np.tile(np.array(mine, dtype=np.int64), (n, 1))
            m_sc = veval(np.concatenate([m, board], axis=1))
            outs.append((m_sc > o_sc) + 0.5 * (m_sc == o_sc))
        return np.array(outs), s[idx]

    def river_exact(self):
        pairs, w, s = self.current_range()
        mine = np.tile(np.array(self.hole, dtype=np.int64), (len(pairs), 1))
        b = np.tile(np.array(self.board, dtype=np.int64), (len(pairs), 1))
        m_sc = veval(np.concatenate([mine, b], axis=1))
        o_sc = veval(np.concatenate([pairs, b], axis=1))
        out = (m_sc > o_sc) + 0.5 * (m_sc == o_sc)
        return out, w, s

    @staticmethod
    def price_fold_probs(strength, w, price, f_obs, typical_price, tau=0.06):
        f_obs = min(0.95, max(0.03, f_obs))
        lo, hi = -1.0, 1.0
        wn = w / w.sum()
        for _ in range(22):
            mid = (lo + hi) / 2
            avg = float((sig((typical_price + mid - strength) / tau) * wn).sum())
            if avg < f_obs:
                lo = mid
            else:
                hi = mid
        delta = (lo + hi) / 2
        return sig((price + delta - strength) / tau)

    def raise_probs(self, s, w):
        f = min(0.8, max(0.01, self.opp.raise_vs_bet.p))
        rho = self.opp.aggr_info
        th = wquantile(s, w, 1.0 - f)
        w_rp = self.params.get("likelihood", {}).get("raise_prob_width", 0.05)
        raw = sig((s - th) / w_rp)
        mean = float((raw * w).sum() / max(w.sum(), 1e-12))
        inf = np.minimum(1.0, raw * (f / max(mean, 1e-9)))
        return (1.0 - rho) * f + rho * inf

    def fold_prob(self, s, w, size_frac):
        price = size_frac / (1.0 + 2.0 * size_frac)
        fm = self.params.get("fold_model", {})
        tau = fm.get("tau_postflop", 0.06)
        typ_price = fm.get("typical_price_postflop", 0.7 / 2.4)
        fp = self.price_fold_probs(s, w, price, self.opp.fold_vs_bet.p, typ_price, tau=tau)
        if size_frac > 1.2:
            # Overbets heavily increase fold probability: opponents fold air & second-best hands
            boost = min(0.35, 0.15 * (size_frac - 1.0))
            fp = np.clip(fp + boost, 0.0, 0.98)
        return fp

    def choose_action(self, req):
        t0 = time.perf_counter()
        try:
            return self._choose_action(req)
        finally:
            self.spent_s += time.perf_counter() - t0

    def _choose_action(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        self.board = [C2I[c] for c in req.get("board", [])]
        if len(self.board) > len(self.board_full):
            self.board_full = list(self.board)
        self.process_history(req.get("history", []))
        if street == "preflop":
            return self.preflop_action(req)
        if self.range3 is None and self.range2 is None:
            self.init_range3()
        return self.postflop_action(req)

    def preflop_action(self, req):
        legal = req["legal_actions"]
        pot, tc = req["pot"], req["to_call"]
        cls = int(PF_CLASS[colex3(np.array([sorted(self.hole)]))[0]])
        pct = float(PF_PCT[colex3(np.array([sorted(self.hole)]))[0]])
        rho = np.ones(10)
        for act in self.opp_pf_acts:
            rho *= self.pf_likelihood(act, DEC_MID)
        rho /= rho.sum()
        e = float(PF_EQDEC[cls] @ rho)
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]
        o = self.opp
        real_p = self.params.get("realisation", {})
        real = real_p.get("pf_real_oop", 0.92) if self.seat == 1 else real_p.get("pf_real_ip", 0.97)
        options = {}
        if "fold" in legal:
            options["fold"] = 0.0
        if "check" in legal:
            options["check"] = e * real * pot
        if "call" in legal:
            # Preflop calling defense: if facing a massive 3-bet/shove, fold marginal trash
            if tc > 10 and pct < 0.45:
                options["call"] = -1.0
            else:
                options["call"] = e * real * (pot + tc) - tc
        if "raise" in legal:
            lo, hi = req["min_raise"], req["max_raise"]
            bs = self.params.get("bet_sizing", {})
            fm = self.params.get("fold_model", {})
            open_sizes = bs.get("preflop_open_sizes", [5, 7])
            mults = bs.get("preflop_3bet_mults", [3, 4])
            shove_pct = bs.get("preflop_shove_pct", 0.90)
            shove_opp = bs.get("preflop_shove_opp_bet", 12)
            tau_pf = fm.get("tau_preflop", 0.08)
            typ_price_pf = fm.get("typical_price_preflop", 0.30)
            
            # Gating: only raise if opening, or if re-raising with solid hands (pct >= 0.62)
            if opp_bet <= 2:
                sizes = list(open_sizes)
            elif pct >= 0.62:
                sizes = [m * opp_bet for m in mults]
            else:
                sizes = []

            # Preflop shove gating: only shove top premium holdings (pct >= shove_pct)
            if pct >= shove_pct or (pct >= 0.82 and opp_bet >= shove_opp):
                sizes.append(hi)
                
            for X in sorted({max(lo, min(hi, int(x))) for x in sizes}):
                add = X - my_bet
                c = X - opp_bet
                price = c / (pot + add + c)
                fp_d = self.price_fold_probs(DEC_MID, rho, price, o.fold_vs_raise_pf.p, typ_price_pf, tau=tau_pf)
                fp = float((fp_d * rho).sum())
                cont = rho * (1 - fp_d)
                if cont.sum() <= 1e-9:
                    cont = rho
                ec = float(PF_EQDEC[cls] @ (cont / cont.sum()))
                r_all = 1.0 if X >= hi else real
                ev = fp * pot + (1 - fp) * (ec * r_all * (pot + add + (X - opp_bet)) - add)
                options[("raise", X)] = ev
        best = max(options, key=options.get)
        return self.to_msg(best, legal)

    def _postflop_options(self, req, outv, sv, wv):
        """EV of every legal option with enhanced calling risk discipline & value bet sizing."""
        legal = req["legal_actions"]
        pot, tc = req["pot"], req["to_call"]
        my_bet, opp_bet = req["my_bet"], req["opp_bet"]
        street = req["street"]
        real_p = self.params.get("realisation", {})
        bs = self.params.get("bet_sizing", {})
        e = float((outv * wv).sum() / wv.sum())
        fl_real = real_p.get("flop_real", 0.88)
        tn_real = real_p.get("turn_real", 0.93)
        rv_real = real_p.get("river_real", 1.0)
        real = rv_real if street == "river" else (tn_real if street == "turn" else fl_real)
        
        # Analyze our current made hand category with the board
        mine_cards = self.hole + self.board
        if len(mine_cards) >= 5:
            my_eval = veval(np.array([mine_cards], dtype=np.int64))[0]
            my_cat = (my_eval >> 20) & 0xF
        else:
            my_cat = 0

        # Board danger analysis
        b_suits = [c & 3 for c in self.board]
        max_suit = max(b_suits.count(s) for s in range(4)) if self.board else 0
        is_flush_board = (max_suit >= 3)
        b_ranks = sorted(set(c >> 2 for c in self.board)) if self.board else []
        is_straight_board = any(b_ranks[i+2] - b_ranks[i] <= 4 for i in range(len(b_ranks)-2)) if len(b_ranks) >= 3 else False

        options = {}
        S = pot + tc
        if "fold" in legal:
            options["fold"] = 0.0
        if "check" in legal:
            options["check"] = e * real * pot
            
        if "call" in legal:
            # 1. Multi-street calling discipline against aggression & all-in shoves
            pot_odds = tc / max(1.0, float(pot + tc))
            can_call = True
            
            if street == "river":
                if tc >= 60:  # Big bet or river shove
                    margin = 0.10
                    if my_cat < 3:  # Less than trips (one-pair or two-pair)
                        margin = 0.16
                    if (is_flush_board and my_cat < 5) or (is_straight_board and my_cat < 4):
                        margin = 0.20
                    if e < (pot_odds + margin):
                        can_call = False
                elif tc >= 20:  # Medium river bet
                    margin = 0.05
                    if my_cat <= 1:
                        margin = 0.10
                    elif my_cat == 2 and (is_flush_board or is_straight_board):
                        margin = 0.08
                    if e < (pot_odds + margin):
                        can_call = False
            elif street == "turn":
                if tc >= 50:  # Big turn bet / shove
                    margin = 0.08
                    if my_cat < 3:
                        margin = 0.12
                    if e < (pot_odds + margin):
                        can_call = False
                elif tc >= 15:  # Standard turn bet
                    if my_cat <= 1 and e < (pot_odds + 0.06):
                        can_call = False
            elif street == "flop":
                if tc >= 35:  # Facing a big raise on flop
                    if my_cat <= 1 and e < (pot_odds + 0.10):
                        can_call = False
                elif tc >= 15:
                    if my_cat == 0 and e < (pot_odds + 0.08):
                        can_call = False

            if can_call:
                options["call"] = e * real * (pot + tc) - tc
            else:
                options["call"] = -1.0  # Force disciplined fold
                
        if "raise" in legal:
            lo, hi = req["min_raise"], req["max_raise"]
            cands = set()
            
            # Gating on facing a bet: NEVER raise/re-raise facing a bet without strong equity
            if tc > 0 and e < 0.65:
                # Facing a bet with marginal holding: do not raise (call or fold only)
                cands = set()
            elif street == "flop":
                if tc == 0:
                    for f in (0.45, 0.70, 0.95):
                        cands.add(max(lo, min(hi, opp_bet + int(round(f * (pot + tc))))))
                    if (hi - opp_bet) <= 1.25 * (pot + tc):
                        cands.add(hi)
                else:
                    # Facing a bet on flop: only raise with high equity
                    if e >= 0.68:
                        cands.add(max(lo, min(hi, opp_bet + int(round(0.85 * (pot + tc))))))
                        if e >= 0.82 and (hi - opp_bet) <= 1.5 * (pot + tc):
                            cands.add(hi)
            elif street == "turn":
                if tc == 0:
                    for f in (0.50, 0.75, 1.0):
                        cands.add(max(lo, min(hi, opp_bet + int(round(f * (pot + tc))))))
                    if (hi - opp_bet) <= 1.35 * (pot + tc) or (e >= 0.96 and (hi - opp_bet) <= 2.2 * (pot + tc)):
                        cands.add(hi)
                else:
                    if e >= 0.68:
                        cands.add(max(lo, min(hi, opp_bet + int(round(0.85 * (pot + tc))))))
                        if e >= 0.85 and (hi - opp_bet) <= 1.5 * (pot + tc):
                            cands.add(hi)
            else:  # river
                if tc == 0:
                    for f in (0.50, 0.80, 1.25):
                        cands.add(max(lo, min(hi, opp_bet + int(round(f * (pot + tc))))))
                    if (hi - opp_bet) <= 1.8 * (pot + tc) or e >= 0.96:
                        cands.add(hi)
                else:
                    if e >= 0.72:
                        cands.add(max(lo, min(hi, opp_bet + int(round(1.0 * (pot + tc))))))
                        if e >= 0.90:
                            cands.add(hi)

            pr_base = self.raise_probs(sv, wv)
            for X in sorted(cands):
                add = X - my_bet
                opp_add = X - opp_bet
                S = max(S, pot + add + opp_add)
                size_frac = opp_add / max(1.0, pot + tc)
                fp_i = self.fold_prob(sv, wv, size_frac)
                pr_i = np.zeros_like(fp_i) if X >= hi else np.minimum(pr_base, 1.0 - fp_i)
                pc_i = np.clip(1.0 - fp_i - pr_i, 0.0, 1.0)
                tot = wv.sum()
                fp = float((fp_i * wv).sum() / tot)
                pr = float((pr_i * wv).sum() / tot)
                pc = float((pc_i * wv).sum() / tot)
                cw = wv * pc_i
                ec = float((outv * cw).sum() / cw.sum()) if cw.sum() > 1e-9 else e
                r = 1.0 if (X >= hi or street == "river") else real
                ev = fp * pot + pc * (ec * r * (pot + add + opp_add) - add)
                if pr > 1e-6:
                    R = min(hi, X + (pot + add + opp_add))
                    rw = wv * pr_i
                    er = float((outv * rw).sum() / rw.sum())
                    call_more = R - X
                    ev_call = er * (pot + add + (R - opp_bet) + call_more) - add - call_more
                    ev += pr * max(-add, ev_call)
                    
                # 3. Turn barreling discipline: don't bloat pot with marginal air
                if street == "turn" and tc == 0 and e < 0.50 and my_cat <= 1:
                    if self.opp.fold_vs_bet.p < 0.50:
                        ev -= 0.6 * pot
                        
                options[("raise", X)] = ev
        return options, S, e

    def postflop_action(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        pot, tc = req["pot"], req["to_call"]
        t_start = time.perf_counter()
        scale = self.budget_scale(req)
        samp_p = self.params.get("sampling", {})
        ad = self.params.get("adaptive", {})
        z = ad.get("z", 1.5)
        max_mult = ad.get("max_mult", 4.0)
        min_gap_pot = ad.get("min_gap_pot", 0.01)
        if street == "river":
            out, w, s = self.river_exact()
            options, S, e = self._postflop_options(req, out, s, w)
            n_used = len(out)
        else:
            n_base = samp_p.get("flop_samples", 500) if street == "flop" else samp_p.get("turn_samples", 600)
            n = max(50, int(n_base * scale))
            if len(self.hole) == 3:
                keeps = [[self.hole[i], self.hole[j]] for i, j in KEEPS]
            else:
                keeps = [self.hole]
            outs, sv = self.equity_samples(keeps, n)
            w_key = "flop_weight" if street == "flop" else "turn_weight"
            allowed = self.extra_time_s(req, ad.get(w_key, 0.30 if street == "flop" else 0.22))
            n_max = int(n_base * max_mult)
            while True:
                outv = outs[int(np.argmax(outs.mean(axis=1)))]
                wv = np.ones(len(sv))
                options, S, e = self._postflop_options(req, outv, sv, wv)
                cur_n = outs.shape[1]
                if scale < 1.0 or cur_n >= n_max or len(options) < 2:
                    break
                vals = sorted(options.values(), reverse=True)
                gap = vals[0] - vals[1]
                se = float(outv.std()) / (cur_n ** 0.5)
                if gap < min_gap_pot * max(pot, 1) or gap >= z * se * S:
                    break
                more = min(cur_n, n_max - cur_n)
                elapsed = time.perf_counter() - t_start
                if elapsed * (1.0 + more / cur_n) > allowed:
                    break
                o2, s2 = self.equity_samples(keeps, more)
                outs = np.concatenate([outs, o2], axis=1)
                sv = np.concatenate([sv, s2])
            n_used = outs.shape[1]
        best = max(options, key=options.get)
        log(f"h{self.hand_id} {street} e={e:.2f} n={n_used} pot={pot} tc={tc} -> {best} " +
            " ".join(f"{k if isinstance(k, str) else k[1]}:{v:.1f}" for k, v in options.items()))
        return self.to_msg(best, legal)

    def to_msg(self, best, legal):
        if isinstance(best, tuple):
            return {"type": "action", "action": "raise", "amount": int(best[1])}
        if best in legal:
            return {"type": "action", "action": best}
        return {"type": "action", "action": "check" if "check" in legal else "fold"}

    def choose_discard(self, req):
        t0 = time.perf_counter()
        try:
            return self._choose_discard(req)
        finally:
            self.spent_s += time.perf_counter() - t0

    def _choose_discard(self, req):
        self.board = [C2I[c] for c in req.get("board", [])]
        if len(self.board) > len(self.board_full):
            self.board_full = list(self.board)
        hole = [C2I[c] for c in req["hole_cards"]]
        self.hole = hole
        if self.range3 is None:
            self.init_range3()
        self.process_history(getattr(self, "last_history", []))
        t_start = time.perf_counter()
        keeps = [[hole[i], hole[j]] for i, j in KEEPS]
        samp_p = self.params.get("sampling", {})
        disc_p = self.params.get("discard", {})
        ad = self.params.get("adaptive", {})
        z = ad.get("z", 1.5)
        n_base = samp_p.get("discard_samples", 900)
        scale = self.budget_scale(req)
        n = max(50, int(n_base * scale))
        outs, _ = self.equity_samples(keeps, n)
        allowed = self.extra_time_s(req, ad.get("discard_weight", 0.25))
        n_max = int(n_base * ad.get("max_mult", 4.0))
        tiny = ad.get("disc_tiny_gap", 0.003)
        while True:
            ev = outs.mean(axis=1)
            order = np.argsort(-ev)
            cur_n = outs.shape[1]
            if scale < 1.0 or cur_n >= n_max:
                break
            gap = float(ev[order[0]] - ev[order[1]])
            if gap < tiny:
                break
            d = outs[order[0]] - outs[order[1]]
            se = float(d.std()) / (cur_n ** 0.5)
            if gap >= z * se:
                break
            more = min(cur_n, n_max - cur_n)
            elapsed = time.perf_counter() - t_start
            if elapsed * (1.0 + more / cur_n) > allowed:
                break
            o2, _ = self.equity_samples(keeps, more)
            outs = np.concatenate([outs, o2], axis=1)
        k = int(order[0])
        near_tie = disc_p.get("near_tie_margin", 0.015)
        mix_p = disc_p.get("mix_prob", 0.35)
        if ev[order[0]] - ev[order[1]] < near_tie and random.random() < mix_p:
            k = int(order[1])
        card = hole[DROP[k]]
        return {"type": "discard", "card": I2C[card]}

    def on_discard_result(self, m):
        if self.hole:
            self.hole = [c for c in self.hole if c != C2I[m["my_discard"]]]
        self.my_discard = C2I[m["my_discard"]]
        self.apply_opp_discard(C2I[m["opp_discard"]])

    def showdown_board(self, m):
        for k in ("board", "community_cards", "final_board", "board_cards"):
            v = m.get(k)
            if isinstance(v, list) and len(v) >= 3:
                return [C2I[c] for c in v]
        return list(self.board_full)

    def on_hand_result(self, m):
        self.hands_played += 1
        o = self.opp
        o.hands += 1
        hist = m.get("history", [])
        try:
            self.process_history(hist, stats_only=True)
            for st, e0, e1 in self.pending_aggr:
                o.fade_add(o.sd_raise, o.sd_raise_prior, max(0.0, e1 - e0))
            self.pending_aggr = []
        except Exception as ex:
            log(f"stats replay error {ex}")
        if m.get("result") == "showdown" and m.get("opp_hole_cards") and m.get("opp_discard"):
            try:
                two = [C2I[c] for c in m["opp_hole_cards"]]
                disc = C2I[m["opp_discard"]]
                board = self.showdown_board(m)
                if len(board) < 5:
                    raise ValueError("full board unavailable at showdown")
                self.learn_discard_style(two, disc, board[:3])
                s_by = {}
                three = np.array([sorted(two + [disc])], dtype=np.int64)
                bi3 = self.board_info(board[:3])
                s_by["flop"] = max(float(bi3.strength(three[:, [i, j]])[0]) for i, j in KEEPS)
                s_by["turn"] = float(self.board_info(board[:4]).strength(np.array([two]))[0])
                s_by["river"] = float(self.board_info(board).strength(np.array([two]))[0])
                for st in ("flop", "turn", "river"):
                    szs = self.opp_sizes.get(st)
                    if szs:
                        if max(szs) >= o.size_split:
                            o.fade_add(o.sz_large, o.sz_large_prior, s_by[st])
                        else:
                            o.fade_add(o.sz_small, o.sz_small_prior, s_by[st])
                bet_seen = False
                for st in ("flop", "turn", "river"):
                    acts = [h.get("action") for h in hist if h.get("street") == st and h.get("actor") == "opp"]
                    if not acts:
                        continue
                    if "raise" in acts:
                        o.fade_add(o.sd_raise, o.sd_raise_prior, s_by[st])
                        bet_seen = True
                    else:
                        o.fade_add(o.sd_pass, o.sd_pass_prior, s_by[st])
                if bet_seen:
                    o.bluff_share.add(s_by["river"] < 0.4)
            except Exception as ex:
                log(f"learn error {ex}")

def rough_equity(hole, board):
    """Cheap hand-strength guess (no simulation) for exception fallbacks."""
    hr = [c >> 2 for c in hole]
    if not hr:
        return 0.4
    if len(board) < 3:
        top = sorted(hr, reverse=True)[:2]
        eq = 0.28 + 0.02 * sum(top)
        if len(set(hr)) < len(hr):
            eq += 0.10
        return min(0.9, eq)
    br = [c >> 2 for c in board]
    rc = {}
    for r in hr + br:
        rc[r] = rc.get(r, 0) + 1
    best = max(rc.values())
    pairs_made = sum(1 for v in rc.values() if v >= 2)
    hit = [r for r in set(hr) if r in br]
    if best >= 4:
        return 0.97
    if best >= 3 and (hit or len(set(hr)) < len(hr)):
        return 0.85
    if pairs_made >= 2 and hit:
        return 0.78
    if hit or len(set(hr)) < len(hr):
        r = max(hit) if hit else max(hr)
        return 0.50 + 0.15 * r / 12.0
    return 0.20 + 0.10 * max(hr) / 12.0

def _plain_safe_action(req):
    legal = req.get("legal_actions", [])
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "call" in legal and req.get("to_call", 0) <= max(4, 0.25 * req.get("pot", 0)):
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else "call"}

def safe_action(req, bot=None):
    """Exception fallback: check if free, else call only when pot odds justify it."""
    try:
        legal = req.get("legal_actions", [])
        if "check" in legal:
            return {"type": "action", "action": "check"}
        tc = req.get("to_call", 0)
        pot = req.get("pot", 0)
        hole = list(getattr(bot, "hole", []) or [])
        board = [C2I[c] for c in req.get("board", [])]
        eq = rough_equity(hole, board)
        po = tc / float(pot + tc) if (pot + tc) > 0 else 0.0
        if "call" in legal and (tc <= max(2, 0.10 * pot) or eq > po + 0.08):
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold" if "fold" in legal else "call"}
    except Exception:
        return _plain_safe_action(req)

def safe_discard(req):
    cards = req.get("hole_cards", [])
    if not cards:
        return {"type": "discard", "card": "2s"}
    return {"type": "discard", "card": min(cards, key=lambda c: RANKS.index(c[0]))}

def main():
    bot = Bot()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        t = msg.get("type")
        reply = None
        try:
            if t == "action_request":
                bot.last_history = msg.get("history", [])
                reply = bot.choose_action(msg)
            elif t == "discard_request":
                reply = bot.choose_discard(msg)
            elif t == "new_hand":
                bot.new_hand(msg)
            elif t == "discard_result":
                bot.on_discard_result(msg)
            elif t == "hand_result":
                bot.on_hand_result(msg)
            elif t == "init":
                bot.on_init(msg)
            elif t == "match_end":
                break
        except Exception as ex:
            log(f"error on {t}: {type(ex).__name__}: {ex}")
            if t == "action_request":
                reply = safe_action(msg, bot)
            elif t == "discard_request":
                reply = safe_discard(msg)
        if reply is not None:
            send_proto(reply)

if __name__ == "__main__":
    main()
