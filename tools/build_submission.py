#!/usr/bin/env python3.11
"""Build a self-contained submissions/bot.py from dev/.

Embeds:
  - NF_TABLE (6188 int32) and FL_TABLE (1287 int32) + vectorised veval()
  - preflop.npz arrays (combo_class int16, eq float32, eq_dec float32, dec_edges float32)
  - params.json as a Python dict literal

Removes sys.path edits, __file__ usage, and all runtime file reads.
Wraps heavy setup (numpy + tables + Bot) in a fallback guard.
Verifies embedded veval == evaluator.eval_cards on 200,000 random 5/6/7-card hands.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np


def _encode_array(arr: np.ndarray) -> str:
    raw = np.ascontiguousarray(arr).tobytes()
    comp = zlib.compress(raw, level=9)
    return base64.b64encode(comp).decode("ascii")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verify_veval_equality(veval_fn, eval_cards_fn, total_hands: int = 200_000, seed: int = 12345) -> None:
    rng = np.random.default_rng(seed)
    counts = {5: total_hands // 3, 6: total_hands // 3, 7: total_hands - 2 * (total_hands // 3)}
    checked = 0
    for k, n in counts.items():
        # Sample n hands of k distinct cards from 0..51
        hands = np.empty((n, k), dtype=np.int64)
        for i in range(n):
            hands[i] = rng.choice(52, size=k, replace=False)
        v_scores = veval_fn(hands)
        e_scores = np.array([eval_cards_fn(row.tolist()).score for row in hands], dtype=np.int32)
        if not np.array_equal(v_scores, e_scores):
            bad = int(np.flatnonzero(v_scores != e_scores)[0])
            raise AssertionError(
                f"veval mismatch on {k}-card hand #{bad} ({hands[bad].tolist()}): "
                f"veval={v_scores[bad]} vs eval_cards={e_scores[bad]}"
            )
        checked += n
    print(f"Verified embedded veval == evaluator.eval_cards on {checked:,} random 5/6/7-card hands.")


def _transform_bot_body(bot_src: str, params_literal: str, pre_b64: dict[str, str], nf_b64: str, fl_b64: str) -> str:
    lines = bot_src.splitlines()

    # Strip leading shebang/docstring and top-level imports up to `# ---------------------------------------------------------------- Clock manager`
    # We'll locate key anchors in dev/bot.py so we preserve all strategy code verbatim.
    def find_line(prefix: str) -> int:
        for idx, ln in enumerate(lines):
            if ln.startswith(prefix):
                return idx
        raise RuntimeError(f"Anchor not found in dev/bot.py: {prefix!r}")

    idx_clock = find_line("# ---------------------------------------------------------------- Clock manager")
    idx_params = find_line("# ---------------------------------------------------------------- params loader")
    idx_static = find_line("# ---------------------------------------------------------------- static tables")
    idx_dec_mid = find_line("DEC_MID = ")
    idx_main = find_line("def main():")
    idx_if_main = find_line('if __name__ == "__main__":')

    # Check strategy body (between DEC_MID+1 and def main()) has no forbidden tokens
    strategy_lines = lines[idx_dec_mid + 1 : idx_main]
    strategy_text = "\n".join(strategy_lines)
    for forbidden in ("__file__", "sys.path", "open(", "np.load", "veval import"):
        if forbidden in strategy_text:
            raise RuntimeError(f"Unexpected {forbidden!r} in strategy body")

    clock_block = "\n".join(lines[idx_clock:idx_params])
    static_head = "\n".join(lines[idx_static : find_line("PRE = ")])

    # Indent a block of code by 4 spaces for inside `try:`
    def indent4(block: str) -> str:
        return "\n".join(("    " + ln if ln.strip() else "") for ln in block.splitlines())

    embedded_veval_and_data = f'''import numpy as np
from itertools import combinations
from math import comb

_NF_B64 = "{nf_b64}"
_FL_B64 = "{fl_b64}"

def _decode_arr(b64_str: str, dtype, shape=None):
    raw = zlib.decompress(base64.b64decode(b64_str))
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(shape) if shape is not None else arr

NF_TABLE = _decode_arr(_NF_B64, np.int32)
FL_TABLE = _decode_arr(_FL_B64, np.int32)
BIN = np.array([[comb(n, k) for k in range(6)] for n in range(18)], dtype=np.int64)
SUBSETS = {{k: np.array(list(combinations(range(k), 5)), dtype=np.int64) for k in (5, 6, 7)}}

def veval(cards):
    """cards: int array (N, k), k in {{5, 6, 7}}. Returns int32 scores (N,)."""
    cards = np.sort(np.asarray(cards, dtype=np.int64), axis=1)
    sub = cards[:, SUBSETS[cards.shape[1]]]
    ranks = sub >> 2
    suits = sub & 3
    flush = (suits == suits[:, :, :1]).all(axis=2)
    nf_idx = (BIN[ranks[..., 0], 1] + BIN[ranks[..., 1] + 1, 2] + BIN[ranks[..., 2] + 2, 3]
              + BIN[ranks[..., 3] + 3, 4] + BIN[ranks[..., 4] + 4, 5])
    fl_idx = (BIN[ranks[..., 0], 1] + BIN[ranks[..., 1], 2] + BIN[ranks[..., 2], 3]
              + BIN[ranks[..., 3], 4] + BIN[ranks[..., 4], 5])
    score = np.where(flush, FL_TABLE[np.where(flush, fl_idx, 0)], NF_TABLE[nf_idx])
    return score.max(axis=1)

DEFAULT_PARAMS = {params_literal}

def load_params(path=None):
    return DEFAULT_PARAMS

{static_head}
PRE = {{
    "combo_class": _decode_arr("{pre_b64['combo_class']}", np.int16, (22100,)),
    "eq": _decode_arr("{pre_b64['eq']}", np.float32, (1755,)),
    "eq_dec": _decode_arr("{pre_b64['eq_dec']}", np.float32, (1755, 10)),
    "dec_edges": _decode_arr("{pre_b64['dec_edges']}", np.float32, (9,)),
}}
PF_CLASS = PRE["combo_class"].astype(np.int64)
PF_EQ = PRE["eq"].astype(np.float64)
PF_EQDEC = PRE["eq_dec"].astype(np.float64)
_eq_combo = PF_EQ[PF_CLASS]
_order = np.argsort(_eq_combo, kind="stable")
PF_PCT = np.empty(len(_eq_combo), dtype=np.float64)
PF_PCT[_order] = np.arange(len(_eq_combo), dtype=np.float64) / (len(_eq_combo) - 1)
DEC_MID = np.linspace(0.05, 0.95, 10)
'''

    main_block = "\n".join(lines[idx_main:])

    header = '''#!/usr/bin/env python3.11
"""Self-contained Discard Hold'em submission bot (auto-generated by tools/build_submission.py)."""

import base64
import json
import os
import random
import sys
import time
import zlib

# Thread limitation for NumPy/OpenBLAS to keep latency predictable
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

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
    text = str(msg) + "\\n"
    b_len = len(text.encode("utf-8", "replace"))
    if _LOG_BYTES_WRITTEN + b_len > _MAX_LOG_BYTES:
        sys.stderr.write("[LOG LIMIT REACHED]\\n")
        _LOG_BYTES_WRITTEN = _MAX_LOG_BYTES
        return
    sys.stderr.write(text)
    sys.stderr.flush()
    _LOG_BYTES_WRITTEN += b_len

def send_proto(obj: dict) -> None:
    data = json.dumps(obj, separators=(",", ":"))
    _PROTO_OUT.write(data + "\\n")
    _PROTO_OUT.flush()

def _fallback_action(req):
    legal = req.get("legal_actions", [])
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "call" in legal:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else "check"}

def _fallback_discard(req):
    cards = req.get("hole_cards", [])
    if not cards:
        return {"type": "discard", "card": "2s"}
    return {"type": "discard", "card": min(cards, key=lambda c: (RANKS.index(c[0]), SUITS.index(c[1])))}

_SETUP_OK = True
_SETUP_ERR = None

try:
'''

    fallback_except = '''except Exception as _ex:
    _SETUP_OK = False
    _SETUP_ERR = _ex
    log(f"setup failed, using fallback bot: {type(_ex).__name__}: {_ex}")

    class Bot:
        def on_init(self, msg): pass
        def new_hand(self, msg): pass
        def on_discard_result(self, msg): pass
        def on_hand_result(self, msg): pass
        def choose_action(self, req): return _fallback_action(req)
        def choose_discard(self, req): return _fallback_discard(req)

    def safe_action(req, bot=None):
        return _fallback_action(req)

    def safe_discard(req):
        return _fallback_discard(req)
'''

    parts = [
        header,
        indent4(clock_block),
        "",
        indent4(embedded_veval_and_data),
        "",
        indent4(strategy_text),
        fallback_except,
        "",
        main_block,
        "",
    ]
    return "\n".join(parts)


def build(dev_dir: Path, out_path: Path, verify_hands: int = 200_000) -> None:
    # Load dev/evaluator.py and dev/veval.py cleanly
    sys.path.insert(0, str(dev_dir))
    try:
        ev_mod = _load_module("dev_evaluator", dev_dir / "evaluator.py")
        ve_mod = _load_module("dev_veval", dev_dir / "veval.py")
    finally:
        if str(dev_dir) in sys.path:
            sys.path.remove(str(dev_dir))

    nf_table = np.asarray(ve_mod.NF_TABLE, dtype=np.int32)
    fl_table = np.asarray(ve_mod.FL_TABLE, dtype=np.int32)
    assert nf_table.shape == (6188,), f"Unexpected NF_TABLE shape {nf_table.shape}"
    assert fl_table.shape == (1287,), f"Unexpected FL_TABLE shape {fl_table.shape}"

    nf_b64 = _encode_array(nf_table)
    fl_b64 = _encode_array(fl_table)

    # Load preflop.npz
    pre = np.load(dev_dir / "preflop.npz")
    pre_b64 = {
        "combo_class": _encode_array(pre["combo_class"].astype(np.int16)),
        "eq": _encode_array(pre["eq"].astype(np.float32)),
        "eq_dec": _encode_array(pre["eq_dec"].astype(np.float32)),
        "dec_edges": _encode_array(pre["dec_edges"].astype(np.float32)),
    }

    # Load params.json
    with open(dev_dir / "params.json", "r", encoding="utf-8") as f:
        params = json.load(f)
    params_literal = repr(params)

    # Transform dev/bot.py
    bot_src = (dev_dir / "bot.py").read_text(encoding="utf-8")
    out_src = _transform_bot_body(bot_src, params_literal, pre_b64, nf_b64, fl_b64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_src, encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024.0
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes / {size_kb:.1f} KB)")

    # Verify the generated submission's own veval against evaluator.eval_cards
    if verify_hands > 0:
        sub_mod = _load_module("sub_bot_verify", out_path)
        assert sub_mod._SETUP_OK, f"Submission setup failed: {sub_mod._SETUP_ERR}"
        _verify_veval_equality(sub_mod.veval, ev_mod.eval_cards, total_hands=verify_hands)


def main():
    parser = argparse.ArgumentParser(description="Build self-contained submissions/bot.py")
    parser.add_argument("--dev-dir", type=Path, default=Path("dev"))
    parser.add_argument("--out", type=Path, default=Path("submissions/bot.py"))
    parser.add_argument("--verify-hands", type=int, default=200_000)
    args = parser.parse_args()
    build(args.dev_dir, args.out, args.verify_hands)


if __name__ == "__main__":
    main()

