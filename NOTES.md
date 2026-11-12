# Discard Hold'em Bot Constraints & Rules (`NOTES.md`)

## 1. Runtime & Submission Limits
- **Language & Packages**: Python 3.11; standard library, `numpy`, and `scipy` only (no runtime installs).
- **Packaging**: Single `bot.py`, or `.zip` with `bot.py` at top level; max size **5 MB**.
- **Resources (provisional)**: 1 vCPU, 1 GB RAM, no network access, read-only filesystem (may read files in own submission; no file writes, sockets, subprocesses, or sandbox escape).
- **Process & State**: Fresh bot process per leg. State allowed within a leg; **zero state** between legs or matches. Randomness allowed.

## 2. Time Limits (provisional)
- **Start-up**: 5 s grace at leg launch (not charged to bank; first reply up to 10 s after launch).
- **Time Bank**: Cumulative per leg (not reset per hand): **30 s** (Stage 1, 200 hands/leg), **75 s** (Stage 2, 500 hands/leg). Discard clocks for both bots run simultaneously.
- **Per-Reply Limit**: **5 s** max for any single `action_request` or `discard_request` reply.
- **Wall-Clock Kill**: **90 s/leg** (Stage 1), **225 s/leg** (Stage 2).

## 3. Protocol & Legality
- **I/O**: UTF-8 JSON, one object per `\n`-terminated line on stdin/stdout; **must flush stdout** after every reply.
- **Stdout / Stderr**: Only protocol JSON on stdout (stray stdout while a request is pending is treated as a malformed reply, and the real reply is ignored). Debug goes to stderr (64 KB/leg limit, unparsed).
- **Lifecycle**: Reply only to `action_request` and `discard_request`; ignore unknown message types/fields; exit after `match_end`.
- **Betting**: Stacks reset to 200 every hand; blinds 1/2 (button=SB acts first preflop; BB acts first postflop). `raise` uses street-total ("to") integer `amount` in `[min_raise, max_raise]`. `min_raise = opp_bet + max(last_raise_size, 2)`. Max **4 bets/raises per street** (BB counts as 1st preflop).
- **Discard**: Simultaneous after flop betting (even if all-in); revealed to both; removed from play. Must be one of current 3 hole cards.
- **Corrections**: Illegal/unparsable action -> `check` if legal else `fold` (+ `notice`). Illegal/missing discard -> lowest rank (`2`<`A`), tie-break suit `s < h < d < c` (+ `notice`). No reply = timeout.

## 4. Scoring & Tournament Structure
- **Match Format**: Two legs of duplicate deals (hole cards & seats swapped); Stage 1 = 200 deals (400 hands), Stage 2 = 500 deals (1000 hands).
- **Match Points**: Win (greater net chips over both legs) = **1**, Draw = **0.5**, Loss = **0**.
- **Tie-Break (`Score_cap`)**: `sum(clip(Delta_m, -C, C)) / sum(H_m)`, where `C = 400` (Stage 1) or `C = 1000` (Stage 2) [provisional]. Further ties for prizes/cut-off settled by 500-deal Stage 2 replays.
- **Forfeit**: Crash, EOF on stdout, time bank exhaustion, >5 s reply, or wall-clock kill in either leg forfeits the entire match (`Delta = -C` for forfeiter, `+C` for opponent; `H_m` = scheduled hands).

## 5. Conduct
- **Originality**: Bot must be team's own work; no copying other teams' or public bot code (standard libraries and published algorithms allowed).
- **No Collusion**: No signalling, sharing code/strategy, or deliberate losing.
- **No Exploits**: No network access, server/bot interference, duplicate-deal detection, or cross-leg state passing. Report any engine weakness found. Code of prize winners (and potentially any team) is reviewed.

