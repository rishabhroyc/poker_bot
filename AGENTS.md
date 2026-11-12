# AGENTS.md - Discard Hold'em pokerbot

## Mission
Build the strongest, safest bot for Discard Hold'em: heads-up no-limit hold'em, 3 hole cards,
simultaneous exposed discard after the flop, 200-chip stacks reset every hand, blinds 1/2,
max 4 bets/raises per street. We win by winning MATCHES (1 point each). Chips are only a capped
tie-break (about 1 chip per hand). Robustness beats cleverness.

## Hard constraints (never violate)
- Bot runtime: Python 3.11; standard library + numpy + scipy ONLY. No network, no subprocesses,
  no file writes, read-only filesystem. 1 vCPU, 1 GB RAM.
- Ship ONE self-contained bot.py (<= 5 MB). No imports of sibling files and no reading of data
  files at runtime: embed any tables/params inside bot.py.
- stdout carries protocol JSON only: exactly one flushed line per reply. Debug goes to stderr
  (<= 64 KB per leg).
- Time: per-leg bank 30 s (Stage 1, 200 hands) or 75 s (Stage 2, 500 hands); 5 s per reply;
  wall-clock kill 90 s / 225 s per leg. A crash or timeout forfeits the whole match.
- No state across legs. Never try to identify duplicate hands, signal other bots, or exploit the
  evaluation system (Rulebook 10.2-10.3). Never copy other teams' or public bot code (10.1);
  published algorithms are fine, write our own implementation.
- starter_kit/*.pdf are the source of truth for rules. engine.py is only the local test runner.

## Working agreements
- Plan first (short), then implement in small verified steps. Show real command output as
  evidence. Never claim a result you did not run.
- bot_v3/ is frozen: never edit. dev/ is the working copy. submissions/ holds tagged builds.
- A strategy change is kept only if tools/gauntlet.py shows it beats the current champion on the
  same seeds (paired duplicate analysis, confidence interval above zero) and does not regress
  any zoo opponent materially. No gate result = not merged.
- One strategy dimension per experiment. Record each experiment in results/ (JSON + 5-line summary).
- Never make the bot slower without a measured reason. Budget: <= 40% of the time bank at nominal
  speed, every reply < 1.5 s.
- Every exception path must still send a legal reply (safe_action / safe_discard).
- Tag every candidate (git tag cand-N). Keep a known-good build at submissions/safe.py.

## Commands
- One match:  python3.11 starter_kit/engine.py dev/bot.py opps/<name>.py --stage 1 --seed 7 --log-dir logs/x
- Gauntlet:   python3.11 tools/gauntlet.py --bot dev/bot.py --stage 1 --seeds 20 --jobs 4
- Build:      python3.11 tools/build_submission.py        (writes submissions/bot.py)
- Release:    python3.11 tools/release_check.py submissions/bot.py

## Style
Small pure functions, type hints where cheap, no global mutable state outside the Bot class,
deterministic given a seed (except where randomisation is intentional and seeded).