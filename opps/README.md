# Opponent Zoo (`opps/`)

- `01_starter_clone.py` — Never bets or folds (passive check/call), naive lowest-rank discard; tests basic value betting and thin value extraction.
- `02_random_bot.py` — Uniformly random sizing/actions and random discard; tests baseline sanity and variance resilience.
- `03_calling_station.py` — Never folds and never raises, naive lowest-rank discard; tests whether the bot stops bluffing and bets value relentlessly.
- `04_maniac.py` — Raises/shoves ~88% of turns, naive lowest-rank discard; tests bluff-catching, trap-calling, and avoiding over-folding.
- `05_nit.py` — Plays only top ~40% hands and folds to aggression, smart equity discard; tests stealing blinds/pots and folding to its rare raises.
- `06_tag.py` — Tight-aggressive MC-equity threshold player with pot-fraction bets and smart discard; tests solid ABC baseline play.
- `07_lag.py` — Wide opens, frequent 3-bets, flop/turn barrels, smart discard; tests defending against relentless aggression and light 3-betting.
- `08_trapper.py` — Slow-plays monsters and check-raises, smart discard; tests pot control and not walking into check-raise traps.
- `09_min_bettor.py` — Limps preflop and min-bets postflop, naive lowest-rank discard; tests exploiting weak min-bet ranges and poor implied odds.
- `10_overbet_shover.py` — Jams/overbets polarized value and bluffs, smart discard; tests polarized range calling/folding discipline.
- `11_over_folder.py` — Folds to any bet above 1/3 pot, naive lowest-rank discard; tests whether the bot detects high fold equity and bluffs profitably.
- `12_adaptive_exploiter.py` — Tracks opponent fold/raise rates and shifts bluff frequency upward vs folders, smart discard; tests resilience against adaptive exploitation.
- `13_llm_generic.py` — Heuristic strength + pot-odds thresholds with 10% action noise and random discard; tests beating typical un-tuned hackathon bots.

