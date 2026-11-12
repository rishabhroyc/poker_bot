# Discard Audit & Experiments (`results/discard.md`)

## A. Ground-Truth Agreement & Equity Loss (`tools/discard_lab.py`)
Evaluated `50,000` random `(3 hole cards, 3 flop cards)` situations with `400` rollouts each for uniform and range-aware (plausible flop bet/call `pstrength` weighting) ground truth, and checked `5,000` situations at `v3`'s `900`-sample budget:
- **Uniform ground truth**: agreement `88.30%`, mean equity lost when disagreeing `0.0108` (`1.08%`).
- **Range-aware ground truth**: agreement `84.84%`, mean equity lost when disagreeing `0.0242` (`2.42%`).

## B. Profiling, Suit-Isomorphism Caching, & Cheap Scoring Prior
- **Baseline `_choose_discard` profile** (200 calls): `10,445.4 ms` total (`52.23 ms/call`); `equity_samples` took `7.972 s` (`76.9%`, with `veval` `6.432 s`), `current_range`/`process_history` `1.587 s`, `init_range3` `0.756 s`.
- **Offline linear scoring prior** (12 numpy features trained on `45,000` situations, tested on `5,000`):
  - Overall prior agreement: `71.18%`.
  - Margin-gated (`top1 - top2 >= 0.05`): triggers on `20.26%` of flops with `95.26%` agreement and `0.0036` mean equity loss.
- **Speedup benchmark** (200 calls with suit-isomorphism cache + margin-gated prior): `8,523.4 ms` (`42.62 ms/call`, `-18.4%` latency).
- **20-seed Gauntlet Gate (`cand_discard` vs `champion_v3`)**:
  - Pooled paired diff: `-2.13 ± 0.20` cph, 95% CI `[-2.52, -1.74]`, `4` forfeits, max leg `12.07 s` — **GATE FAIL** (caching across different preflop/flop action histories and reusing flop `outs` stale-weighted `range3`, hurting against `01_starter_clone` `-13.52` cph and `03_calling_station` `-15.23` cph). **Reverted.**

## C. Information Leakage & Mixing (`opps/14_discard_exploiter.py`, 10 seeds Stage 1)
Tested `near_tie_margin` and `mix_prob` against a zoo bot that conditions on our revealed discard:

| Config | Mean (cph) | SE |
| --- | ---: | ---: |
| `base (m=0.028, p=0.332)` | +9.11 | 1.40 |
| `no_mix (m=0.0, p=0.0)` | +12.19 | 1.39 |
| `m=0.015, p=0.332` | +12.22 | 1.26 |
| `m=0.045, p=0.332` | +10.23 | 1.37 |
| `m=0.028, p=0.15` | +6.64 | 1.32 |
| `m=0.028, p=0.50` | +6.48 | 1.79 |

## D. Opponent Discard Inference Audit (`400` trials/policy)
Measured Total Variation (TV) distance between the true posterior over kept pairs and `Bot.apply_opp_discard`:

| Opponent Policy | Current `apply_opp_discard` TV | Softmax (`temp=0.023`) TV |
| --- | ---: | ---: |
| `best_pstrength` | 0.0600 | 0.2045 |
| `softmax_0.023` | 0.1539 | 0.0405 |
| `lowest_rank` | 0.0416 | 0.0416 |
| `random` | `1.3e-9` | `1.2e-9` |

No double counting with flop-action likelihoods was found (`hist_seen` advances monotonically and `apply_opp_discard` multiplies `range3` by `p(opp_disc | c1, c2, c3)` once).

## Decision
Per the gate rule, **reverted all `dev/bot.py` discard edits** after `cand_discard` failed the 20-seed gauntlet gate (`-2.13 ± 0.20` cph). `dev/bot.py` remains at `champion_v3`.

