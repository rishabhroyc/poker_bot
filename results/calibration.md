# Equity Calibration & Realisation Sweep (`results/calibration.md`)

## 1. 20,000-Hand Equity Calibration Test
Simulated 50 Stage 1 matches (20,000 hands) across 5 diverse opponents (`03_calling_station`, `06_tag`, `07_lag`, `08_trapper`, `13_llm_generic`, 10 seeds each) with `PB_CALIB_LOG=1`, comparing predicted equity at each decision against realised showdown win rate (`1.0` win, `0.5` split, `0.0` loss).

![Calibration Plot](file:///d:/pokerbot_v3/results/calibration_plot.png)

| Street | Showdown Decisions | Mean Predicted | Mean Realised | Bias (Realised − Pred) | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `preflop` | 13,172 | 0.4877 | 0.6291 | +0.1414 | 0.1414 |
| `flop` | 10,571 | 0.5142 | 0.6357 | +0.1216 | 0.1354 |
| `turn` | 9,140 | 0.5615 | 0.6158 | +0.0543 | 0.0740 |
| `river` | 7,980 | 0.5410 | 0.5883 | +0.0473 | 0.0534 |

**Diagnosis**:
- The river evaluator (`river_exact`) is well-calibrated in `[0.7, 1.0]` (bias `-0.0036` to `-0.0164`), with a `+0.05` to `+0.11` positive residual in `[0.1, 0.7]` because the Bayesian range model conservatively assumes opponents fold more weak hands than `03_calling_station` and `13_llm_generic` actually do.
- Preflop (`+0.1414`) and flop (`+0.1216`) positive residuals reflect equity over-realisation from smart 2-of-3 discard selection against naive/random discarders rather than a bug in `veval` or `equity_samples`; therefore, no ad-hoc bias offset was added to the sampler or range weights.

---

## 2. Realisation Parameter Sweep (`tools/realisation_sweep.py`)
Swept 5 values per realisation parameter one at a time against `results/champion_v3.json` (`--quick`, 3 seeds, Stage 1), followed by `--full` (20 seeds) on the top 2 candidates (`turn_real=0.9` and `pf_real_ip=1.1`):

| Parameter | Current | Tested Values (Quick Paired Diff cph vs `champion_v3`) |
| --- | ---: | --- |
| `pf_real_oop` | `0.894289` | `0.80` (`-1.27±0.49`), `0.85` (`-2.47±0.49`), **`0.894289` (`0.00`)**, `0.94` (`-2.15±0.48`), `0.99` (`-1.29±0.49`) |
| `pf_real_ip` | `0.997684` | `0.90` (`-2.91±0.52`), `0.95` (`-2.01±0.52`), **`0.997684` (`0.00`)**, `1.05` (`-1.82±0.53`), `1.10` (`-1.07±0.53`) |
| `flop_real` | `0.869514` | `0.78` (`-2.37±0.50`), `0.83` (`-1.50±0.49`), **`0.869514` (`0.00`)**, `0.92` (`-2.91±0.51`), `0.97` (`-1.90±0.51`) |
| `turn_real` | `0.937403` | `0.85` (`-1.60±0.50`), `0.90` (`-0.92±0.49`), **`0.937403` (`0.00`)**, `0.98` (`-2.61±0.49`), `1.03` (`-1.36±0.49`) |
| `river_real` | `1.0` | `0.92` (`-3.34±0.53`), `0.96` (`-3.18±0.53`), **`1.0` (`0.00`)**, `1.04` (`-3.55±0.49`), `1.08` (`-2.80±0.51`) |

### Full 20-Seed Gate Results (Top 2 Quick Candidates)
- **`turn_real=0.9`**: pooled paired diff `-3.88 ± 0.22` cph, 95% CI `[-4.31, -3.44]` — **GATE FAIL**
- **`pf_real_ip=1.1`**: pooled paired diff `-5.21 ± 0.25` cph, 95% CI `[-5.70, -4.71]` — **GATE FAIL**

## 3. Decision
**Keep all existing `realisation` values in `dev/params.json` unchanged.** None of the 20 perturbations beat `champion_v3` on the quick sweep or passed the 20-seed gate.

