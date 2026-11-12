#!/usr/bin/env python3.11
"""Run 20,000 simulated hands (50 Stage 1 matches) with PB_CALIB_LOG=1 and measure equity calibration."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OPPS = [
    "opps/03_calling_station.py",
    "opps/06_tag.py",
    "opps/07_lag.py",
    "opps/08_trapper.py",
    "opps/13_llm_generic.py",
]


def run_one(py_exe: str, bot: str, opp: str, seed: int, log_root: Path) -> Path:
    opp_name = Path(opp).stem
    log_dir = log_root / f"{opp_name}_s{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PB_CALIB_LOG"] = "1"
    cmd = [
        py_exe,
        "starter_kit/engine.py",
        bot,
        opp,
        "--stage",
        "1",
        "--seed",
        str(seed),
        "--log-dir",
        str(log_dir),
        "--python",
        py_exe,
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Match failed {opp} seed={seed}: {res.stderr}")
    return log_dir


def main():
    parser = argparse.ArgumentParser(description="Equity calibration over 20,000 hands")
    parser.add_argument("--bot", default="dev/bot.py")
    parser.add_argument("--seeds-per-opp", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=Path("results/calibration_stats.json"))
    parser.add_argument("--out-png", type=Path, default=Path("results/calibration_plot.png"))
    args = parser.parse_args()

    log_root = Path("logs/calib_20k")
    tasks = [(opp, s) for opp in DEFAULT_OPPS for s in range(1, args.seeds_per_opp + 1)]
    print(f"Running {len(tasks)} Stage 1 matches ({len(tasks) * 400:,} hands) with PB_CALIB_LOG=1, jobs={args.jobs}")
    t0 = time.perf_counter()
    dirs = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_one, sys.executable, args.bot, opp, s, log_root) for opp, s in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            dirs.append(fut.result())
            if i % 10 == 0 or i == len(futs):
                print(f"  Progress: {i}/{len(futs)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    by_street = defaultdict(list)
    for d in dirs:
        for f in sorted(d.glob("A_leg*_stderr.txt")):
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split()
                if len(parts) == 4 and parts[0] == "C":
                    st, peq, y = parts[1], float(parts[2]), float(parts[3])
                    by_street[st].append((peq, y))

    stats = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    streets = ("preflop", "flop", "turn", "river")
    for ax, st in zip(axes.flat, streets):
        rows = by_street.get(st, [])
        arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 2))
        preds = arr[:, 0] if len(arr) else np.array([])
        ys = arr[:, 1] if len(arr) else np.array([])
        overall_bias = float((ys - preds).mean()) if len(arr) else 0.0
        bins_out = []
        bx, by_pred, by_real = [], [], []
        ece = 0.0
        for b in range(10):
            lo, hi = b * 0.1, (b + 1) * 0.1
            mask = (preds >= lo) & (preds <= hi if b == 9 else preds < hi)
            cnt = int(mask.sum())
            if cnt > 0:
                mp = float(preds[mask].mean())
                mr = float(ys[mask].mean())
                bias = mr - mp
                ece += (cnt / len(arr)) * abs(bias)
                bx.append((lo + hi) / 2.0)
                by_pred.append(mp)
                by_real.append(mr)
            else:
                mp, mr, bias = (lo + hi) / 2.0, None, None
            bins_out.append({"bin": f"[{lo:.1f},{hi:.1f}]", "count": cnt, "pred": mp, "realised": mr, "bias": bias})

        stats[st] = {"count": len(arr), "mean_pred": float(preds.mean()) if len(arr) else 0.0,
                     "mean_realised": float(ys.mean()) if len(arr) else 0.0,
                     "overall_bias": overall_bias, "ece": float(ece), "bins": bins_out}
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect (y=x)")
        if by_pred:
            ax.plot(by_pred, by_real, "o-", lw=2, label=f"{st} (n={len(arr):,}, bias={overall_bias:+.3f}, ECE={ece:.3f})")
        ax.set_title(st.capitalize())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.supxlabel("Predicted Equity")
    fig.supylabel("Realised Showdown Win Rate")
    fig.suptitle("Discard Hold'em Equity Calibration (20,000 Hands)")
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=140)
    args.out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    for st in streets:
        s = stats[st]
        print(f"\n=== {st.upper()} (n={s['count']:,}, mean_pred={s['mean_pred']:.4f}, mean_real={s['mean_realised']:.4f}, bias={s['overall_bias']:+.4f}, ECE={s['ece']:.4f}) ===")
        for b in s["bins"]:
            if b["count"] > 0:
                print(f"  {b['bin']} n={b['count']:>5d}  pred={b['pred']:.4f}  real={b['realised']:.4f}  bias={b['bias']:+.4f}")
    print(f"\nSaved {args.out_png} and {args.out_json}")


if __name__ == "__main__":
    main()

