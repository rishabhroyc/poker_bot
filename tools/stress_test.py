#!/usr/bin/env python3.11
"""Stress-test dev/bot.py pinned to 1 CPU core with 1x, 2x (1 CPU hog), and 3x (2 CPU hogs) contention."""

from __future__ import annotations

import argparse
import ctypes
import re
import subprocess
import sys
import time
from pathlib import Path


def pin_to_cpu0() -> None:
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype = ctypes.c_int
        handle = k32.GetCurrentProcess()
        ok = k32.SetProcessAffinityMask(handle, 1)
        if not ok:
            raise OSError(f"SetProcessAffinityMask failed: {ctypes.GetLastError()}")
    elif hasattr(os, "sched_setaffinity"):
        import os as _os
        _os.sched_setaffinity(0, {0})


import os  # noqa: E402


def spawn_hogs(n_hogs: int) -> list[subprocess.Popen]:
    hogs = []
    for _ in range(n_hogs):
        p = subprocess.Popen(
            [sys.executable, "-c", "while True: pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        hogs.append(p)
    return hogs


def run_match(bot_path: str, opp_path: str, stage: int, seed: int, log_dir: Path) -> tuple[int, float, int, int, str]:
    cmd = [
        sys.executable,
        "starter_kit/engine.py",
        bot_path,
        opp_path,
        "--stage",
        str(stage),
        "--seed",
        str(seed),
        "--log-dir",
        str(log_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    # Parse A's line: A  <label>  net  +15229   time used  16.44 s (bank 30 s per leg)   notices 0
    m = re.search(
        r"^\s*A\s+\S+\s+net\s+([+-]?\d+)\s+time used\s+([\d.]+)\s+s\s+\(bank\s+(\d+)\s+s per leg\)\s+notices\s+(\d+)",
        out,
        re.MULTILINE,
    )
    if not m:
        raise RuntimeError(f"Could not parse engine output:\n{out}")
    net = int(m.group(1))
    time_s = float(m.group(2))
    bank_s = int(m.group(3))
    notices = int(m.group(4))
    forfeit = 1 if "forfeit" in out.lower() else 0
    return net, time_s, bank_s, notices, forfeit, out


def main():
    parser = argparse.ArgumentParser(description="1-core CPU contention stress test")
    parser.add_argument("--bot", default="dev/bot.py")
    parser.add_argument("--opp", default="starter_kit/bot.py")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--slowdowns", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    pin_to_cpu0()
    print(f"Pinned to 1 core (CPU 0). Testing {args.bot} vs {args.opp} (seed {args.seed})")
    print(f"{'Stage':>5} {'Load':>5} {'Net':>8} {'TimeTotal(s)':>13} {'AvgLeg(s)':>10} {'BankPct':>8} {'Notices':>8} {'Forfeit':>8}")

    for stage in args.stages:
        for slow in args.slowdowns:
            n_hogs = max(0, slow - 1)
            hogs = spawn_hogs(n_hogs)
            time.sleep(0.1)
            try:
                log_dir = Path(f"logs/stress_s{stage}_{slow}x")
                net, time_s, bank_s, notices, forfeit, _ = run_match(args.bot, args.opp, stage, args.seed, log_dir)
                avg_leg = time_s / 2.0
                bank_pct = 100.0 * avg_leg / bank_s
                print(
                    f"{stage:>5d} {slow:>4d}x {net:>+8d} {time_s:>13.2f} {avg_leg:>10.2f} "
                    f"{bank_pct:>7.1f}% {notices:>8d} {forfeit:>8d}",
                    flush=True,
                )
            finally:
                for h in hogs:
                    h.terminate()
                for h in hogs:
                    h.wait(timeout=5)


if __name__ == "__main__":
    main()
