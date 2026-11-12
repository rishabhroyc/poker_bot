#!/usr/bin/env python3.11
"""Parse bot stderr instrumentation logs and report p50/p90/p99/max latency per decision type."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_stderr(path: Path):
    by_type = defaultdict(list)  # dtype -> list of (n_samples, ms, level)
    total_ms_marker = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "D" and len(parts) >= 4:
            dtype = parts[1]
            n_samp = int(parts[2])
            ms = float(parts[3])
            lvl = int(parts[4]) if len(parts) >= 5 else 0
            by_type[dtype].append((n_samp, ms, lvl))
        elif parts[0] == "T" and len(parts) >= 2:
            total_ms_marker = float(parts[1])
    return by_type, total_ms_marker, path.stat().st_size


def summarize_file(path: Path) -> None:
    by_type, total_ms_marker, nbytes = parse_stderr(path)
    all_ms = [ms for rows in by_type.values() for _, ms, _ in rows]
    sum_ms = float(sum(all_ms))
    tot = total_ms_marker if total_ms_marker is not None else sum_ms
    print(f"=== {path} ({nbytes} bytes / 65536 cap) ===")
    print(f"  Decisions: {len(all_ms)} | Sum decision time: {sum_ms/1000.0:.3f} s | Reported total: {tot/1000.0:.3f} s")
    header = f"  {'Type':<10} {'Count':>6} {'MeanN':>7} {'p50(ms)':>9} {'p90(ms)':>9} {'p99(ms)':>9} {'max(ms)':>9}  Levels"
    print(header)
    for dtype in ("preflop", "flop", "turn", "river", "discard"):
        rows = by_type.get(dtype)
        if not rows:
            continue
        ns = np.array([r[0] for r in rows], dtype=np.float64)
        ms = np.array([r[1] for r in rows], dtype=np.float64)
        lvls = defaultdict(int)
        for _, _, l in rows:
            lvls[l] += 1
        lvl_str = " ".join(f"L{k}:{v}" for k, v in sorted(lvls.items()))
        p50, p90, p99 = np.percentile(ms, [50, 90, 99])
        print(
            f"  {dtype:<10} {len(rows):>6d} {ns.mean():>7.1f} "
            f"{p50:>9.2f} {p90:>9.2f} {p99:>9.2f} {ms.max():>9.2f}  {lvl_str}"
        )


def main():
    parser = argparse.ArgumentParser(description="Latency report from bot stderr logs")
    parser.add_argument("files", nargs="*", type=Path, help="Stderr log files")
    parser.add_argument("--log-dir", type=Path, help="Match log directory (reads A_leg*_stderr.txt)")
    parser.add_argument("--side", default="A", choices=["A", "B"], help="Bot side for --log-dir")
    args = parser.parse_args()

    paths = list(args.files)
    if args.log_dir:
        paths.extend(sorted(args.log_dir.glob(f"{args.side}_leg*_stderr.txt")))
    if not paths:
        raise SystemExit("No stderr log files provided.")
    for p in paths:
        if p.exists():
            summarize_file(p)
        else:
            print(f"Missing: {p}")


if __name__ == "__main__":
    main()

