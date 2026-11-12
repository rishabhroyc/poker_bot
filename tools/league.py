#!/usr/bin/env python3.11
"""Round-robin league runner across submissions/*.py and opps/*.py with Bradley-Terry + bootstrap CIs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


SUMMARY_AB_RE = re.compile(
    r"^\s*([AB])\s+\S+\s+net\s+([+-]?\d+)\s+time used\s+([\d.]+)\s+s\s+\(bank\s+\d+\s+s per leg\)\s+notices\s+(\d+)",
    re.MULTILINE,
)


def discover_players() -> list[tuple[str, Path]]:
    players = []
    for folder in ("submissions", "opps"):
        pdir = Path(folder)
        if not pdir.exists():
            continue
        for p in sorted(pdir.glob("*.py")):
            players.append((f"{folder}/{p.stem}", p))
    return players


def run_match(py_exe: str, name_a: str, path_a: Path, name_b: str, path_b: Path, seed: int, tag: str) -> dict:
    log_dir = Path("logs") / tag / f"{name_a.replace('/', '_')}__vs__{name_b.replace('/', '_')}_s{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py_exe,
        "starter_kit/engine.py",
        str(path_a),
        str(path_b),
        "--stage",
        "1",
        "--seed",
        str(seed),
        "--log-dir",
        str(log_dir),
        "--python",
        py_exe,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + "\n" + res.stderr
    matches = {m.group(1): (int(m.group(2)), float(m.group(3)), int(m.group(4))) for m in SUMMARY_AB_RE.finditer(out)}
    if "A" not in matches or "B" not in matches:
        raise RuntimeError(f"Failed to parse match {name_a} vs {name_b} seed={seed}:\n{out}")
    net_a, time_a, not_a = matches["A"]
    net_b, time_b, not_b = matches["B"]
    return {
        "a": name_a,
        "b": name_b,
        "seed": seed,
        "net_a": net_a,
        "net_b": net_b,
        "time_a": time_a,
        "time_b": time_b,
        "notices_a": not_a,
        "notices_b": not_b,
        "forfeit": 1 if "forfeit" in out.lower() else 0,
    }


def fit_bradley_terry_elo(n_players: int, records: list[tuple[int, int, float]], iters: int = 200) -> np.ndarray:
    """MM algorithm for Bradley-Terry with tiny Laplace smoothing; returns Elo-scaled ratings (mean 1500)."""
    wins = np.zeros((n_players, n_players), dtype=np.float64)
    games = np.zeros((n_players, n_players), dtype=np.float64)
    for i, j, score_i in records:
        wins[i, j] += score_i
        wins[j, i] += 1.0 - score_i
        games[i, j] += 1.0
        games[j, i] += 1.0
    # Tiny prior (0.05) keeps disconnected/0-loss players finite
    wins += 0.05
    games += 0.10
    p = np.ones(n_players, dtype=np.float64)
    for _ in range(iters):
        p_sum = p[:, None] + p[None, :]
        denom = (games / p_sum).sum(axis=1)
        w_i = wins.sum(axis=1)
        p_new = w_i / np.maximum(denom, 1e-12)
        p_new /= np.exp(np.mean(np.log(np.maximum(p_new, 1e-12))))
        if np.max(np.abs(p_new - p)) < 1e-8:
            p = p_new
            break
        p = p_new
    elo = 1500.0 + (400.0 / np.log(10.0)) * np.log(np.maximum(p, 1e-12))
    return elo - np.mean(elo) + 1500.0


def find_cycles(names: list[str], h2h_scores: dict[tuple[int, int], list[float]]) -> list[tuple[str, str, str]]:
    """Find directed 3-cycles A > B > C > A where '>' means head-to-head match score > 0.5."""
    n = len(names)
    beats = [[False] * n for _ in range(n)]
    for (i, j), scs in h2h_scores.items():
        if not scs:
            continue
        avg = sum(scs) / len(scs)
        if avg > 0.5:
            beats[i][j] = True
        elif avg < 0.5:
            beats[j][i] = True
    cycles = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(a + 1, n):
                if b == c:
                    continue
                if beats[a][b] and beats[b][c] and beats[c][a]:
                    cycles.append((names[a], names[b], names[c]))
    return cycles


def main():
    parser = argparse.ArgumentParser(description="Round-robin league across submissions/ and opps/")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per pairing")
    parser.add_argument("--jobs", type=int, default=8, help="Max concurrent matches (capped to cpu_count // 2)")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tag", default=f"league_{date.today().isoformat()}")
    parser.add_argument("--bootstrap", type=int, default=400)
    args = parser.parse_args()

    players = discover_players()
    if len(players) < 2:
        raise SystemExit("Need at least 2 players across submissions/ and opps/.")

    cpu_cnt = os.cpu_count() or 2
    max_safe_jobs = max(1, cpu_cnt // 2)
    jobs = min(args.jobs, max_safe_jobs)
    py_exe = args.python
    if py_exe == "python3.11" and not Path(py_exe).exists():
        py_exe = sys.executable

    pairings = list(itertools.combinations(range(len(players)), 2))
    tasks = [(i, j, s) for (i, j) in pairings for s in range(1, args.seeds + 1)]
    print(
        f"League: {len(players)} players, {len(pairings)} pairings, {args.seeds} seeds = {len(tasks)} matches "
        f"(jobs={jobs}, cpu_count={cpu_cnt})"
    )

    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {
            ex.submit(run_match, py_exe, players[i][0], players[i][1], players[j][0], players[j][1], s, args.tag): (i, j, s)
            for i, j, s in tasks
        }
        done = 0
        for fut in as_completed(futs):
            i, j, s = futs[fut]
            r = fut.result()
            r["i"], r["j"] = i, j
            results.append(r)
            done += 1
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                print(f"  Progress: {done}/{len(tasks)} matches ({time.perf_counter() - t0:.1f}s)", flush=True)

    wall_s = time.perf_counter() - t0
    names = [p[0] for p in players]
    n = len(names)

    records = []
    h2h = {}
    stats = [{"w": 0, "d": 0, "l": 0, "net": 0, "not": 0, "max_leg": 0.0} for _ in range(n)]
    for r in results:
        i, j = r["i"], r["j"]
        if r["net_a"] > 0:
            sc_i = 1.0
            stats[i]["w"] += 1
            stats[j]["l"] += 1
        elif r["net_a"] < 0:
            sc_i = 0.0
            stats[i]["l"] += 1
            stats[j]["w"] += 1
        else:
            sc_i = 0.5
            stats[i]["d"] += 1
            stats[j]["d"] += 1
        stats[i]["net"] += r["net_a"]
        stats[j]["net"] += r["net_b"]
        stats[i]["not"] += r["notices_a"]
        stats[j]["not"] += r["notices_b"]
        stats[i]["max_leg"] = max(stats[i]["max_leg"], r["time_a"] / 2.0)
        stats[j]["max_leg"] = max(stats[j]["max_leg"], r["time_b"] / 2.0)
        records.append((i, j, sc_i))
        h2h.setdefault((i, j), []).append(sc_i)

    elo = fit_bradley_terry_elo(n, records)
    rng = np.random.default_rng(42)
    boot_elos = np.empty((args.bootstrap, n), dtype=np.float64)
    m_cnt = len(records)
    for b in range(args.bootstrap):
        idxs = rng.integers(0, m_cnt, size=m_cnt)
        sample = [records[k] for k in idxs]
        boot_elos[b] = fit_bradley_terry_elo(n, sample)
    elo_sd = boot_elos.std(axis=0, ddof=1)
    elo_lo = np.percentile(boot_elos, 2.5, axis=0)
    elo_hi = np.percentile(boot_elos, 97.5, axis=0)

    cycles = find_cycles(names, h2h)
    order = sorted(range(n), key=lambda k: (-elo[k], -stats[k]["net"]))

    lines = [
        f"# League Results ({args.tag})",
        "",
        f"- **Format**: Stage 1 round robin, {len(players)} players, {args.seeds} seeds/pairing ({len(results)} matches total)",
        f"- **Execution**: `jobs={jobs}` (capped at `cpu_count // 2 = {max_safe_jobs}` on {cpu_cnt} logical cores), wall time `{wall_s:.1f} s`",
        "",
        "| Rank | Player | W-D-L | Win% | BT Elo | 95% Bootstrap CI | Avg Net/Match | Max AvgLeg(s) | Notices |",
        "| ---: | --- | :---: | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for rank, idx in enumerate(order, 1):
        st = stats[idx]
        tot = st["w"] + st["d"] + st["l"]
        wr = 100.0 * (st["w"] + 0.5 * st["d"]) / max(1, tot)
        avg_net = st["net"] / max(1, tot)
        lines.append(
            f"| {rank} | `{names[idx]}` | {st['w']}-{st['d']}-{st['l']} | {wr:.1f}% | "
            f"{elo[idx]:.0f} ± {elo_sd[idx]:.0f} | [{elo_lo[idx]:.0f}, {elo_hi[idx]:.0f}] | "
            f"{avg_net:+.0f} | {st['max_leg']:.2f} | {st['not']} |"
        )

    lines.extend(["", "## Intransitive Cycles (A > B > C > A)"])
    if cycles:
        for a, b, c in cycles[:20]:
            lines.append(f"- `{a}` > `{b}` > `{c}` > `{a}`")
        if len(cycles) > 20:
            lines.append(f"- ... and {len(cycles) - 20} more")
    else:
        lines.append("- None detected.")

    top_name = names[order[0]]
    lines.extend([
        "",
        "## Interpretation",
        f"- Top-ranked player is `{top_name}` with BT Elo `{elo[order[0]]:.0f}` (win rate `{100.0 * (stats[order[0]]['w'] + 0.5 * stats[order[0]]['d']) / max(1, stats[order[0]]['w'] + stats[order[0]]['d'] + stats[order[0]]['l']):.1f}%`).",
        f"- Detected `{len(cycles)}` intransitive 3-cycle(s) across the pool; rock-paper-scissors dynamics among zoo archetypes highlight why multi-opponent round-robin testing is needed alongside single-bot gauntlets.",
    ])

    out_md = Path("results") / f"{args.tag}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json = Path("results") / f"{args.tag}.json"
    out_json.write_text(json.dumps({"matches": results, "elo": elo.tolist(), "cycles": cycles}, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_md} and {out_json}")


if __name__ == "__main__":
    main()

