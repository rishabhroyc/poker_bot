#!/usr/bin/env python3.11
"""Parallel paired-deal gauntlet runner and statistical gate for Discard Hold'em."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


PAIR_LINE_RE = re.compile(r"^L([12])\s+#(\d+)\s+.*\b([AB])\s+([+-]?\d+),\s+([AB])\s+([+-]?\d+)\s*$")
SUMMARY_A_RE = re.compile(
    r"^\s*A\s+\S+\s+net\s+([+-]?\d+)\s+time used\s+([\d.]+)\s+s\s+\(bank\s+(\d+)\s+s per leg\)\s+notices\s+(\d+)",
    re.MULTILINE,
)


def resolve_opponents(opps_arg: str) -> list[Path]:
    all_opps = sorted(Path("opps").glob("*.py"))
    if opps_arg.strip().lower() == "all":
        return all_opps
    wanted = [x.strip() for x in opps_arg.split(",") if x.strip()]
    selected = []
    for w in wanted:
        matches = [p for p in all_opps if p.stem == w or p.name == w or w in p.stem]
        if not matches:
            raise SystemExit(f"Opponent not found in opps/: {w!r}")
        selected.append(matches[0])
    return selected


def _read_stderr_total_s(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "T":
            try:
                return float(parts[1]) / 1000.0
            except ValueError:
                pass
    return None


def parse_match_log(log_dir: Path) -> tuple[list[int], float | None]:
    match_log = log_dir / "match.log"
    leg1, leg2 = {}, {}
    if match_log.exists():
        for line in match_log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = PAIR_LINE_RE.match(line)
            if m:
                leg = int(m.group(1))
                hand_idx = int(m.group(2))
                p1, v1, p2, v2 = m.group(3), int(m.group(4)), m.group(5), int(m.group(6))
                net_a = v1 if p1 == "A" else v2
                if leg == 1:
                    leg1[hand_idx] = net_a
                else:
                    leg2[hand_idx] = net_a
    common = sorted(set(leg1) & set(leg2))
    pair_nets = [leg1[i] + leg2[i] for i in common]

    t1 = _read_stderr_total_s(log_dir / "A_leg1_stderr.txt")
    t2 = _read_stderr_total_s(log_dir / "A_leg2_stderr.txt")
    max_leg_from_stderr = max(t1, t2) if (t1 is not None and t2 is not None) else (t1 or t2)
    return pair_nets, max_leg_from_stderr


def run_one(
    py_exe: str,
    bot_path: str,
    opp_path: Path,
    stage: int,
    seed: int,
    tag: str,
) -> dict:
    log_dir = Path("logs") / tag / f"{opp_path.stem}_s{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py_exe,
        "starter_kit/engine.py",
        bot_path,
        str(opp_path),
        "--stage",
        str(stage),
        "--seed",
        str(seed),
        "--log-dir",
        str(log_dir),
        "--python",
        py_exe,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + "\n" + res.stderr
    m = SUMMARY_A_RE.search(out)
    if not m:
        raise RuntimeError(f"Failed to parse engine output for {opp_path.name} seed={seed}:\n{out}")
    net = int(m.group(1))
    time_total_s = float(m.group(2))
    bank_s = int(m.group(3))
    notices = int(m.group(4))
    forfeit = 1 if "forfeit" in out.lower() else 0

    pair_nets, max_leg_stderr = parse_match_log(log_dir)
    max_leg_s = max_leg_stderr if max_leg_stderr is not None else (time_total_s / 2.0)
    arr = np.array(pair_nets, dtype=np.float64)
    n_pairs = len(arr)
    mean_cph = float(arr.mean() / 2.0) if n_pairs > 0 else 0.0
    se_cph = float((arr.std(ddof=1) / np.sqrt(n_pairs)) / 2.0) if n_pairs > 1 else 0.0

    return {
        "opp": opp_path.stem,
        "seed": seed,
        "net": net,
        "hands": n_pairs * 2,
        "pairs": n_pairs,
        "mean_cph": mean_cph,
        "se_cph": se_cph,
        "time_total_s": time_total_s,
        "avg_leg_s": time_total_s / 2.0,
        "max_leg_s": max_leg_s,
        "bank_s": bank_s,
        "notices": notices,
        "forfeit": forfeit,
        "win": 1 if net > 0 else 0,
        "draw": 1 if net == 0 else 0,
        "loss": 1 if net < 0 else 0,
        "pair_nets": pair_nets,
    }


def summarize_group(rows: list[dict]) -> dict:
    all_pairs = np.array([x for r in rows for x in r["pair_nets"]], dtype=np.float64)
    n_pairs = len(all_pairs)
    mean_cph = float(all_pairs.mean() / 2.0) if n_pairs > 0 else 0.0
    se_cph = float((all_pairs.std(ddof=1) / np.sqrt(n_pairs)) / 2.0) if n_pairs > 1 else 0.0
    ci_lo = mean_cph - 1.96 * se_cph
    ci_hi = mean_cph + 1.96 * se_cph
    wins = sum(r["win"] for r in rows)
    draws = sum(r["draw"] for r in rows)
    losses = sum(r["loss"] for r in rows)
    max_leg_s = max((r["max_leg_s"] for r in rows), default=0.0)
    notices = sum(r["notices"] for r in rows)
    forfeits = sum(r["forfeit"] for r in rows)
    return {
        "matches": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "pairs": n_pairs,
        "hands": n_pairs * 2,
        "mean_cph": mean_cph,
        "se_cph": se_cph,
        "ci95": [ci_lo, ci_hi],
        "max_leg_s": max_leg_s,
        "notices": notices,
        "forfeits": forfeits,
    }


def compare_with_baseline(cand_rows: list[dict], base_path: Path, stage: int) -> bool:
    base_data = json.loads(base_path.read_text(encoding="utf-8"))
    base_map = {(r["opp"], r["seed"]): r["pair_nets"] for r in base_data["matches"]}
    cand_map = {(r["opp"], r["seed"]): r for r in cand_rows}

    opps = sorted({r["opp"] for r in cand_rows})
    pooled_diffs = []
    gate_ok = True
    max_allowed_leg_s = 0.40 * (30.0 if stage == 1 else 75.0)

    print("\n=== Paired Comparison vs Baseline ===")
    print(f"Baseline: {base_path}")
    print(f"{'Opponent':<25} {'Pairs':>7} {'Diff(cph)':>10} {'SE':>7} {'95% CI':>19} {'Status':>8}")

    for opp in opps:
        opp_diffs = []
        for (o, s), r in sorted(cand_map.items()):
            if o != opp or (o, s) not in base_map:
                continue
            c_p = r["pair_nets"]
            b_p = base_map[(o, s)]
            n = min(len(c_p), len(b_p))
            if n > 0:
                opp_diffs.extend((np.array(c_p[:n]) - np.array(b_p[:n])).tolist())
        if not opp_diffs:
            continue
        arr = np.array(opp_diffs, dtype=np.float64)
        mean_d = float(arr.mean() / 2.0)
        se_d = float((arr.std(ddof=1) / np.sqrt(len(arr))) / 2.0) if len(arr) > 1 else 0.0
        lo, hi = mean_d - 1.96 * se_d, mean_d + 1.96 * se_d
        regressed = mean_d < -2.0 and hi < 0.0
        if regressed:
            gate_ok = False
        status = "FAIL" if regressed else "OK"
        print(f"{opp:<25} {len(arr):>7d} {mean_d:>+10.2f} {se_d:>7.2f} [{lo:+7.2f}, {hi:+7.2f}] {status:>8}")
        pooled_diffs.extend(opp_diffs)

    if not pooled_diffs:
        print("GATE FAIL: No matching (opponent, seed) pairs found in baseline.")
        return False

    p_arr = np.array(pooled_diffs, dtype=np.float64)
    p_mean = float(p_arr.mean() / 2.0)
    p_se = float((p_arr.std(ddof=1) / np.sqrt(len(p_arr))) / 2.0)
    p_lo, p_hi = p_mean - 1.96 * p_se, p_mean + 1.96 * p_se
    print("-" * 82)
    print(f"{'POOLED':<25} {len(p_arr):>7d} {p_mean:>+10.2f} {p_se:>7.2f} [{p_lo:+7.2f}, {p_hi:+7.2f}]")

    total_notices = sum(r["notices"] for r in cand_rows)
    total_forfeits = sum(r["forfeit"] for r in cand_rows)
    max_leg_s = max((r["max_leg_s"] for r in cand_rows), default=0.0)

    gates = [
        ("Pooled CI lower > 0", p_lo > 0.0, f"{p_lo:+.2f}"),
        ("No opp < -2 cph (CI < 0)", gate_ok, "OK" if gate_ok else "REGRESSED"),
        ("Zero notices & forfeits", total_notices == 0 and total_forfeits == 0, f"notices={total_notices}, forfeits={total_forfeits}"),
        (f"Max leg time <= {max_allowed_leg_s:.1f}s", max_leg_s <= max_allowed_leg_s, f"{max_leg_s:.2f}s"),
    ]
    print("\n=== Gate Checks ===")
    all_pass = True
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            all_pass = False
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Paired-deal gauntlet runner")
    parser.add_argument("--bot", default="dev/bot.py")
    parser.add_argument("--opps", default="all")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="Run 3 seeds")
    parser.add_argument("--full", action="store_true", help="Run at least 20 seeds")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tag", default="gauntlet")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()

    n_seeds = 3 if args.quick else (max(20, args.seeds) if args.full else args.seeds)
    seeds = list(range(args.seed_start, args.seed_start + n_seeds))
    opps = resolve_opponents(args.opps)

    py_exe = args.python
    if py_exe == "python3.11" and not Path(py_exe).exists():
        # Fall back to current python3.11 interpreter if python3.11 is not on PATH
        py_exe = sys.executable

    tasks = [(opp, s) for opp in opps for s in seeds]
    print(f"Gauntlet: bot={args.bot}, stage={args.stage}, opps={len(opps)}, seeds={n_seeds}, jobs={args.jobs}, total_matches={len(tasks)}")
    t0 = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, py_exe, args.bot, opp, args.stage, s, args.tag): (opp, s) for opp, s in tasks}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                print(f"  Progress: {done}/{len(tasks)} matches ({time.perf_counter() - t0:.1f}s)", flush=True)
    elapsed_s = time.perf_counter() - t0

    rows.sort(key=lambda r: (r["opp"], r["seed"]))
    by_opp = {}
    for r in rows:
        by_opp.setdefault(r["opp"], []).append(r)

    print("\n=== Gauntlet Results ===")
    print(f"{'Opponent':<25} {'W-D-L':>9} {'Pairs':>7} {'Mean(cph)':>10} {'SE':>7} {'95% CI':>19} {'MaxLeg(s)':>10} {'Not':>4}")
    opp_summaries = {}
    for opp, rlist in sorted(by_opp.items()):
        st = summarize_group(rlist)
        opp_summaries[opp] = st
        wdl = f"{st['wins']}-{st['draws']}-{st['losses']}"
        ci_str = f"[{st['ci95'][0]:+7.2f}, {st['ci95'][1]:+7.2f}]"
        print(
            f"{opp:<25} {wdl:>9} {st['pairs']:>7d} {st['mean_cph']:>+10.2f} "
            f"{st['se_cph']:>7.2f} {ci_str:>19} {st['max_leg_s']:>10.2f} {st['notices']:>4d}"
        )

    pooled = summarize_group(rows)
    wdl = f"{pooled['wins']}-{pooled['draws']}-{pooled['losses']}"
    ci_str = f"[{pooled['ci95'][0]:+7.2f}, {pooled['ci95'][1]:+7.2f}]"
    print("-" * 97)
    print(
        f"{'POOLED':<25} {wdl:>9} {pooled['pairs']:>7d} {pooled['mean_cph']:>+10.2f} "
        f"{pooled['se_cph']:>7.2f} {ci_str:>19} {pooled['max_leg_s']:>10.2f} {pooled['notices']:>4d}"
    )
    print(f"Wall time: {elapsed_s:.2f}s (jobs={args.jobs}, cpu_count={os.cpu_count()})")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "opp", "seed", "net", "hands", "pairs", "mean_cph", "se_cph",
                "time_total_s", "avg_leg_s", "max_leg_s", "bank_s", "notices", "forfeit",
                "win", "draw", "loss",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    json_path = results_dir / f"{args.tag}.json"
    summary_obj = {
        "tag": args.tag,
        "bot": args.bot,
        "stage": args.stage,
        "seeds": seeds,
        "jobs": args.jobs,
        "wall_time_s": round(elapsed_s, 2),
        "pooled": pooled,
        "opponents": opp_summaries,
        "matches": rows,
    }
    json_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    print(f"Saved {csv_path} and {json_path}")

    if args.baseline is not None:
        ok = compare_with_baseline(rows, args.baseline, args.stage)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()

