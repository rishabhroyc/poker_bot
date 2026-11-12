#!/usr/bin/env python3.11
"""Sweep realisation parameters one at a time via tools/gauntlet.py."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


SWEEP_GRID = {
    "pf_real_oop": [0.80, 0.85, 0.894289, 0.94, 0.99],
    "pf_real_ip": [0.90, 0.95, 0.997684, 1.05, 1.10],
    "flop_real": [0.78, 0.83, 0.869514, 0.92, 0.97],
    "turn_real": [0.85, 0.90, 0.937403, 0.98, 1.03],
    "river_real": [0.92, 0.96, 1.0, 1.04, 1.08],
}


def paired_diff_vs_base(cand_json: Path, base_map: dict) -> tuple[float, float, float, float]:
    cdata = json.loads(cand_json.read_text(encoding="utf-8"))
    diffs = []
    for r in cdata["matches"]:
        key = (r["opp"], r["seed"])
        if key in base_map:
            cp, bp = r["pair_nets"], base_map[key]
            n = min(len(cp), len(bp))
            if n > 0:
                diffs.extend((np.array(cp[:n]) - np.array(bp[:n])).tolist())
    arr = np.array(diffs, dtype=np.float64)
    mean_d = float(arr.mean() / 2.0)
    se_d = float((arr.std(ddof=1) / np.sqrt(len(arr))) / 2.0)
    return mean_d, se_d, mean_d - 1.96 * se_d, mean_d + 1.96 * se_d


def main():
    parser = argparse.ArgumentParser(description="Realisation parameter sweep")
    parser.add_argument("--baseline", type=Path, default=Path("results/champion_v3.json"))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--opps", default="all")
    args = parser.parse_args()

    base_params = json.loads(Path("dev/params.json").read_text(encoding="utf-8"))
    base_data = json.loads(args.baseline.read_text(encoding="utf-8"))
    base_map = {(r["opp"], r["seed"]): r["pair_nets"] for r in base_data["matches"]}

    tmp_dir = Path("results/sweep_params")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows = []
    for param, vals in SWEEP_GRID.items():
        cur_val = base_params["realisation"][param]
        for v in vals:
            if abs(v - cur_val) < 1e-6:
                sweep_rows.append({"param": param, "value": v, "is_baseline": True, "diff": 0.0, "se": 0.0, "ci": [0.0, 0.0], "tag": "champion_v3"})
                print(f"[BASE] {param}={v}", flush=True)
                continue
            tag = f"sweep_{param}_{str(v).replace('.', 'p')}"
            p_copy = json.loads(json.dumps(base_params))
            p_copy["realisation"][param] = v
            p_file = (tmp_dir / f"{tag}.json").resolve()
            p_file.write_text(json.dumps(p_copy, indent=2), encoding="utf-8")

            env = os.environ.copy()
            env["PB_PARAMS_FILE"] = str(p_file)
            cmd = [
                sys.executable, "tools/gauntlet.py",
                "--bot", "dev/bot.py",
                "--opps", args.opps,
                "--stage", "1",
                "--quick",
                "--jobs", str(args.jobs),
                "--tag", tag,
            ]
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
            mean_d, se_d, lo, hi = paired_diff_vs_base(Path("results") / f"{tag}.json", base_map)
            sweep_rows.append({"param": param, "value": v, "is_baseline": False, "diff": mean_d, "se": se_d, "ci": [lo, hi], "tag": tag})
            print(f"[QUICK] {param}={v:<8} diff={mean_d:+.2f} +/- {se_d:.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)

    Path("results/realisation_sweep_quick.json").write_text(json.dumps(sweep_rows, indent=2), encoding="utf-8")

    # Pick top 2 non-baseline candidates by quick paired diff
    cands = sorted([r for r in sweep_rows if not r["is_baseline"]], key=lambda x: -x["diff"])
    top2 = cands[:2]
    full_results = []
    for c in top2:
        param, v = c["param"], c["value"]
        tag = f"full_{c['tag']}"
        p_file = (tmp_dir / f"{c['tag']}.json").resolve()
        env = os.environ.copy()
        env["PB_PARAMS_FILE"] = str(p_file)
        cmd = [
            sys.executable, "tools/gauntlet.py",
            "--bot", "dev/bot.py",
            "--opps", args.opps,
            "--stage", "1",
            "--full",
            "--jobs", str(args.jobs),
            "--tag", tag,
            "--baseline", str(args.baseline),
        ]
        print(f"\n[FULL] Running 20-seed gate for {param}={v}...", flush=True)
        res = subprocess.run(cmd, env=env)
        passed = (res.returncode == 0)
        mean_d, se_d, lo, hi = paired_diff_vs_base(Path("results") / f"{tag}.json", base_map)
        full_results.append({"param": param, "value": v, "diff": mean_d, "se": se_d, "ci": [lo, hi], "passed_gate": passed, "tag": tag})
        print(f"[FULL] {param}={v}: diff={mean_d:+.2f} +/- {se_d:.2f} [{lo:+.2f}, {hi:+.2f}] gate={'PASS' if passed else 'FAIL'}", flush=True)

    Path("results/realisation_sweep_full.json").write_text(json.dumps(full_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

