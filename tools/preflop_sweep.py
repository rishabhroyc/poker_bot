#!/usr/bin/env python3.11
"""One-parameter preflop gate sweep against aggressive/exploitative zoo + full 20-seed gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


AGGR_OPPS = "04_maniac,07_lag,10_overbet_shover,12_adaptive_exploiter"

SWEEPS = [
    ("preflop_3bet_pct", [0.45, 0.54, 0.62, 0.70, 0.78]),
    ("preflop_shove_pct", [0.82, 0.87, 0.914578, 0.95, 0.98]),
    ("preflop_call_def_pct", [0.25, 0.35, 0.45, 0.55, 0.65]),
    ("preflop_open_sizes", [[4, 6], [5, 7], [6, 9], [5]]),
    ("preflop_3bet_mults", [[2.5, 3.5], [3, 4], [3.5, 4.5]]),
    ("preflop_limp_max_pct", [0.0, 0.48, 0.58]),
    ("pos_open_min", [(0.0, 0.0), (0.25, 0.45), (0.35, 0.50)]),
]


def paired_diff(cand_json: Path, base_map: dict) -> tuple[float, float, float, float]:
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
    m = float(arr.mean() / 2.0)
    se = float((arr.std(ddof=1) / np.sqrt(len(arr))) / 2.0)
    return m, se, m - 1.96 * se, m + 1.96 * se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("results/champion_v3.json"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    base_params = json.loads(Path("dev/params.json").read_text(encoding="utf-8"))
    base_data = json.loads(args.baseline.read_text(encoding="utf-8"))
    base_map = {(r["opp"], r["seed"]): r["pair_nets"] for r in base_data["matches"]}

    tmp = Path("results/sweep_params")
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, (param, vals) in enumerate(SWEEPS):
        for v_idx, v in enumerate(vals):
            tag = f"pf_{param}_{v_idx}"
            p_copy = json.loads(json.dumps(base_params))
            is_base = False
            if param == "pos_open_min":
                if v == (0.0, 0.0):
                    is_base = True
                else:
                    p_copy["bet_sizing"]["preflop_btn_open_min_pct"] = v[0]
                    p_copy["bet_sizing"]["preflop_bb_open_min_pct"] = v[1]
            else:
                cur = p_copy["bet_sizing"].get(param, {"preflop_3bet_pct": 0.62, "preflop_call_def_pct": 0.45, "preflop_limp_max_pct": 0.0}.get(param))
                if v == cur:
                    is_base = True
                else:
                    p_copy["bet_sizing"][param] = v

            if is_base:
                rows.append({"param": param, "value": v, "is_base": True, "diff": 0.0, "se": 0.0, "ci": [0.0, 0.0], "tag": "base"})
                print(f"[BASE] {param}={v}", flush=True)
                continue

            pf = (tmp / f"{tag}.json").resolve()
            pf.write_text(json.dumps(p_copy, indent=2), encoding="utf-8")
            env = os.environ.copy()
            env["PB_PARAMS_FILE"] = str(pf)
            cmd = [
                sys.executable, "tools/gauntlet.py",
                "--bot", "dev/bot.py",
                "--opps", AGGR_OPPS,
                "--stage", "1",
                "--quick",
                "--jobs", str(args.jobs),
                "--tag", tag,
            ]
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
            m, se, lo, hi = paired_diff(Path("results") / f"{tag}.json", base_map)
            rows.append({"param": param, "value": v, "is_base": False, "diff": m, "se": se, "ci": [lo, hi], "tag": tag, "p_file": str(pf)})
            print(f"[QUICK] {param}={str(v):<14} diff={m:+.2f} +/- {se:.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)

    Path("results/preflop_sweep_quick.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Run full 20-seed gate on any candidate with quick diff > 0 (or top 1)
    cands = sorted([r for r in rows if not r["is_base"] and r["diff"] > 0.0], key=lambda x: -x["diff"])
    print(f"\nCandidates beating baseline on quick aggr zoo: {len(cands)}")
    full_res = []
    for c in cands[:2]:
        tag = f"full_{c['tag']}"
        env = os.environ.copy()
        env["PB_PARAMS_FILE"] = c["p_file"]
        cmd = [
            sys.executable, "tools/gauntlet.py",
            "--bot", "dev/bot.py",
            "--opps", "01_starter_clone,02_random_bot,03_calling_station,04_maniac,05_nit,06_tag,07_lag,08_trapper,09_min_bettor,10_overbet_shover,11_over_folder,12_adaptive_exploiter,13_llm_generic",
            "--stage", "1",
            "--full",
            "--jobs", str(args.jobs),
            "--tag", tag,
            "--baseline", str(args.baseline),
        ]
        print(f"[FULL] Testing {c['param']}={c['value']}...", flush=True)
        res = subprocess.run(cmd, env=env)
        m, se, lo, hi = paired_diff(Path("results") / f"{tag}.json", base_map)
        full_res.append({"param": c["param"], "value": c["value"], "diff": m, "se": se, "ci": [lo, hi], "passed_gate": res.returncode == 0})
        print(f"[FULL] {c['param']}={c['value']}: diff={m:+.2f} +/- {se:.2f} [{lo:+.2f}, {hi:+.2f}] gate={'PASS' if res.returncode == 0 else 'FAIL'}", flush=True)

    Path("results/preflop_sweep_full.json").write_text(json.dumps(full_res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

