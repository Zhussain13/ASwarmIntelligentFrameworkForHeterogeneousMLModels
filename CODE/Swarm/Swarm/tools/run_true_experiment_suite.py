"""Run strict cross-backbone publication suite and aggregate results.

Usage:
  python tools/run_true_experiment_suite.py --output-root d:\\Swarm
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def _run(cmd, env=None, cwd=None):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def _collect_metrics(run_dir: Path):
    files = sorted((run_dir / "metrics").glob("metrics_agent_*.csv"))
    if not files:
        return None
    agg = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n = len(agg)
    heavy = int((agg["decision"] == "heavy").sum())
    skipped = int((agg["decision"] == "skipped").sum())
    withpkt = (agg["decision"] == "skipped") & agg["used_packet_agent"].notna()
    cross = withpkt & (agg["used_packet_agent"] != agg["agent_id"])
    sim = pd.to_numeric(agg.loc[agg["decision"] == "skipped", "sim"], errors="coerce").dropna()
    heavy_lat = pd.to_numeric(agg.loc[agg["decision"] == "heavy", "heavy_latency_ms"], errors="coerce").dropna()

    return {
        "total_decisions": n,
        "heavy_calls": heavy,
        "skipped_calls": skipped,
        "skip_rate_pct": (skipped / n) * 100 if n else 0.0,
        "cross_reuse_rate_pct": (cross.sum() / max(1, withpkt.sum())) * 100,
        "cross_reuse_count": int(cross.sum()),
        "skip_with_packet_count": int(withpkt.sum()),
        "sim_mean": float(sim.mean()) if len(sim) else float("nan"),
        "sim_std": float(sim.std()) if len(sim) else float("nan"),
        "heavy_latency_ms_mean": float(heavy_lat.mean()) if len(heavy_lat) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="d:\\Swarm")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--video", default="d:\\Swarm\\data\\video_for_swarm.avi")
    parser.add_argument("--seeds", default="7,17,27")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    ts = int(time.time())
    suite_dir = Path(args.output_root) / f"paper_results_true_suite_{ts}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    runner = Path("d:\\Swarm\\paper_results_hetero_1770271738\\artifacts\\run_three_agents_local.py")
    cwd = runner.parent
    env = os.environ.copy()
    env["TORCH_HOME"] = str(Path(args.output_root) / ".torch_cache")

    base = [
        sys.executable,
        str(runner),
        "--num-agents",
        "3",
        "--mode",
        "video",
        "--video",
        args.video,
        "--max-frames",
        str(args.max_frames),
        "--stagger",
        "0.1",
        "--hetero",
        "none",
        "--hetero-backbone",
        "diverse",
        "--prefer-peer-transfer",
        "--use-pretrained-backbone",
        "--similarity-threshold",
        "0.85",
        "--confidence-threshold",
        "0.8",
        "--heavy-detector",
        "opencv_hog",
        "--calibrate-epochs",
        "12",
        "--calibrate-lr",
        "1e-3",
    ]

    for seed in seeds:
        out1 = suite_dir / f"strict_no_calib_seed{seed}"
        out1.mkdir(parents=True, exist_ok=True)
        cmd1 = base + ["--seed", str(seed), "--calibrate-frames", "0", "--output-dir", str(out1)]
        _run(cmd1, env=env, cwd=str(cwd))

        out2 = suite_dir / f"strict_calib_seed{seed}"
        out2.mkdir(parents=True, exist_ok=True)
        cmd2 = base + ["--seed", str(seed), "--calibrate-frames", "15", "--output-dir", str(out2)]
        _run(cmd2, env=env, cwd=str(cwd))

    rows = []
    for run_dir in sorted([p for p in suite_dir.iterdir() if p.is_dir()]):
        m = _collect_metrics(run_dir)
        if m is None:
            continue
        name = run_dir.name
        m["run"] = name
        m["condition"] = "strict_calib" if "strict_calib" in name else "strict_no_calib"
        m["seed"] = int(name.split("seed")[-1])
        rows.append(m)

    df = pd.DataFrame(rows).sort_values(["condition", "seed"])
    art = suite_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    df.to_csv(art / "suite_run_metrics.csv", index=False)

    summary_rows = []
    for cond, g in df.groupby("condition"):
        summary_rows.append(
            {
                "condition": cond,
                "runs": len(g),
                "skip_rate_mean_pct": g["skip_rate_pct"].mean(),
                "skip_rate_std_pct": g["skip_rate_pct"].std(ddof=1) if len(g) > 1 else 0.0,
                "heavy_calls_mean": g["heavy_calls"].mean(),
                "heavy_calls_std": g["heavy_calls"].std(ddof=1) if len(g) > 1 else 0.0,
                "cross_reuse_rate_mean_pct": g["cross_reuse_rate_pct"].mean(),
                "cross_reuse_rate_std_pct": g["cross_reuse_rate_pct"].std(ddof=1) if len(g) > 1 else 0.0,
                "sim_mean_mean": g["sim_mean"].mean(),
                "sim_mean_std": g["sim_mean"].std(ddof=1) if len(g) > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("condition")
    summary.to_csv(art / "suite_condition_summary.csv", index=False)

    print(f"Suite complete: {suite_dir}")
    print(f"Wrote {art / 'suite_run_metrics.csv'}")
    print(f"Wrote {art / 'suite_condition_summary.csv'}")


if __name__ == "__main__":
    main()

