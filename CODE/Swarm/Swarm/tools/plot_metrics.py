"""Simple plotting utility to visualize metrics CSV produced by main.py.

Usage:
    python tools/plot_metrics.py metrics_agent_A_123456.csv

Produces two plots:
- skip/heavy decisions over time
- cumulative published bytes over time
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List


def plot_single(df: pd.DataFrame, out_prefix: str):
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # decision counts over frames
    df_dec = df.groupby(["frame", "decision"]).size().unstack(fill_value=0)
    plt.figure(figsize=(10, 4))
    if "skipped" in df_dec.columns:
        plt.plot(df_dec.index, df_dec["skipped"], label="skipped")
    if "heavy" in df_dec.columns:
        plt.plot(df_dec.index, df_dec["heavy"], label="heavy")
    plt.xlabel("frame")
    plt.ylabel("count")
    plt.legend()
    plt.title(f"Decisions per frame ({out_prefix})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_decisions.png")
    plt.close()

    # cumulative published bytes (if present)
    if "total_published_bytes" in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df["frame"], df["total_published_bytes"].cumsum(), label="cum_published_bytes")
        plt.xlabel("frame")
        plt.ylabel("bytes")
        plt.title(f"Cumulative published bytes ({out_prefix})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_cum_bytes.png")
        plt.close()


def plot_aggregate(dfs: List[pd.DataFrame], out_prefix: str):
    # concatenate
    agg = pd.concat(dfs, ignore_index=True)
    # simple summary plot: stacked decisions per frame aggregated across agents
    df_dec = agg.groupby(["frame", "decision"]).size().unstack(fill_value=0)
    plt.figure(figsize=(10, 4))
    if "skipped" in df_dec.columns:
        plt.plot(df_dec.index, df_dec["skipped"], label="skipped")
    if "heavy" in df_dec.columns:
        plt.plot(df_dec.index, df_dec["heavy"], label="heavy")
    plt.xlabel("frame")
    plt.ylabel("count")
    plt.legend()
    plt.title(f"Aggregate decisions per frame ({out_prefix})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_aggregate_decisions.png")
    plt.close()


def _pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def write_summary_tables(dfs: List[pd.DataFrame], out_prefix: str):
    agg = pd.concat(dfs, ignore_index=True)
    total = len(agg)
    heavy_after = int((agg["decision"] == "heavy").sum())
    skipped_after = int((agg["decision"] == "skipped").sum())

    # Baseline (before implementation): all decisions require heavy inference.
    baseline_heavy = total
    baseline_skipped = 0

    heavy_reduction = baseline_heavy - heavy_after
    heavy_reduction_pct = _pct(heavy_reduction, baseline_heavy)
    skip_rate_before = _pct(baseline_skipped, total)
    skip_rate_after = _pct(skipped_after, total)
    heavy_rate_before = _pct(baseline_heavy, total)
    heavy_rate_after = _pct(heavy_after, total)
    compute_speedup = (baseline_heavy / heavy_after) if heavy_after else float("inf")

    rows = [
        {
            "metric": "total_decisions",
            "before_baseline": baseline_heavy,
            "after_implementation": total,
            "advantage": "Context metric",
        },
        {
            "metric": "heavy_inference_calls",
            "before_baseline": baseline_heavy,
            "after_implementation": heavy_after,
            "advantage": f"{heavy_reduction} fewer heavy calls ({heavy_reduction_pct:.2f}% reduction)",
        },
        {
            "metric": "skipped_calls",
            "before_baseline": baseline_skipped,
            "after_implementation": skipped_after,
            "advantage": f"+{skipped_after} calls skipped",
        },
        {
            "metric": "skip_rate_percent",
            "before_baseline": f"{skip_rate_before:.2f}",
            "after_implementation": f"{skip_rate_after:.2f}",
            "advantage": f"+{(skip_rate_after - skip_rate_before):.2f} pp",
        },
        {
            "metric": "heavy_rate_percent",
            "before_baseline": f"{heavy_rate_before:.2f}",
            "after_implementation": f"{heavy_rate_after:.2f}",
            "advantage": f"-{(heavy_rate_before - heavy_rate_after):.2f} pp",
        },
        {
            "metric": "relative_heavy_compute_factor",
            "before_baseline": "1.00x",
            "after_implementation": f"{(heavy_after / baseline_heavy):.2f}x",
            "advantage": f"Approx. {compute_speedup:.2f}x fewer heavy inferences",
        },
    ]

    sim_stats = {"mean": "", "std": "", "min": "", "median": ""}
    if "sim" in agg.columns:
        skipped_sim = pd.to_numeric(agg.loc[agg["decision"] == "skipped", "sim"], errors="coerce").dropna()
        if not skipped_sim.empty:
            sim_stats = {
                "mean": f"{skipped_sim.mean():.6f}",
                "std": f"{skipped_sim.std():.6f}",
                "min": f"{skipped_sim.min():.6f}",
                "median": f"{skipped_sim.median():.6f}",
            }

    rows.extend([
        {
            "metric": "skipped_similarity_mean",
            "before_baseline": "N/A",
            "after_implementation": sim_stats["mean"],
            "advantage": "Higher is better confidence proxy",
        },
        {
            "metric": "skipped_similarity_std",
            "before_baseline": "N/A",
            "after_implementation": sim_stats["std"],
            "advantage": "Lower indicates more consistency",
        },
    ])

    if {"agent_id", "used_packet_agent", "decision"}.issubset(agg.columns):
        skipped_mask = agg["decision"] == "skipped"
        with_packet = skipped_mask & agg["used_packet_agent"].notna()
        cross_agent = with_packet & (agg["used_packet_agent"] != agg["agent_id"])
        cross_count = int(cross_agent.sum())
        with_packet_count = int(with_packet.sum())
        cross_rate = _pct(cross_count, with_packet_count)
        rows.append({
            "metric": "cross_agent_transfer_rate_percent",
            "before_baseline": "0.00",
            "after_implementation": f"{cross_rate:.2f}",
            "advantage": f"{cross_count}/{with_packet_count} skipped decisions reused peer packets",
        })

    summary_df = pd.DataFrame(rows)
    summary_csv = f"{out_prefix}_before_after_summary.csv"
    summary_md = f"{out_prefix}_before_after_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_markdown(summary_md, index=False)

    per_agent = (
        agg.groupby(["agent_id", "decision"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "skipped" not in per_agent.columns:
        per_agent["skipped"] = 0
    if "heavy" not in per_agent.columns:
        per_agent["heavy"] = 0
    per_agent["total"] = per_agent["heavy"] + per_agent["skipped"]
    per_agent["skip_rate_percent"] = per_agent.apply(
        lambda r: _pct(r["skipped"], r["total"]), axis=1
    )
    per_agent_csv = f"{out_prefix}_per_agent_summary.csv"
    per_agent.to_csv(per_agent_csv, index=False)

    print(f"Saved summary table: {summary_csv}")
    print(f"Saved summary markdown: {summary_md}")
    print(f"Saved per-agent summary: {per_agent_csv}")


def main(paths: List[str]):
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            continue
        df = pd.read_csv(p)
        dfs.append(df)
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            plot_single(df, base)
            print(f"Saved plots for {p} as {base}_*.png")
        except Exception as e:
            print(f"Failed to plot {p}: {e}")

    if dfs:
        try:
            plot_aggregate(dfs, "swarm_aggregate")
            print("Saved aggregate plots as swarm_aggregate_*.png")
        except Exception as e:
            print(f"Failed to create aggregate plots: {e}")
        try:
            write_summary_tables(dfs, "swarm_aggregate")
        except Exception as e:
            print(f"Failed to create summary tables: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_metrics.py <metrics1.csv> [metrics2.csv ...]")
        sys.exit(1)
    main(sys.argv[1:])
