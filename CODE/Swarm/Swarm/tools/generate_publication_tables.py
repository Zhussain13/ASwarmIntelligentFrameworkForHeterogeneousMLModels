"""Generate publication-ready result tables from Swarm experiment CSVs.

Usage:
    python tools/generate_publication_tables.py --results-dir paper_results_hetero_1770271738

Outputs:
    <results-dir>/artifacts/publication_tables.md
    <results-dir>/artifacts/publication_tables.tex
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


@dataclass
class RunStats:
    agents: int
    frames: int
    total_decisions: int
    heavy_calls: int
    skipped_calls: int
    heavy_rate_pct: float
    skip_rate_pct: float
    heavy_reduction_pct: float
    heavy_speedup_x: float
    sim_mean: float
    sim_std: float
    sim_min: float
    sim_median: float
    cross_agent_rate_pct: float
    cross_agent_count: int
    skipped_with_packet_count: int


def _load_metrics(metrics_dir: Path) -> List[pd.DataFrame]:
    files = sorted(metrics_dir.glob("metrics_agent_*.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics_agent_*.csv files found in {metrics_dir}")
    return [pd.read_csv(p) for p in files]


def _safe_pct(num: float, den: float) -> float:
    return 0.0 if den == 0 else (num / den) * 100.0


def _run_stats(agg: pd.DataFrame) -> RunStats:
    total = int(len(agg))
    heavy = int((agg["decision"] == "heavy").sum())
    skipped = int((agg["decision"] == "skipped").sum())
    frames = int(pd.to_numeric(agg["frame"], errors="coerce").nunique())
    agents = int(agg["agent_id"].nunique())

    # Baseline definition for this table:
    # heavy-only baseline = one heavy inference per decision.
    baseline_heavy = total
    heavy_reduction_pct = _safe_pct(baseline_heavy - heavy, baseline_heavy)
    heavy_speedup_x = (baseline_heavy / heavy) if heavy else float("inf")

    skipped_sim = pd.to_numeric(
        agg.loc[agg["decision"] == "skipped", "sim"], errors="coerce"
    ).dropna()
    sim_mean = float(skipped_sim.mean()) if not skipped_sim.empty else 0.0
    sim_std = float(skipped_sim.std()) if not skipped_sim.empty else 0.0
    sim_min = float(skipped_sim.min()) if not skipped_sim.empty else 0.0
    sim_median = float(skipped_sim.median()) if not skipped_sim.empty else 0.0

    cross_agent_count = 0
    skipped_with_packet_count = 0
    cross_agent_rate_pct = 0.0
    required_cols = {"agent_id", "used_packet_agent", "decision"}
    if required_cols.issubset(set(agg.columns)):
        with_packet_mask = (agg["decision"] == "skipped") & agg["used_packet_agent"].notna()
        cross_mask = with_packet_mask & (agg["used_packet_agent"] != agg["agent_id"])
        skipped_with_packet_count = int(with_packet_mask.sum())
        cross_agent_count = int(cross_mask.sum())
        cross_agent_rate_pct = _safe_pct(cross_agent_count, skipped_with_packet_count)

    return RunStats(
        agents=agents,
        frames=frames,
        total_decisions=total,
        heavy_calls=heavy,
        skipped_calls=skipped,
        heavy_rate_pct=_safe_pct(heavy, total),
        skip_rate_pct=_safe_pct(skipped, total),
        heavy_reduction_pct=heavy_reduction_pct,
        heavy_speedup_x=heavy_speedup_x,
        sim_mean=sim_mean,
        sim_std=sim_std,
        sim_min=sim_min,
        sim_median=sim_median,
        cross_agent_rate_pct=cross_agent_rate_pct,
        cross_agent_count=cross_agent_count,
        skipped_with_packet_count=skipped_with_packet_count,
    )


def _per_agent_table(agg: pd.DataFrame) -> pd.DataFrame:
    t = agg.groupby(["agent_id", "decision"]).size().unstack(fill_value=0).reset_index()
    if "heavy" not in t.columns:
        t["heavy"] = 0
    if "skipped" not in t.columns:
        t["skipped"] = 0
    t["total"] = t["heavy"] + t["skipped"]
    t["skip_rate_percent"] = t.apply(lambda r: _safe_pct(r["skipped"], r["total"]), axis=1)
    t["heavy_rate_percent"] = t.apply(lambda r: _safe_pct(r["heavy"], r["total"]), axis=1)
    return t[["agent_id", "total", "heavy", "skipped", "skip_rate_percent", "heavy_rate_percent"]]


def _transfer_table(agg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Receiver x source matrix from per-frame records.
    if not {"agent_id", "used_packet_agent", "decision"}.issubset(set(agg.columns)):
        empty = pd.DataFrame()
        return empty, empty

    s = agg[(agg["decision"] == "skipped") & agg["used_packet_agent"].notna()].copy()
    if s.empty:
        empty = pd.DataFrame()
        return empty, empty

    matrix = (
        s.groupby(["agent_id", "used_packet_agent"]).size().unstack(fill_value=0).reset_index()
    )

    self_counts = (s["agent_id"] == s["used_packet_agent"]).sum()
    peer_counts = (s["agent_id"] != s["used_packet_agent"]).sum()
    breakdown = pd.DataFrame(
        [
            {"transfer_type": "self-packet reuse", "count": int(self_counts)},
            {"transfer_type": "cross-agent reuse", "count": int(peer_counts)},
            {
                "transfer_type": "cross-agent reuse percent",
                "count": f"{_safe_pct(peer_counts, len(s)):.2f}",
            },
        ]
    )
    return matrix, breakdown


def _adaptation_table(metrics_dir: Path) -> pd.DataFrame:
    import csv

    rows = []
    for p in sorted(metrics_dir.glob("adaptation_log_agent_*.csv")):
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            for parts in reader:
                if len(parts) < 3:
                    continue
                try:
                    ts = float(parts[0])
                    loss = float(parts[1])
                    samples = int(float(parts[2]))
                except ValueError:
                    continue
                backbone = parts[3].strip() if len(parts) >= 4 else ""
                rows.append(
                    {
                        "file": p.name,
                        "timestamp": ts,
                        "loss": loss,
                        "samples_used": samples,
                        "backbone": backbone,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["file", "timestamp", "loss", "samples_used", "backbone"])
    return pd.DataFrame(rows)


def _format_markdown(
    results_dir: Path,
    stats: RunStats,
    per_agent: pd.DataFrame,
    transfer_matrix: pd.DataFrame,
    transfer_breakdown: pd.DataFrame,
    adaptation: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# Publication-Ready Tables")
    lines.append("")
    lines.append(
        f"Source: `{results_dir.name}/metrics/` | Baseline assumption: heavy-only inference for all decisions."
    )
    lines.append("")
    lines.append("## Table 1. Main Results (Baseline vs Proposed)")
    main_table = pd.DataFrame(
        [
            {
                "metric": "Total decisions",
                "baseline_heavy_only": stats.total_decisions,
                "proposed_swarm_kt": stats.total_decisions,
                "gain": "Context metric",
            },
            {
                "metric": "Heavy inference calls",
                "baseline_heavy_only": stats.total_decisions,
                "proposed_swarm_kt": stats.heavy_calls,
                "gain": f"{stats.heavy_reduction_pct:.2f}% reduction",
            },
            {
                "metric": "Skipped decisions",
                "baseline_heavy_only": 0,
                "proposed_swarm_kt": stats.skipped_calls,
                "gain": f"+{stats.skipped_calls}",
            },
            {
                "metric": "Skip rate (%)",
                "baseline_heavy_only": "0.00",
                "proposed_swarm_kt": f"{stats.skip_rate_pct:.2f}",
                "gain": f"+{stats.skip_rate_pct:.2f} pp",
            },
            {
                "metric": "Heavy compute factor",
                "baseline_heavy_only": "1.00x",
                "proposed_swarm_kt": f"{stats.heavy_calls / stats.total_decisions:.2f}x",
                "gain": f"{stats.heavy_speedup_x:.2f}x fewer heavy calls",
            },
        ]
    )
    lines.append(main_table.to_markdown(index=False))
    lines.append("")
    lines.append("## Table 2. Per-Agent Breakdown")
    lines.append(per_agent.to_markdown(index=False, floatfmt=".2f"))
    lines.append("")
    lines.append("## Table 3. Similarity Quality (Skipped Decisions)")
    sim_tbl = pd.DataFrame(
        [
            {"stat": "mean", "value": f"{stats.sim_mean:.6f}"},
            {"stat": "std", "value": f"{stats.sim_std:.6f}"},
            {"stat": "min", "value": f"{stats.sim_min:.6f}"},
            {"stat": "median", "value": f"{stats.sim_median:.6f}"},
        ]
    )
    lines.append(sim_tbl.to_markdown(index=False))
    lines.append("")
    lines.append("## Table 4. Knowledge Transfer Usage")
    if transfer_matrix.empty:
        lines.append("No transfer matrix available in this run.")
    else:
        lines.append("Receiver agent (rows) vs packet source agent (columns).")
        lines.append("")
        lines.append(transfer_matrix.to_markdown(index=False))
        lines.append("")
        lines.append(transfer_breakdown.to_markdown(index=False))
    lines.append("")
    lines.append("## Table 5. Adaptation Log Summary")
    if adaptation.empty:
        lines.append("No adaptation log records found.")
    else:
        lines.append(adaptation.to_markdown(index=False, floatfmt=".8f"))
    lines.append("")
    lines.append("## Caption Notes (Use in Paper)")
    lines.append(
        f"- Pilot run configuration: {stats.frames} frames, {stats.agents} agents, {stats.total_decisions} agent-decisions."
    )
    lines.append(
        "- Baseline is analytically derived from the same run (heavy-only path), not from a separate hardware-timed baseline run."
    )
    lines.append(
        f"- Cross-agent reuse in this run: {stats.cross_agent_count}/{stats.skipped_with_packet_count} ({stats.cross_agent_rate_pct:.2f}%)."
    )
    return "\n".join(lines) + "\n"


def _format_latex(
    stats: RunStats, per_agent: pd.DataFrame, sim_tbl: pd.DataFrame, transfer_breakdown: pd.DataFrame
) -> str:
    main_tbl = pd.DataFrame(
        [
            ["Total decisions", stats.total_decisions, stats.total_decisions, "Context metric"],
            ["Heavy inference calls", stats.total_decisions, stats.heavy_calls, f"{stats.heavy_reduction_pct:.2f}\\% reduction"],
            ["Skipped decisions", 0, stats.skipped_calls, f"+{stats.skipped_calls}"],
            ["Skip rate (\\%)", "0.00", f"{stats.skip_rate_pct:.2f}", f"+{stats.skip_rate_pct:.2f} pp"],
            ["Heavy compute factor", "1.00x", f"{stats.heavy_calls / stats.total_decisions:.2f}x", f"{stats.heavy_speedup_x:.2f}x fewer heavy calls"],
        ],
        columns=["Metric", "Baseline", "Proposed", "Gain"],
    )
    blocks = []
    blocks.append("% Auto-generated by tools/generate_publication_tables.py")
    blocks.append("% Table A: Main results")
    blocks.append(main_tbl.to_latex(index=False, escape=False))
    blocks.append("")
    blocks.append("% Table B: Per-agent breakdown")
    blocks.append(per_agent.to_latex(index=False, float_format="%.2f"))
    blocks.append("")
    blocks.append("% Table C: Similarity quality")
    blocks.append(sim_tbl.to_latex(index=False, escape=False))
    blocks.append("")
    blocks.append("% Table D: Transfer usage breakdown")
    blocks.append(transfer_breakdown.to_latex(index=False, escape=False))
    return "\n".join(blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="paper_results_hetero_1770271738",
        help="Path to a results folder containing metrics/*.csv files.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    artifacts_dir = results_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dfs = _load_metrics(metrics_dir)
    agg = pd.concat(dfs, ignore_index=True)
    stats = _run_stats(agg)
    per_agent = _per_agent_table(agg)
    transfer_matrix, transfer_breakdown = _transfer_table(agg)
    adaptation = _adaptation_table(metrics_dir)
    sim_tbl = pd.DataFrame(
        [
            {"Stat": "Mean", "Value": f"{stats.sim_mean:.6f}"},
            {"Stat": "Std", "Value": f"{stats.sim_std:.6f}"},
            {"Stat": "Min", "Value": f"{stats.sim_min:.6f}"},
            {"Stat": "Median", "Value": f"{stats.sim_median:.6f}"},
        ]
    )

    md_out = artifacts_dir / "publication_tables.md"
    tex_out = artifacts_dir / "publication_tables.tex"

    md_text = _format_markdown(
        results_dir=results_dir,
        stats=stats,
        per_agent=per_agent,
        transfer_matrix=transfer_matrix,
        transfer_breakdown=transfer_breakdown,
        adaptation=adaptation,
    )
    tex_text = _format_latex(stats, per_agent, sim_tbl, transfer_breakdown)

    md_out.write_text(md_text, encoding="utf-8")
    tex_out.write_text(tex_text, encoding="utf-8")

    print(f"Saved: {md_out}")
    print(f"Saved: {tex_out}")


if __name__ == "__main__":
    main()
