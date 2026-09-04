"""Run teacher-facing training/adaptation demo scenarios.

This wrapper keeps the project narrative honest: it demonstrates
cross-backbone projection-head calibration and inference-time knowledge
transfer, not full offline CNN training.

Examples:
    python tools/run_training_demo.py --scenario synthetic
    python tools/run_training_demo.py --scenario compare-video
    python tools/run_training_demo.py --scenario suite
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "paper_results_hetero_1770271738" / "artifacts" / "run_three_agents_local.py"
SUITE_RUNNER = ROOT / "tools" / "run_true_experiment_suite.py"
DASHBOARD_RUNNER = ROOT / "tools" / "build_demo_dashboard.py"
RETRAIN_RUNNER = ROOT / "tools" / "retrain_from_packets.py"
DEFAULT_VIDEO = ROOT / "data" / "video_for_swarm.avi"
DEFAULT_OUTPUT_ROOT = ROOT / "demo_runs"


def _run(cmd: List[str], dry_run: bool) -> None:
    print("\nRUN:", " ".join(cmd))
    if not dry_run:
        sys.stdout.flush()
        env = os.environ.copy()
        env.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def _reset_output_dir(output_dir: Path, output_root: Path, dry_run: bool) -> None:
    resolved_dir = output_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_dir == resolved_root or resolved_root not in resolved_dir.parents:
        raise ValueError(f"Refusing to reset output outside demo root: {resolved_dir}")
    if dry_run:
        return
    if resolved_dir.exists():
        shutil.rmtree(resolved_dir)


def _base_experiment_cmd(
    output_dir: Path,
    mode: str,
    video: Path,
    max_frames: int,
    detector: str,
    calibrate_frames: int,
    calibrate_epochs: int,
    use_pretrained: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--mode",
        mode,
        "--max-frames",
        str(max_frames),
        "--num-agents",
        "3",
        "--hetero-backbone",
        "diverse",
        "--prefer-peer-transfer",
        "--heavy-detector",
        detector,
        "--output-dir",
        str(output_dir),
        "--calibrate-frames",
        str(calibrate_frames),
        "--calibrate-epochs",
        str(calibrate_epochs),
    ]
    if use_pretrained:
        cmd.append("--use-pretrained-backbone")
    if mode == "video":
        cmd.extend(["--video", str(video)])
    return cmd


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _summarize_run(output_dir: Path) -> Dict[str, object]:
    metric_rows: List[Dict[str, str]] = []
    metrics_dir = output_dir / "metrics"
    for path in sorted(metrics_dir.glob("metrics_agent_*.csv")):
        metric_rows.extend(_read_csv(path))

    total = len(metric_rows)
    heavy = sum(1 for row in metric_rows if row.get("decision") == "heavy")
    skipped = sum(1 for row in metric_rows if row.get("decision") == "skipped")
    with_packet = [
        row
        for row in metric_rows
        if row.get("decision") == "skipped" and row.get("used_packet_agent")
    ]
    cross = [
        row
        for row in with_packet
        if row.get("used_packet_agent") and row.get("used_packet_agent") != row.get("agent_id")
    ]

    calibration_rows = _read_csv(output_dir / "artifacts" / "calibration_log.csv")
    calibration_ok = sum(1 for row in calibration_rows if row.get("status") == "ok")

    return {
        "output_dir": str(output_dir),
        "total_decisions": total,
        "heavy_calls": heavy,
        "skipped_calls": skipped,
        "skip_rate_pct": round((skipped / total) * 100, 3) if total else 0.0,
        "cross_agent_reuse_count": len(cross),
        "cross_agent_reuse_rate_pct": round((len(cross) / len(with_packet)) * 100, 3)
        if with_packet
        else 0.0,
        "calibration_rows": len(calibration_rows),
        "calibration_ok_rows": calibration_ok,
    }


def _print_summary(summary: Dict[str, object]) -> None:
    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def _print_comparison(summaries: Iterable[Dict[str, object]]) -> None:
    rows = list(summaries)
    print("\nCOMPARISON")
    print(
        "  {name:<18} {skip:>10} {cross:>14} {heavy:>12} {calib:>12}".format(
            name="run",
            skip="skip_pct",
            cross="cross_pct",
            heavy="heavy_calls",
            calib="calib_ok",
        )
    )
    for row in rows:
        name = Path(str(row["output_dir"])).name
        print(
            "  {name:<18} {skip:>10} {cross:>14} {heavy:>12} {calib:>12}".format(
                name=name,
                skip=row["skip_rate_pct"],
                cross=row["cross_agent_reuse_rate_pct"],
                heavy=row["heavy_calls"],
                calib=row["calibration_ok_rows"],
            )
        )


def _write_demo_summary(output_root: Path, summaries: List[Dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / "demo_summary.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\nWrote {out_path}")


def _build_dashboard(output_root: Path) -> None:
    if not DASHBOARD_RUNNER.exists():
        return
    cmd = [
        sys.executable,
        str(DASHBOARD_RUNNER),
        "--demo-root",
        str(output_root),
        "--output",
        str(output_root / "index.html"),
    ]
    _run(cmd, dry_run=False)


def _run_retraining(output_dir: Path, dry_run: bool) -> None:
    if not RETRAIN_RUNNER.exists():
        return
    cmd = [
        sys.executable,
        str(RETRAIN_RUNNER),
        "--results-dir",
        str(output_dir),
        "--epochs",
        "35",
        "--lr",
        "1e-2",
        "--use-pretrained-backbone",
    ]
    _run(cmd, dry_run=dry_run)


def run_scenario(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    video = Path(args.video).resolve()

    if args.scenario == "synthetic":
        out = output_root / "synthetic_calib"
        cmd = _base_experiment_cmd(
            out,
            mode="synthetic",
            video=video,
            max_frames=args.max_frames or 50,
            detector="simulated",
            calibrate_frames=10,
            calibrate_epochs=8,
            use_pretrained=False,
        )
        _reset_output_dir(out, output_root, args.dry_run)
        _run(cmd, args.dry_run)
        if not args.dry_run:
            summary = _summarize_run(out)
            _print_summary(summary)
            _write_demo_summary(output_root, [summary])
            _build_dashboard(output_root)
        return

    if args.scenario in {"video-calib", "video-no-calib"}:
        calib = args.scenario == "video-calib"
        out = output_root / ("video_calib" if calib else "video_no_calib")
        cmd = _base_experiment_cmd(
            out,
            mode="video",
            video=video,
            max_frames=args.max_frames or 80,
            detector="opencv_hog",
            calibrate_frames=15 if calib else 0,
            calibrate_epochs=12,
            use_pretrained=True,
        )
        _reset_output_dir(out, output_root, args.dry_run)
        _run(cmd, args.dry_run)
        if calib:
            _run_retraining(out, args.dry_run)
        if not args.dry_run:
            summary = _summarize_run(out)
            _print_summary(summary)
            _write_demo_summary(output_root, [summary])
            _build_dashboard(output_root)
        return

    if args.scenario == "compare-video":
        summaries: List[Dict[str, object]] = []
        for scenario in ("video-no-calib", "video-calib"):
            nested = argparse.Namespace(**vars(args))
            nested.scenario = scenario
            run_scenario(nested)
            if not args.dry_run:
                out = output_root / ("video_no_calib" if scenario == "video-no-calib" else "video_calib")
                summaries.append(_summarize_run(out))
        if not args.dry_run:
            _print_comparison(summaries)
            _write_demo_summary(output_root, summaries)
            _build_dashboard(output_root)
        return

    if args.scenario == "suite":
        cmd = [
            sys.executable,
            str(SUITE_RUNNER),
            "--output-root",
            str(ROOT),
            "--max-frames",
            str(args.max_frames or 80),
            "--video",
            str(video),
            "--seeds",
            args.seeds,
        ]
        _run(cmd, args.dry_run)
        return

    raise ValueError(f"Unsupported scenario: {args.scenario}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training/adaptation demo scenarios")
    parser.add_argument(
        "--scenario",
        choices=["synthetic", "video-calib", "video-no-calib", "compare-video", "suite"],
        default="compare-video",
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    if not RUNNER.exists():
        raise FileNotFoundError(f"Experiment runner not found: {RUNNER}")
    if args.scenario != "synthetic" and not Path(args.video).exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.scenario == "suite" and not SUITE_RUNNER.exists():
        raise FileNotFoundError(f"Suite runner not found: {SUITE_RUNNER}")

    run_scenario(args)


if __name__ == "__main__":
    main()
