"""Build a polished local HTML dashboard for the training/adaptation demo.

Usage:
    python tools/build_demo_dashboard.py
    python tools/build_demo_dashboard.py --demo-root demo_runs --output demo_runs/index.html
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_ROOT = ROOT / "demo_runs"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _pct(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else (numerator / denominator) * 100.0


def _load_metric_rows(run_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted((run_dir / "metrics").glob("metrics_agent_*.csv")):
        rows.extend(_read_csv(path))
    return rows


def _run_summary(run_dir: Path) -> Dict[str, object]:
    rows = _load_metric_rows(run_dir)
    manifest = _read_json(run_dir / "artifacts" / "run_manifest.json") or {}
    calibration_rows = _read_csv(run_dir / "artifacts" / "calibration_log.csv")
    with_packet = [
        row
        for row in rows
        if row.get("decision") == "skipped" and row.get("used_packet_agent")
    ]
    cross = [
        row
        for row in with_packet
        if row.get("used_packet_agent") and row.get("used_packet_agent") != row.get("agent_id")
    ]
    heavy = sum(1 for row in rows if row.get("decision") == "heavy")
    skipped = sum(1 for row in rows if row.get("decision") == "skipped")
    total = len(rows)
    sims = [_safe_float(row.get("sim")) for row in rows if row.get("decision") == "skipped" and row.get("sim")]
    return {
        "name": run_dir.name,
        "path": run_dir,
        "manifest": manifest,
        "metrics": rows,
        "calibration": calibration_rows,
        "total_decisions": total,
        "heavy_calls": heavy,
        "skipped_calls": skipped,
        "skip_rate_pct": _pct(skipped, total),
        "heavy_reduction_pct": _pct(total - heavy, total),
        "cross_agent_reuse_count": len(cross),
        "cross_agent_reuse_rate_pct": _pct(len(cross), len(with_packet)),
        "avg_similarity": sum(sims) / len(sims) if sims else 0.0,
        "calibration_ok_rows": sum(1 for row in calibration_rows if row.get("status") == "ok"),
    }


def _discover_runs(demo_root: Path) -> List[Dict[str, object]]:
    runs = []
    for run_dir in sorted(path for path in demo_root.iterdir() if path.is_dir()):
        if (run_dir / "metrics").exists():
            summary = _run_summary(run_dir)
            if summary["total_decisions"]:
                runs.append(summary)
    preferred = {"video_no_calib": 0, "video_calib": 1, "synthetic_calib": 2}
    return sorted(runs, key=lambda row: preferred.get(str(row["name"]), 99))


def _bar(value: float, maximum: float = 100.0, label: str = "") -> str:
    width = max(0.0, min(100.0, _pct(value, maximum)))
    safe_label = html.escape(label or f"{value:.1f}%")
    return (
        '<div class="bar-shell">'
        f'<div class="bar-fill" style="width:{width:.2f}%"></div>'
        f'<span>{safe_label}</span>'
        "</div>"
    )


def _metric_card(title: str, value: str, subtitle: str, accent: str = "") -> str:
    return (
        f'<article class="metric-card {accent}">'
        f"<p>{html.escape(title)}</p>"
        f"<strong>{html.escape(value)}</strong>"
        f"<span>{html.escape(subtitle)}</span>"
        "</article>"
    )


def _comparison_cards(runs: Sequence[Dict[str, object]]) -> str:
    if not runs:
        return '<p class="muted">No demo runs found. Run <code>python tools\\run_training_demo.py --scenario compare-video</code>.</p>'
    packet_count = 0
    candidate_count = 0
    retrained_heads = 0
    best_after_accuracy = 0.0
    for run in runs:
        artifacts_dir = Path(run["path"]) / "artifacts"
        for path in artifacts_dir.glob("knowledge_packets_*.jsonl"):
            packet_count += len(_read_jsonl(path))
        candidate_count += len(_candidate_rows(run))
        retraining = _read_json(artifacts_dir / "retraining_metrics.json")
        if isinstance(retraining, list):
            for row in retraining:
                if row.get("status") == "ok":
                    retrained_heads += 1
                    best_after_accuracy = max(best_after_accuracy, _safe_float(row.get("after_accuracy")))
    best_agreement = max(
        (_safe_int(candidate.get("agreement_count")) for run in runs for candidate in _candidate_rows(run)),
        default=0,
    )
    return "".join(
        [
            _metric_card("Packets Generated", str(packet_count), "Rich detection packets shared", "violet"),
            _metric_card("Validated Candidates", str(candidate_count), "Packets prepared for retraining", "green"),
            _metric_card("Retrained Heads", str(retrained_heads), "Student heads updated live", "blue"),
            _metric_card("Best After Accuracy", f"{best_after_accuracy:.1%}", f"Agreement count up to {best_agreement}", "amber"),
        ]
    )


def _run_table(runs: Sequence[Dict[str, object]]) -> str:
    rows = []
    for run in runs:
        manifest = run["manifest"]
        skip_rate = _safe_float(run["skip_rate_pct"])
        cross_rate = _safe_float(run["cross_agent_reuse_rate_pct"])
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(str(run['name']))}</strong></td>"
            f"<td>{html.escape(str(manifest.get('mode', '')))}</td>"
            f"<td>{html.escape(str(manifest.get('max_frames', '')))}</td>"
            f"<td>{html.escape(', '.join(manifest.get('requested_backbones', [])))}</td>"
            f"<td>{_bar(skip_rate, label=f'{skip_rate:.1f}%')}</td>"
            f"<td>{_bar(cross_rate, label=f'{cross_rate:.1f}%')}</td>"
            f"<td>{html.escape(str(run['heavy_calls']))}</td>"
            f"<td>{html.escape(str(run['calibration_ok_rows']))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Run</th><th>Mode</th><th>Frames</th><th>Backbones</th>"
        "<th>Skip Rate</th><th>Peer Reuse</th><th>Heavy Calls</th><th>Trained Heads</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _calibration_section(runs: Sequence[Dict[str, object]]) -> str:
    cards = []
    for run in runs:
        if not run["calibration"]:
            continue
        rows = []
        for item in run["calibration"]:
            loss = _safe_float(item.get("loss"))
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('agent_id', '')))}</td>"
                f"<td>{html.escape(str(item.get('backbone', '')))}</td>"
                f"<td>{html.escape(str(item.get('samples', '')))}</td>"
                f"<td>{html.escape(str(item.get('epochs', '')))}</td>"
                f"<td>{loss:.6f}</td>"
                "</tr>"
            )
        cards.append(
            '<article class="panel">'
            f"<h3>{html.escape(str(run['name']))}</h3>"
            '<div class="table-wrap compact"><table><thead><tr>'
            "<th>Agent</th><th>Backbone</th><th>Samples</th><th>Epochs</th><th>Final Loss</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div></article>"
        )
    if not cards:
        return '<p class="muted">No calibration logs found yet.</p>'
    return "".join(cards)


def _retraining_section(runs: Sequence[Dict[str, object]]) -> str:
    panels = []
    for run in runs:
        metrics = _read_json(Path(run["path"]) / "artifacts" / "retraining_metrics.json")
        if not isinstance(metrics, list) or not metrics:
            continue
        rows = []
        for item in metrics:
            status = str(item.get("status", ""))
            if status != "ok":
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(item.get('backbone', 'all')))}</td>"
                    f"<td colspan='7'>{html.escape(status)} ({html.escape(str(item.get('samples', 0)))} samples)</td>"
                    "</tr>"
                )
                continue
            before_loss = _safe_float(item.get("before_loss"))
            after_loss = _safe_float(item.get("after_loss"))
            before_acc = _safe_float(item.get("before_accuracy")) * 100.0
            after_acc = _safe_float(item.get("after_accuracy")) * 100.0
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('backbone', '')))}</td>"
                f"<td>{html.escape(str(item.get('samples', '')))}</td>"
                f"<td>{html.escape(str(item.get('positive_samples', '')))}</td>"
                f"<td>{html.escape(str(item.get('negative_samples', '')))}</td>"
                f"<td>{before_loss:.4f}</td>"
                f"<td>{after_loss:.4f}</td>"
                f"<td>{before_acc:.1f}%</td>"
                f"<td>{after_acc:.1f}%</td>"
                "</tr>"
            )
        panels.append(
            '<article class="panel wide">'
            f"<h3>{html.escape(str(run['name']))}: Live Student-Head Retraining</h3>"
            '<p class="muted">Frozen backbones stay fixed; a lightweight person/background head is retrained from validated packet crops.</p>'
            '<div class="table-wrap"><table><thead><tr>'
            "<th>Backbone</th><th>Samples</th><th>Person</th><th>Background</th>"
            "<th>Loss Before</th><th>Loss After</th><th>Acc Before</th><th>Acc After</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div></article>"
        )
    if not panels:
        return '<article class="panel wide"><p class="muted">No retraining metrics yet. Run the calibrated video demo.</p></article>'
    return "".join(panels)


def _transfer_table(path: Path) -> str:
    rows = _read_csv(path)
    if not rows:
        return '<p class="muted">No transfer matrix found.</p>'
    headers = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers) + "</tr>"
        )
    return f'<div class="table-wrap compact"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def _transfer_section(runs: Sequence[Dict[str, object]]) -> str:
    panels = []
    for run in runs:
        metrics_dir = Path(run["path"]) / "metrics"
        panels.append(
            '<article class="panel">'
            f"<h3>{html.escape(str(run['name']))}: Agent Reuse</h3>"
            + _transfer_table(metrics_dir / "cross_agent_transfer.csv")
            + f"<h3>{html.escape(str(run['name']))}: Backbone Reuse</h3>"
            + _transfer_table(metrics_dir / "cross_backbone_transfer.csv")
            + "</article>"
        )
    return "".join(panels)


def _candidate_rows(run: Dict[str, object]) -> List[Dict[str, object]]:
    candidates = _read_json(Path(run["path"]) / "artifacts" / "validated_training_candidates.json")
    if isinstance(candidates, list):
        return candidates
    packet_rows: List[Dict[str, object]] = []
    for path in sorted((Path(run["path"]) / "artifacts").glob("knowledge_packets_*.jsonl")):
        packet_rows.extend(_read_jsonl(path))
    return packet_rows


def _prediction_rows(run: Dict[str, object]) -> List[Dict[str, object]]:
    predictions = _read_json(Path(run["path"]) / "artifacts" / "retraining_predictions.json")
    return predictions if isinstance(predictions, list) else []


def _bbox_overlay(candidate: Dict[str, object]) -> str:
    image = candidate.get("image")
    coordinates = candidate.get("coordinates")
    if not isinstance(image, dict) or not image.get("b64"):
        return '<div class="packet-image empty">No image payload</div>'
    width = _safe_float(image.get("width"), 1.0)
    height = _safe_float(image.get("height"), 1.0)
    box = ""
    if isinstance(coordinates, dict):
        left = _safe_float(coordinates.get("x")) / width * 100.0
        top = _safe_float(coordinates.get("y")) / height * 100.0
        box_width = _safe_float(coordinates.get("w")) / width * 100.0
        box_height = _safe_float(coordinates.get("h")) / height * 100.0
        box = (
            f'<span class="bbox" style="left:{left:.2f}%;top:{top:.2f}%;'
            f'width:{box_width:.2f}%;height:{box_height:.2f}%"></span>'
        )
    src = f"data:image/jpeg;base64,{html.escape(str(image['b64']))}"
    return f'<div class="packet-image"><img src="{src}" alt="packet frame">{box}</div>'


def _packet_gallery(runs: Sequence[Dict[str, object]]) -> str:
    cards = []
    for run in runs:
        if "no_calib" in str(run["name"]):
            continue
        candidates = _candidate_rows(run)[:6]
        if not candidates:
            continue
        card_items = []
        for candidate in candidates:
            confidence = _safe_float(candidate.get("best_confidence", candidate.get("confidence")))
            source_model = str(candidate.get("source_model") or candidate.get("model_name") or "")
            best_model = str(candidate.get("best_model") or source_model)
            agreement = str(candidate.get("agreement_count") or 1)
            detection_id = str(candidate.get("detection_id") or "")
            card_items.append(
                '<article class="packet-card">'
                + _bbox_overlay(candidate)
                + '<div class="packet-meta">'
                + f"<strong>{html.escape(str(candidate.get('label') or candidate.get('object_label') or 'Detection'))}</strong>"
                + f"<span>ID: {html.escape(detection_id[-28:] or 'n/a')}</span>"
                + f"<span>Source: {html.escape(source_model)}</span>"
                + f"<span>Best: {html.escape(best_model)} @ {confidence:.1%}</span>"
                + f"<span>Agreement: {html.escape(agreement)} agent(s)</span>"
                + "</div></article>"
            )
        cards.append(
            '<article class="panel wide">'
            f"<h3>{html.escape(str(run['name']))}: Shared Detection Packets</h3>"
            '<p class="muted">These are full-frame packet payloads with the detected person coordinates overlaid. Validated packets become retraining candidates.</p>'
            f'<div class="packet-grid">{"".join(card_items)}</div>'
            "</article>"
        )
    if not cards:
        return '<article class="panel wide"><p class="muted">No rich packet payloads found yet. Run the live video demo once.</p></article>'
    return "".join(cards)


def _focused_transfer_showcase(runs: Sequence[Dict[str, object]]) -> str:
    for run in runs:
        if "no_calib" in str(run["name"]):
            continue
        candidates = [
            candidate
            for candidate in _candidate_rows(run)
            if str(candidate.get("label") or candidate.get("object_label") or "").lower() == "person"
        ]
        predictions = _prediction_rows(run)
        if not candidates or not predictions:
            continue
        candidate = candidates[0]
        detection_id = candidate.get("detection_id")
        prediction_rows = [row for row in predictions if row.get("detection_id") == detection_id]
        prediction_rows = sorted(prediction_rows, key=lambda row: str(row.get("backbone", "")))
        rows = []
        for row in prediction_rows:
            before = _safe_float(row.get("before_student_confidence"))
            after = _safe_float(row.get("after_student_confidence"))
            delta = after - before
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('backbone', '')))}</td>"
                f"<td>{before:.1%}</td>"
                f"<td>{after:.1%}</td>"
                f"<td class='delta'>{delta:+.1%}</td>"
                "</tr>"
            )
        source_conf = _safe_float(candidate.get("source_confidence", candidate.get("confidence")))
        best_conf = _safe_float(candidate.get("best_confidence", candidate.get("confidence")))
        return "".join(
            [
                '<article class="panel wide showcase">',
                '<div class="showcase-grid">',
                "<div>",
                "<h3>1. Packet Generated By Main Model</h3>",
                _bbox_overlay(candidate),
                "</div>",
                '<div class="flow-card">',
                "<h3>2. Packet Transfer + Peer Validation</h3>",
                f"<p><strong>Detection ID:</strong> {html.escape(str(detection_id))}</p>",
                f"<p><strong>Label:</strong> {html.escape(str(candidate.get('label') or candidate.get('object_label')))}</p>",
                f"<p><strong>Source model:</strong> {html.escape(str(candidate.get('source_model')))} @ {source_conf:.1%}</p>",
                f"<p><strong>Best metadata:</strong> {html.escape(str(candidate.get('best_model')))} @ {best_conf:.1%}</p>",
                f"<p><strong>Agreement:</strong> {html.escape(str(candidate.get('agreement_count')))} agent(s)</p>",
                "<p class='muted'>This is the packet stored in collective memory and reused by peer models for validation/retraining.</p>",
                "</div>",
                '<div class="flow-card">',
                "<h3>3. Same Packet: Before vs After Retraining</h3>",
                '<div class="table-wrap compact"><table><thead><tr>',
                "<th>Model</th><th>Before</th><th>After</th><th>Gain</th>",
                "</tr></thead><tbody>",
                "".join(rows),
                "</tbody></table></div>",
                "<p class='muted'>These are student-head person confidences for the same shared packet crop.</p>",
                "</div>",
                "</div>",
                "</article>",
            ]
        )
    return '<article class="panel wide"><p class="muted">Run a calibrated video with visible people to show packet transfer and before/after retraining for the same frame.</p></article>'


def _timeline(run: Dict[str, object]) -> str:
    by_agent: Dict[str, List[Dict[str, str]]] = {}
    for row in run["metrics"]:
        by_agent.setdefault(str(row.get("agent_id", "agent")), []).append(row)
    lines = []
    for agent, rows in sorted(by_agent.items()):
        rows = sorted(rows, key=lambda item: _safe_int(item.get("frame")))
        ticks = []
        for row in rows[:100]:
            decision = row.get("decision", "")
            cross = row.get("cross_agent_reuse") == "1"
            klass = "heavy" if decision == "heavy" else "cross" if cross else "skip"
            title = f"{agent} frame {row.get('frame')} {decision}"
            ticks.append(f'<span class="tick {klass}" title="{html.escape(title)}"></span>')
        lines.append(
            '<div class="timeline-row">'
            f'<label>{html.escape(agent)}</label>'
            f'<div class="ticks">{"".join(ticks)}</div>'
            "</div>"
        )
    return "".join(lines)


def _timeline_section(runs: Sequence[Dict[str, object]]) -> str:
    panels = []
    for run in runs:
        panels.append(
            '<article class="panel wide">'
            f"<h3>{html.escape(str(run['name']))}: Decisions Over Frames</h3>"
            + _timeline(run)
            + '<p class="legend"><span class="dot heavy"></span> Heavy inference '
            + '<span class="dot skip"></span> Self/packet skip '
            + '<span class="dot cross"></span> Cross-agent reuse</p>'
            + "</article>"
        )
    return "".join(panels)


def _video_recommendations() -> str:
    return """
    <div class="recommend-grid">
      <article><strong>Stable indoor scene</strong><span>Good for high skip rates and clear packet reuse.</span></article>
      <article><strong>Crowded or moving people</strong><span>Good for lower similarity and more varied calibration behavior.</span></article>
      <article><strong>Lighting change</strong><span>Good for showing robustness limits and adaptation need.</span></article>
      <article><strong>Different camera angle</strong><span>Good for making cross-backbone reuse less predictable.</span></article>
    </div>
    """


def build_dashboard(demo_root: Path, output: Path) -> None:
    runs = _discover_runs(demo_root) if demo_root.exists() else []
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Swarm Knowledge Transfer Demo Dashboard</title>
  <style>
    :root {{
      --bg: #0b1020; --panel: #121a33; --panel-2: #17213f; --text: #eef3ff;
      --muted: #9fb0d0; --line: #25345d; --green: #51d88a; --blue: #6aa9ff;
      --violet: #b38cff; --amber: #ffd166; --red: #ff6b7a;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background:
      radial-gradient(circle at 15% 10%, #1e376f 0, transparent 28rem),
      radial-gradient(circle at 85% 5%, #432778 0, transparent 24rem), var(--bg);
      color: var(--text); }}
    header {{ padding: 56px 6vw 28px; }}
    header p {{ color: var(--muted); max-width: 900px; font-size: 1.08rem; line-height: 1.65; }}
    h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 5vw, 4.6rem); letter-spacing: -0.06em; }}
    h2 {{ margin: 0 0 18px; font-size: 1.45rem; }}
    h3 {{ margin: 0 0 14px; font-size: 1rem; color: #dfe7ff; }}
    main {{ padding: 0 6vw 56px; }}
    section {{ margin-top: 24px; }}
    .hero-chip {{ display: inline-flex; gap: 8px; align-items: center; padding: 8px 12px;
      border: 1px solid var(--line); border-radius: 999px; background: rgba(255,255,255,.06);
      color: #dbe6ff; font-size: .9rem; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }}
    .metric-card, .panel {{ border: 1px solid var(--line); background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
      border-radius: 22px; box-shadow: 0 18px 50px rgba(0,0,0,.25); }}
    .metric-card {{ padding: 20px; min-height: 145px; }}
    .metric-card p {{ margin: 0; color: var(--muted); }}
    .metric-card strong {{ display: block; margin: 14px 0 8px; font-size: 2.2rem; letter-spacing: -0.04em; }}
    .metric-card span {{ color: var(--muted); }}
    .metric-card.violet strong {{ color: var(--violet); }} .metric-card.green strong {{ color: var(--green); }}
    .metric-card.blue strong {{ color: var(--blue); }} .metric-card.amber strong {{ color: var(--amber); }}
    .panel-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .panel {{ padding: 20px; overflow: hidden; }} .panel.wide {{ grid-column: 1 / -1; }}
    .table-wrap {{ overflow-x: auto; border-radius: 16px; border: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; background: rgba(10,15,30,.45); }}
    .compact table {{ min-width: 0; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid rgba(159,176,208,.18); text-align: left; vertical-align: middle; }}
    th {{ color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }}
    td {{ color: #edf3ff; }}
    code {{ color: var(--amber); }}
    .bar-shell {{ position: relative; height: 28px; min-width: 130px; border-radius: 999px; background: rgba(255,255,255,.08); overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--blue), var(--green)); }}
    .bar-shell span {{ position: absolute; inset: 0; display: grid; place-items: center; font-size: .82rem; font-weight: 700; }}
    .timeline-row {{ display: grid; grid-template-columns: 88px 1fr; align-items: center; gap: 12px; margin: 10px 0; }}
    .timeline-row label {{ color: var(--muted); }}
    .ticks {{ display: flex; gap: 3px; min-height: 26px; align-items: center; }}
    .tick {{ display: inline-block; width: 9px; height: 22px; border-radius: 999px; background: var(--green); opacity: .9; }}
    .tick.heavy, .dot.heavy {{ background: var(--red); }} .tick.skip, .dot.skip {{ background: var(--green); }}
    .tick.cross, .dot.cross {{ background: var(--violet); }}
    .legend {{ color: var(--muted); display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
    .dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 4px; }}
    .packet-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .packet-card {{ border: 1px solid var(--line); border-radius: 18px; overflow: hidden; background: rgba(255,255,255,.045); }}
    .packet-image {{ position: relative; aspect-ratio: 16 / 10; background: #050a14; overflow: hidden; }}
    .packet-image img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .packet-image.empty {{ display: grid; place-items: center; color: var(--muted); }}
    .bbox {{ position: absolute; border: 3px solid var(--red); box-shadow: 0 0 0 2px rgba(0,0,0,.35); border-radius: 8px; }}
    .packet-meta {{ display: grid; gap: 5px; padding: 12px; }}
    .packet-meta span {{ color: var(--muted); font-size: .86rem; }}
    .showcase {{ border-color: rgba(179,140,255,.55); }}
    .showcase-grid {{ display: grid; grid-template-columns: 1.1fr .9fr 1fr; gap: 16px; align-items: stretch; }}
    .flow-card {{ border: 1px solid var(--line); border-radius: 18px; padding: 16px; background: rgba(255,255,255,.045); }}
    .flow-card p {{ color: var(--muted); margin: 10px 0; }}
    .flow-card strong {{ color: var(--text); }}
    .delta {{ color: var(--green); font-weight: 800; }}
    .muted {{ color: var(--muted); }}
    .story {{ display: grid; grid-template-columns: 1.2fr .8fr; gap: 18px; }}
    .story p, .recommend-grid span {{ color: var(--muted); line-height: 1.6; }}
    .recommend-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .recommend-grid article {{ padding: 16px; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,.045); }}
    .recommend-grid strong, .recommend-grid span {{ display: block; }}
    footer {{ padding: 26px 6vw 48px; color: var(--muted); }}
    @media (max-width: 1000px) {{ .metrics, .panel-grid, .story, .recommend-grid, .packet-grid, .showcase-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <span class="hero-chip">Live Swarm Demo • Packet Transfer • Retraining</span>
    <h1>Knowledge Transfer And Live Retraining</h1>
    <p>This dashboard focuses only on the live knowledge flow: a model detects a person, generates a rich packet, peer models validate it, the best metadata is selected, and lightweight student heads are retrained from validated packet crops.</p>
  </header>
  <main>
    <section class="metrics">{_comparison_cards(runs)}</section>

    <section class="story">
      <article class="panel">
        <h2>Live Demo Flow</h2>
        <p><strong>1. Packet generation:</strong> the main model detects a person and stores label, confidence, coordinates, timestamp/id, full image, source model, and embedding.</p>
        <p><strong>2. Packet transfer:</strong> peer models receive and validate the shared image packet instead of blindly trusting one detector.</p>
        <p><strong>3. Retraining:</strong> validated person crops and background crops train lightweight student heads for each backbone.</p>
      </article>
      <article class="panel">
        <h2>What Is Being Learned</h2>
        <p>The demo retrains lightweight person/background heads on top of frozen backbones. This gives visible before/after learning without pretending that full detector backbones are fine-tuned live.</p>
        <p><code>{html.escape(str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output))}</code></p>
      </article>
    </section>

    <section class="panel-grid">
      <article class="panel wide">
        <h2>Same Frame Transfer Story</h2>
        <p class="muted">This is the core spectator view: one detected frame, the generated packet, peer validation, and before/after retraining confidence for every model.</p>
      </article>
      {_focused_transfer_showcase(runs)}
    </section>

    <section class="panel-grid">
      <article class="panel wide">
        <h2>Retraining Metrics For All Models</h2>
        <p class="muted">The same validated packet crops are used to retrain lightweight heads for the main model and peer models.</p>
      </article>
      {_retraining_section(runs)}
    </section>

    <section class="panel-grid">
      <article class="panel wide">
        <h2>Packet Transfer Store Evidence</h2>
        <p class="muted">This is the collective store view: packet source, peer reuse, and cross-backbone transfer.</p>
      </article>
      {_transfer_section(runs)}
    </section>
  </main>
  <footer>Generated from local CSV/JSON artifacts in <code>{html.escape(str(demo_root))}</code>.</footer>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    print(f"Wrote dashboard: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local demo dashboard")
    parser.add_argument("--demo-root", default=str(DEFAULT_DEMO_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_DEMO_ROOT / "index.html"))
    args = parser.parse_args()

    build_dashboard(Path(args.demo_root).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
