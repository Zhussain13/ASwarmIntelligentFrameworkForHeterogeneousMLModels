"""Browser frontend for the live swarm training/adaptation demo.

Run:
    python tools/demo_frontend.py

Then open:
    http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, Response, jsonify, request, send_file


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demo_runs"
FRONTEND_INPUTS = DEMO_ROOT / "frontend_inputs"
RUNNER = ROOT / "tools" / "run_training_demo.py"
DASHBOARD = DEMO_ROOT / "index.html"
DEFAULT_VIDEO = ROOT / "data" / "video_for_swarm.avi"

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)

JOB_LOCK = threading.Lock()
JOB: Dict[str, object] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "status": "idle",
    "logs": [],
    "summary": None,
    "command": None,
    "dashboard": str(DASHBOARD),
}


def _append_log(line: str) -> None:
    with JOB_LOCK:
        logs = JOB.setdefault("logs", [])
        assert isinstance(logs, list)
        logs.append(line.rstrip())
        if len(logs) > 500:
            del logs[: len(logs) - 500]


def _set_job(**updates: object) -> None:
    with JOB_LOCK:
        JOB.update(updates)


def _snapshot_job() -> Dict[str, object]:
    with JOB_LOCK:
        return json.loads(json.dumps(JOB, default=str))


def _candidate_media() -> List[Dict[str, str]]:
    roots = [ROOT / "data", Path("D:/Downloads")]
    items: List[Dict[str, str]] = []
    for base in roots:
        if not base.exists():
            continue
        try:
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                    items.append({"type": "video", "path": str(path), "name": path.name})
                if len(items) >= 30:
                    return items
        except OSError:
            continue
    return items


def _make_video_from_images(image_dir: Path, max_frames: int) -> Path:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to convert an image folder into a demo video") from exc

    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    FRONTEND_INPUTS.mkdir(parents=True, exist_ok=True)
    output = FRONTEND_INPUTS / f"images_as_video_{int(time.time())}.avi"

    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Could not read first image: {images[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"XVID"),
        8.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video file: {output}")

    try:
        for path in images[:max_frames]:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()

    return output


def _resolve_media(media_type: str, media_path: str, max_frames: int) -> Path:
    if not media_path.strip():
        raise ValueError("Please provide a video file path or image folder path.")
    path = Path(media_path.strip().strip('"')).resolve()
    if media_type == "video":
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        if path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Unsupported video extension: {path.suffix}")
        return path
    if media_type == "images":
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Image folder not found: {path}")
        return _make_video_from_images(path, max_frames)
    raise ValueError(f"Unsupported media type: {media_type}")


def _run_background(payload: Dict[str, object]) -> None:
    try:
        media_type = str(payload.get("media_type", "video"))
        media_path = str(payload.get("media_path", str(DEFAULT_VIDEO)))
        scenario = str(payload.get("scenario", "compare-video"))
        max_frames = int(payload.get("max_frames") or 80)
        if max_frames < 5:
            raise ValueError("max_frames must be at least 5 for a visible demo.")

        video = _resolve_media(media_type, media_path, max_frames)
        cmd = [
            sys.executable,
            str(RUNNER),
            "--scenario",
            scenario,
            "--video",
            str(video),
            "--max-frames",
            str(max_frames),
            "--output-root",
            str(DEMO_ROOT),
        ]

        _set_job(
            running=True,
            status="running",
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            finished_at=None,
            logs=[],
            summary=None,
            command=" ".join(cmd),
        )
        _append_log("Starting live demo run...")
        _append_log(f"Media type: {media_type}")
        _append_log(f"Resolved video input: {video}")
        _append_log("Command: " + " ".join(cmd))

        env = os.environ.copy()
        env.setdefault("TORCH_HOME", str(ROOT / ".torch_cache"))
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _append_log(line)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Demo command failed with exit code {code}")

        summary_path = DEMO_ROOT / "demo_summary.json"
        summary = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _set_job(
            running=False,
            status="complete",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            summary=summary,
        )
        _append_log("Demo run complete. Dashboard refreshed.")
    except Exception as exc:
        _set_job(
            running=False,
            status="error",
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        _append_log(f"ERROR: {exc}")


@app.get("/")
def index() -> Response:
    default_video = html_escape(str(DEFAULT_VIDEO))
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Swarm Demo Control</title>
  <style>
    :root {{ --bg:#09101f; --panel:#121c35; --panel2:#182746; --text:#edf4ff; --muted:#9fb1d4;
      --line:#2b3c68; --green:#51d88a; --blue:#6aa9ff; --violet:#b38cff; --red:#ff6b7a; --amber:#ffd166; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--text);
      background: radial-gradient(circle at top left,#203d7a,transparent 34rem), radial-gradient(circle at top right,#4b2679,transparent 30rem), var(--bg); }}
    header, main {{ width:min(1180px,92vw); margin:auto; }}
    header {{ padding:42px 0 20px; }}
    h1 {{ font-size:clamp(2rem,5vw,4rem); margin:0 0 10px; letter-spacing:-.05em; }}
    p {{ color:var(--muted); line-height:1.6; }}
    .grid {{ display:grid; grid-template-columns: .9fr 1.1fr; gap:18px; align-items:start; }}
    .panel {{ background:linear-gradient(180deg,rgba(255,255,255,.07),rgba(255,255,255,.035)); border:1px solid var(--line);
      border-radius:24px; padding:22px; box-shadow:0 18px 48px rgba(0,0,0,.28); }}
    label {{ display:block; color:#dbe6ff; font-weight:700; margin:14px 0 8px; }}
    input, select {{ width:100%; padding:13px 14px; border-radius:14px; border:1px solid var(--line); background:#0b1428; color:var(--text); }}
    button {{ border:0; border-radius:16px; padding:14px 18px; font-weight:800; color:#08111f; background:linear-gradient(90deg,var(--green),var(--blue)); cursor:pointer; margin-top:18px; }}
    button:disabled {{ opacity:.55; cursor:not-allowed; }}
    .hint {{ font-size:.9rem; color:var(--muted); }}
    .chips {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
    .chip {{ padding:12px; border:1px solid var(--line); border-radius:16px; background:rgba(255,255,255,.045); }}
    .chip strong {{ display:block; color:#fff; }}
    pre {{ height:300px; overflow:auto; white-space:pre-wrap; background:#050a14; border:1px solid var(--line); padding:14px; border-radius:16px; color:#dce8ff; }}
    .status {{ display:flex; gap:10px; align-items:center; margin-bottom:12px; }}
    .dot {{ width:12px; height:12px; border-radius:50%; background:var(--amber); }}
    .dot.running {{ background:var(--blue); animation:pulse 1s infinite alternate; }} .dot.complete {{ background:var(--green); }} .dot.error {{ background:var(--red); }}
    iframe {{ width:100%; height:760px; border:1px solid var(--line); border-radius:22px; background:#fff; }}
    .wide {{ grid-column:1/-1; }}
    .candidates button {{ margin:6px 6px 0 0; padding:8px 10px; font-size:.82rem; background:#22345f; color:#dce8ff; }}
    @keyframes pulse {{ from {{ opacity:.45 }} to {{ opacity:1 }} }}
    @media(max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} .chips {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Live Swarm Training/Adaptation Demo</h1>
    <p>Choose the kind of real media you have, run calibration/inference from this frontend, then show the generated dashboard. No synthetic calibration is used here.</p>
  </header>
  <main class="grid">
    <section class="panel">
      <h2>1. What media do you have?</h2>
      <label>Input type</label>
      <select id="mediaType">
        <option value="video">Video file: corridor, classroom, outdoor, crowd, lighting change</option>
        <option value="images">Image folder: frames/photos to convert into a demo video</option>
      </select>
      <label>Video file path or image folder path</label>
      <input id="mediaPath" value="{default_video}">
      <p class="hint">For different calibration scores, try crowded movement, lighting changes, outdoor scenes, or a different camera angle.</p>
      <div class="candidates" id="candidates"></div>

      <label>Demo run</label>
      <select id="scenario">
        <option value="compare-video">Before vs after calibration comparison</option>
        <option value="video-calib">Calibration only</option>
        <option value="video-no-calib">No-calibration baseline only</option>
      </select>
      <label>Frames to process</label>
      <input id="maxFrames" type="number" min="5" max="500" value="80">
      <button id="runBtn" onclick="startRun()">Run Live Demo</button>
    </section>

    <section class="panel">
      <h2>2. Video/image choices that change scores</h2>
      <div class="chips">
        <div class="chip"><strong>Stable indoor</strong><span class="hint">Usually high similarity and high skip rate.</span></div>
        <div class="chip"><strong>Crowded movement</strong><span class="hint">More variation; peer reuse may drop.</span></div>
        <div class="chip"><strong>Lighting changes</strong><span class="hint">Good for showing calibration limits.</span></div>
        <div class="chip"><strong>Outdoor angle shift</strong><span class="hint">Backbones may diverge more.</span></div>
      </div>
      <p class="hint">If you have multiple options, run the comparison once per video and keep the one with the clearest before/after difference.</p>
    </section>

    <section class="panel wide">
      <h2>3. Live Execution Log</h2>
      <div class="status"><span id="dot" class="dot"></span><strong id="status">idle</strong></div>
      <pre id="logs"></pre>
    </section>

    <section class="panel wide">
      <h2>4. Knowledge Transfer + Retraining Dashboard</h2>
      <p class="hint">The dashboard refreshes once after a run completes, so it will not keep jumping while you inspect it.</p>
      <iframe id="dashboard" src="/dashboard"></iframe>
    </section>
  </main>
  <script>
    let pollTimer = null;
    let lastDashboardFinishedAt = null;
    let logAutoScroll = true;

    window.addEventListener('DOMContentLoaded', () => {{
      const logs = document.getElementById('logs');
      logs.addEventListener('scroll', () => {{
        logAutoScroll = logs.scrollTop + logs.clientHeight >= logs.scrollHeight - 12;
      }});
    }});

    async function loadCandidates() {{
      const res = await fetch('/api/candidates');
      const items = await res.json();
      const box = document.getElementById('candidates');
      if (!items.length) return;
      box.innerHTML = '<p class="hint">Detected videos:</p>' + items.slice(0, 8).map(item =>
        `<button type="button" onclick="pickPath(${{JSON.stringify(item.path)}})">${{item.name}}</button>`
      ).join('');
    }}
    function pickPath(path) {{ document.getElementById('mediaPath').value = path; document.getElementById('mediaType').value = 'video'; }}
    async function startRun() {{
      const payload = {{
        media_type: document.getElementById('mediaType').value,
        media_path: document.getElementById('mediaPath').value,
        scenario: document.getElementById('scenario').value,
        max_frames: Number(document.getElementById('maxFrames').value)
      }};
      document.getElementById('runBtn').disabled = true;
      await fetch('/api/run', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload)}});
      pollNow();
    }}

    function schedulePoll(delay) {{
      if (pollTimer) clearTimeout(pollTimer);
      pollTimer = setTimeout(pollNow, delay);
    }}

    async function pollNow() {{
      const res = await fetch('/api/status');
      const job = await res.json();
      document.getElementById('status').textContent = job.status;
      const dot = document.getElementById('dot');
      dot.className = 'dot ' + job.status;
      const logs = document.getElementById('logs');
      logs.textContent = (job.logs || []).join('\\n');
      if (logAutoScroll) logs.scrollTop = logs.scrollHeight;
      document.getElementById('runBtn').disabled = !!job.running;
      if (job.status === 'complete' && job.finished_at && job.finished_at !== lastDashboardFinishedAt) {{
        lastDashboardFinishedAt = job.finished_at;
        document.getElementById('dashboard').src = '/dashboard?ts=' + Date.now();
      }}
      schedulePoll(job.running ? 1500 : 5000);
    }}
    loadCandidates();
    pollNow();
  </script>
</body>
</html>"""
    return Response(page, mimetype="text/html")


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@app.get("/api/candidates")
def candidates() -> Response:
    return jsonify(_candidate_media())


@app.post("/api/run")
def run_demo() -> Response:
    with JOB_LOCK:
        if JOB.get("running"):
            return jsonify({"ok": False, "error": "A demo run is already active."}), 409
    payload = request.get_json(force=True, silent=True) or {}
    thread = threading.Thread(target=_run_background, args=(payload,), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.get("/api/status")
def status() -> Response:
    return jsonify(_snapshot_job())


@app.get("/dashboard")
def dashboard() -> Response:
    if DASHBOARD.exists():
        return send_file(DASHBOARD)
    fallback = ROOT / "tools" / "build_demo_dashboard.py"
    subprocess.run(
        [sys.executable, str(fallback), "--demo-root", str(DEMO_ROOT), "--output", str(DASHBOARD)],
        cwd=str(ROOT),
        check=False,
    )
    if DASHBOARD.exists():
        return send_file(DASHBOARD)
    return Response("<h1>No dashboard yet</h1><p>Run the live demo first.</p>", mimetype="text/html")


def main() -> None:
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    print("Live demo frontend: http://127.0.0.1:7860")
    print("Use real video or image-folder input; synthetic demo is intentionally not exposed.")
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)


if __name__ == "__main__":
    main()
