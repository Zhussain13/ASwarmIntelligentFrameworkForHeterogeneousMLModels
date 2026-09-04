"""Run a local multi-agent experiment with shared FakeRedis collective memory.

Publication-oriented extensions:
- True heterogeneous backbone assignment across agents.
- Optional cross-backbone calibration before the run.
- Real heavy-inference path via OpenCV HOG person detector.
- Reproducible artifacts and richer per-decision provenance logs.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import fakeredis
import numpy as np
import pandas as pd

from swarm_kt.agent_communication import AgentCommunication
from swarm_kt.collective_memory import CollectiveMemory
from swarm_kt.config import Config
from swarm_kt.encoder import Encoder
from swarm_kt.knowledge_packet import KnowledgePacket


def _backbone_capacity(backbone_name: str) -> float:
    caps = {
        "resnet18": 0.72,
        "mobilenet_v3_small": 0.58,
        "efficientnet_b0": 0.76,
        "fallback_histogram": 0.30,
    }
    return float(caps.get(backbone_name, 0.50))


def _assign_backbones(num_agents: int, mode: str, custom_list: List[str]) -> List[str]:
    if mode == "none":
        return ["resnet18"] * num_agents
    if mode == "diverse":
        presets = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
        return [presets[i % len(presets)] for i in range(num_agents)]
    if mode == "custom":
        if not custom_list:
            raise ValueError("--backbones is required when --hetero-backbone custom")
        return [custom_list[i % len(custom_list)] for i in range(num_agents)]
    raise ValueError(f"Unsupported hetero-backbone mode: {mode}")


def _write_manifest(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _prepare_frames(video_mode: bool, video_path: str, max_frames: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    frames: List[np.ndarray] = []

    if video_mode:
        import cv2

        cap = cv2.VideoCapture(video_path)
        while len(frames) < max_frames:
            ret, img = cap.read()
            if not ret:
                break
            frames.append(img)
        cap.release()

    while len(frames) < max_frames:
        base = np.full((240, 320, 3), 128, dtype=np.uint8)
        noise = rng.normal(0, 4, size=(240, 320, 3)).astype(np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(img)

    return frames[:max_frames]


def _encode_frame_payload(frame: np.ndarray) -> Optional[Dict[str, object]]:
    try:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if not ok:
            return None
        height, width = frame.shape[:2]
        return {
            "format": "jpeg",
            "width": int(width),
            "height": int(height),
            "b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    except Exception:
        return None


def _decode_frame_payload(payload: Optional[Dict[str, object]]) -> Optional[np.ndarray]:
    if not payload or payload.get("format") != "jpeg" or not payload.get("b64"):
        return None
    try:
        import cv2

        raw = base64.b64decode(str(payload["b64"]).encode("ascii"))
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _bbox_dict(rect) -> Optional[Dict[str, int]]:
    if rect is None:
        return None
    x, y, w, h = [int(v) for v in rect]
    return {"x": x, "y": y, "w": w, "h": h}


def _run_heavy_inference(
    detector_mode: str,
    frame: np.ndarray,
    cfg: Config,
    detector_obj,
) -> Tuple[str, float, float, Optional[Dict[str, int]]]:
    t0 = time.perf_counter()
    if detector_mode == "simulated":
        time.sleep(cfg.heavy_inference_ms / 1000.0)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        height, width = frame.shape[:2]
        bbox = {"x": int(width * 0.35), "y": int(height * 0.20), "w": int(width * 0.30), "h": int(height * 0.65)}
        return "Person", 0.90, dt_ms, bbox

    # opencv_hog
    rects, weights = detector_obj.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    if len(rects) > 0:
        best_idx = int(np.argmax(weights)) if len(weights) > 0 else 0
        if len(weights) > 0:
            wmax = float(np.max(weights))
            conf = float(1.0 / (1.0 + np.exp(-wmax)))
        else:
            conf = 0.75
        conf = float(max(0.51, min(conf, 0.999)))
        return "Person", conf, dt_ms, _bbox_dict(rects[best_idx])
    return "Background", 0.60, dt_ms, None


def _calibrate_encoders(
    encoders: Dict[str, Encoder],
    agent_ids: List[str],
    frames: List[np.ndarray],
    calib_frames: int,
    calib_epochs: int,
    calib_lr: float,
    artifacts_dir: str,
) -> List[Dict[str, object]]:
    logs: List[Dict[str, object]] = []
    if calib_frames <= 0:
        return logs

    ref_agent = agent_ids[0]
    ref_enc = encoders[ref_agent]

    ref_targets = []
    max_use = min(calib_frames, len(frames))
    for idx in range(max_use):
        _, emb = ref_enc.get_feature_and_embedding(frames[idx])
        ref_targets.append(emb)
    if not ref_targets:
        return logs
    target_arr = np.stack(ref_targets)

    for aid in agent_ids[1:]:
        enc = encoders[aid]
        feats = []
        for idx in range(max_use):
            feat, _ = enc.get_feature_and_embedding(frames[idx])
            if feat is None:
                feats = []
                break
            feats.append(feat)
        if not feats:
            logs.append(
                {
                    "agent_id": aid,
                    "backbone": enc.backbone_name,
                    "status": "skipped_no_torch_features",
                    "samples": 0,
                    "epochs": calib_epochs,
                    "lr": calib_lr,
                    "loss": "",
                }
            )
            continue

        loss = enc.adapt_projection(feats, target_arr, epochs=calib_epochs, lr=calib_lr)
        logs.append(
            {
                "agent_id": aid,
                "backbone": enc.backbone_name,
                "status": "ok",
                "samples": len(feats),
                "epochs": calib_epochs,
                "lr": calib_lr,
                "loss": loss,
            }
        )

    out_csv = os.path.join(artifacts_dir, "calibration_log.csv")
    with open(out_csv, "w", newline="") as fh:
        fieldnames = ["agent_id", "backbone", "status", "samples", "epochs", "lr", "loss"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in logs:
            writer.writerow(row)

    return logs


def _append_jsonl(path: str, payload: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _write_validated_training_candidates(artifacts_dir: str) -> None:
    packet_rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    for name in os.listdir(artifacts_dir):
        path = os.path.join(artifacts_dir, name)
        if name.startswith("knowledge_packets_") and name.endswith(".jsonl"):
            packet_rows.extend(_read_jsonl(path))
        if name.startswith("packet_validations_") and name.endswith(".jsonl"):
            validation_rows.extend(_read_jsonl(path))

    by_detection: Dict[str, Dict[str, object]] = {}
    for pkt in packet_rows:
        detection_id = str(pkt.get("detection_id") or "")
        if not detection_id:
            continue
        by_detection[detection_id] = {
            "detection_id": detection_id,
            "label": pkt.get("object_label"),
            "coordinates": pkt.get("coordinates"),
            "source_agent": pkt.get("agent_id"),
            "source_model": pkt.get("source_model") or pkt.get("model_name"),
            "source_confidence": pkt.get("confidence"),
            "best_agent": pkt.get("agent_id"),
            "best_model": pkt.get("source_model") or pkt.get("model_name"),
            "best_confidence": float(pkt.get("confidence") or 0.0),
            "agreement_count": 1,
            "verified_by": [],
            "_verified_agents": {pkt.get("agent_id")},
            "image": pkt.get("image"),
        }

    for val in validation_rows:
        detection_id = str(val.get("detection_id") or "")
        candidate = by_detection.get(detection_id)
        if candidate is None:
            continue
        validator = {
            "agent_id": val.get("validator_agent"),
            "model": val.get("validator_model"),
            "label": val.get("label"),
            "confidence": val.get("confidence"),
            "coordinates": val.get("coordinates"),
        }
        candidate["verified_by"].append(validator)
        if val.get("label") == candidate.get("label"):
            candidate["_verified_agents"].add(val.get("validator_agent"))
        confidence = float(val.get("confidence") or 0.0)
        if confidence > float(candidate["best_confidence"]):
            candidate["best_confidence"] = confidence
            candidate["best_agent"] = val.get("validator_agent")
            candidate["best_model"] = val.get("validator_model")
            candidate["coordinates"] = val.get("coordinates") or candidate.get("coordinates")

    candidates = sorted(by_detection.values(), key=lambda row: str(row["detection_id"]))
    for candidate in candidates:
        verified_agents = {agent for agent in candidate.pop("_verified_agents", set()) if agent}
        candidate["agreement_count"] = len(verified_agents)
    out_json = os.path.join(artifacts_dir, "validated_training_candidates.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(candidates, fh, indent=2)

    out_csv = os.path.join(artifacts_dir, "validated_training_candidates.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "detection_id",
            "label",
            "source_agent",
            "source_model",
            "source_confidence",
            "best_agent",
            "best_model",
            "best_confidence",
            "agreement_count",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow({key: row.get(key) for key in fieldnames})


def agent_thread(
    agent_id: str,
    cfg: Config,
    memory: CollectiveMemory,
    frames: List[np.ndarray],
    delay: float,
    encoder: Encoder,
    prefer_peer_transfer: bool,
    metrics_dir: str,
    artifacts_dir: str,
    detector_mode: str,
):
    comm = AgentCommunication(
        collective_memory=memory,
        agent_id=agent_id,
        broker_host=cfg.mqtt_broker_host,
        broker_port=cfg.mqtt_broker_port,
        topic=cfg.mqtt_topic,
    )

    detector_obj = None
    if detector_mode == "opencv_hog":
        import cv2

        detector_obj = cv2.HOGDescriptor()
        detector_obj.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    time.sleep(delay)
    metrics = []

    try:
        for frame_idx, img in enumerate(frames, start=1):
            feat_tensor, emb = encoder.get_feature_and_embedding(img)

            best_pkt = None
            best_sim = 0.0
            if prefer_peer_transfer:
                best_pkt, best_sim = memory.get_best_match(emb, exclude_agent_id=agent_id)
            if best_pkt is None:
                best_pkt, best_sim = memory.get_best_match(emb)

            match_info = None
            used_packet_agent = None
            used_packet_ts = None
            used_packet_model_name = None
            used_packet_model_version = None
            used_packet_detection_id = None
            used_packet_coordinates = None
            heavy_label = None
            heavy_confidence = None
            heavy_latency_ms = None
            heavy_coordinates = None
            packet_detection_id = None

            if best_pkt is not None:
                decision = "skipped"
                match_info = {"sim": best_sim, "label": best_pkt.object_label, "packet": best_pkt}
                used_packet_agent = best_pkt.agent_id
                used_packet_ts = best_pkt.timestamp
                used_packet_model_name = best_pkt.model_name
                used_packet_model_version = best_pkt.model_version
                used_packet_detection_id = best_pkt.detection_id
                used_packet_coordinates = best_pkt.coordinates
            else:
                decision = "heavy"
                heavy_label, heavy_confidence, heavy_latency_ms, heavy_coordinates = _run_heavy_inference(
                    detector_mode,
                    img,
                    cfg,
                    detector_obj,
                )
                packet_detection_id = f"{agent_id}_frame{frame_idx}_{int(time.time() * 1000)}"
                pkt = KnowledgePacket(
                    agent_id=agent_id,
                    timestamp=time.time(),
                    object_label=heavy_label,
                    confidence=heavy_confidence,
                    embedding=emb,
                    model_name=encoder.backbone_name,
                    model_version="torchvision",
                    model_capacity=_backbone_capacity(encoder.backbone_name),
                    detection_id=packet_detection_id,
                    coordinates=heavy_coordinates,
                    image=_encode_frame_payload(img),
                    source_model=encoder.backbone_name,
                    best_model=encoder.backbone_name,
                    best_confidence=heavy_confidence,
                    verified_by=[agent_id],
                    agreement_count=1,
                )
                comm.publish_packet(pkt)
                packet_log = os.path.join(artifacts_dir, f"knowledge_packets_{agent_id}.jsonl")
                _append_jsonl(packet_log, pkt.to_dict())

            metrics.append(
                {
                    "agent_id": agent_id,
                    "frame": frame_idx,
                    "decision": decision,
                    "sim": match_info["sim"] if match_info else None,
                    "stored_label": match_info["label"] if match_info else None,
                    "used_packet_agent": used_packet_agent,
                    "used_packet_ts": used_packet_ts,
                    "local_backbone": encoder.backbone_name,
                    "local_embedding_dim": encoder.embedding_dim,
                    "used_packet_model_name": used_packet_model_name,
                    "used_packet_model_version": used_packet_model_version,
                    "used_packet_detection_id": used_packet_detection_id,
                    "used_packet_coordinates": json.dumps(used_packet_coordinates) if used_packet_coordinates else None,
                    "cross_agent_reuse": int(used_packet_agent is not None and used_packet_agent != agent_id),
                    "heavy_label": heavy_label,
                    "heavy_confidence": heavy_confidence,
                    "heavy_coordinates": json.dumps(heavy_coordinates) if heavy_coordinates else None,
                    "packet_detection_id": packet_detection_id,
                    "heavy_latency_ms": heavy_latency_ms,
                    "heavy_detector_mode": detector_mode,
                }
            )

            if decision == "skipped" and feat_tensor is not None and match_info is not None:
                pkt = match_info["packet"]
                if pkt.detection_id and pkt.agent_id != agent_id:
                    packet_img = _decode_frame_payload(pkt.image)
                    if packet_img is not None:
                        v_label, v_confidence, v_latency_ms, v_coordinates = _run_heavy_inference(
                            detector_mode,
                            packet_img,
                            cfg,
                            detector_obj,
                        )
                        validation_log = os.path.join(artifacts_dir, f"packet_validations_{agent_id}.jsonl")
                        _append_jsonl(
                            validation_log,
                            {
                                "timestamp": time.time(),
                                "detection_id": pkt.detection_id,
                                "source_agent": pkt.agent_id,
                                "source_model": pkt.source_model or pkt.model_name,
                                "source_confidence": pkt.confidence,
                                "validator_agent": agent_id,
                                "validator_model": encoder.backbone_name,
                                "label": v_label,
                                "confidence": v_confidence,
                                "coordinates": v_coordinates,
                                "latency_ms": v_latency_ms,
                            },
                        )

                if not hasattr(threading.current_thread(), "adapt_buf"):
                    threading.current_thread().adapt_buf = {"feats": [], "targets": []}
                buf = threading.current_thread().adapt_buf
                buf["feats"].append(feat_tensor)
                buf["targets"].append(match_info["packet"].embedding)
                if len(buf["feats"]) >= cfg.adapt_batch_size:
                    try:
                        loss = encoder.adapt_projection(
                            buf["feats"],
                            np.stack(buf["targets"]),
                            epochs=cfg.adapt_epochs,
                            lr=cfg.adapt_lr,
                        )
                        ats = int(time.time())
                        alog = os.path.join(metrics_dir, f"adaptation_log_{agent_id}_{ats}.csv")
                        with open(alog, "a", newline="") as fh:
                            w = csv.writer(fh)
                            w.writerow([time.time(), loss, len(buf["feats"]), encoder.backbone_name])
                        buf["feats"] = []
                        buf["targets"] = []
                    except Exception:
                        pass

    finally:
        try:
            comm.stop()
        except Exception:
            pass

    ts = int(time.time())
    out = os.path.join(metrics_dir, f"metrics_{agent_id}_{ts}.csv")
    with open(out, "w", newline="") as fh:
        fieldnames = [
            "agent_id",
            "frame",
            "decision",
            "sim",
            "stored_label",
            "used_packet_agent",
            "used_packet_ts",
            "local_backbone",
            "local_embedding_dim",
            "used_packet_model_name",
            "used_packet_model_version",
            "used_packet_detection_id",
            "used_packet_coordinates",
            "cross_agent_reuse",
            "heavy_label",
            "heavy_confidence",
            "heavy_coordinates",
            "packet_detection_id",
            "heavy_latency_ms",
            "heavy_detector_mode",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    print(f"Agent {agent_id} ({encoder.backbone_name}) finished, wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Run local experiment with heterogeneous backbones")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--mode", choices=["synthetic", "video"], default="synthetic")
    parser.add_argument("--video", type=str, default=None, help="Path to video file when using video mode")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--stagger", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hetero", choices=["none", "simulated"], default="none", help="Simulate embedding-dimension heterogeneity")
    parser.add_argument(
        "--hetero-backbone",
        choices=["none", "diverse", "custom"],
        default="none",
        help="Assign different backbone architectures across agents",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default="",
        help="Comma-separated backbones for custom mode, e.g. resnet18,mobilenet_v3_small,efficientnet_b0",
    )
    parser.add_argument("--prefer-peer-transfer", action="store_true", help="Try peer packets first before self packets")
    parser.add_argument("--use-pretrained-backbone", action="store_true", help="Request pretrained torchvision weights")
    parser.add_argument("--similarity-threshold", type=float, default=None, help="Override similarity threshold")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--output-dir", type=str, default=".", help="Output root directory for this run")
    parser.add_argument("--heavy-detector", choices=["simulated", "opencv_hog"], default="opencv_hog")
    parser.add_argument("--calibrate-frames", type=int, default=0, help="Number of warm-up frames used for cross-backbone calibration")
    parser.add_argument("--calibrate-epochs", type=int, default=10)
    parser.add_argument("--calibrate-lr", type=float, default=1e-3)
    args = parser.parse_args()

    np.random.seed(args.seed)

    cfg = Config.load(None)
    if args.similarity_threshold is not None:
        cfg.similarity_threshold = float(args.similarity_threshold)
    if args.confidence_threshold is not None:
        cfg.confidence_threshold = float(args.confidence_threshold)

    output_dir = os.path.abspath(args.output_dir)
    metrics_dir = os.path.join(output_dir, "metrics")
    artifacts_dir = os.path.join(output_dir, "artifacts")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    frames = _prepare_frames(args.mode == "video", args.video, args.max_frames, args.seed)

    server = fakeredis.FakeServer()
    shared_client = fakeredis.FakeStrictRedis(server=server)

    memory = CollectiveMemory(
        redis_client=shared_client,
        ttl=cfg.redis_ttl,
        similarity_threshold=cfg.similarity_threshold,
        confidence_threshold=cfg.confidence_threshold,
        max_scan=cfg.max_scan,
    )

    num_agents = int(args.num_agents)
    embedding_dims = [cfg.embedding_dim] * num_agents
    if args.hetero == "simulated":
        presets = [max(64, cfg.embedding_dim // 2), cfg.embedding_dim, min(512, cfg.embedding_dim * 2)]
        for i in range(num_agents):
            embedding_dims[i] = presets[i % len(presets)]

    custom_backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    backbones = _assign_backbones(num_agents, args.hetero_backbone, custom_backbones)

    encoders: Dict[str, Encoder] = {}
    actual_backbones: Dict[str, str] = {}
    agent_ids = [f"agent_{i + 1}" for i in range(num_agents)]
    for i, aid in enumerate(agent_ids):
        enc = Encoder(
            embedding_dim=embedding_dims[i],
            backbone_name=backbones[i],
            use_pretrained=bool(args.use_pretrained_backbone),
        )
        encoders[aid] = enc
        actual_backbones[aid] = enc.backbone_name

    calibration_log = _calibrate_encoders(
        encoders=encoders,
        agent_ids=agent_ids,
        frames=frames,
        calib_frames=int(args.calibrate_frames),
        calib_epochs=int(args.calibrate_epochs),
        calib_lr=float(args.calibrate_lr),
        artifacts_dir=artifacts_dir,
    )

    manifest = {
        "timestamp": int(time.time()),
        "num_agents": num_agents,
        "mode": args.mode,
        "video": args.video,
        "max_frames": args.max_frames,
        "seed": args.seed,
        "stagger": args.stagger,
        "hetero_embedding_mode": args.hetero,
        "embedding_dims": embedding_dims,
        "hetero_backbone_mode": args.hetero_backbone,
        "requested_backbones": backbones,
        "actual_backbones": actual_backbones,
        "prefer_peer_transfer": bool(args.prefer_peer_transfer),
        "use_pretrained_backbone": bool(args.use_pretrained_backbone),
        "similarity_threshold": cfg.similarity_threshold,
        "confidence_threshold": cfg.confidence_threshold,
        "heavy_detector": args.heavy_detector,
        "calibrate_frames": int(args.calibrate_frames),
        "calibrate_epochs": int(args.calibrate_epochs),
        "calibrate_lr": float(args.calibrate_lr),
        "calibration_log": calibration_log,
    }
    _write_manifest(os.path.join(artifacts_dir, "run_manifest.json"), manifest)

    print("Run manifest:")
    print(json.dumps(manifest, indent=2))

    threads = []
    for i, aid in enumerate(agent_ids):
        t = threading.Thread(
            target=agent_thread,
            args=(
                aid,
                cfg,
                memory,
                frames,
                i * args.stagger,
                encoders[aid],
                bool(args.prefer_peer_transfer),
                metrics_dir,
                artifacts_dir,
                args.heavy_detector,
            ),
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    _write_validated_training_candidates(artifacts_dir)

    files = sorted(
        [
            os.path.join(metrics_dir, f)
            for f in os.listdir(metrics_dir)
            if f.startswith("metrics_agent_") and f.endswith(".csv")
        ]
    )
    if files:
        dfs = [pd.read_csv(f) for f in files]
        agg = pd.concat(dfs, ignore_index=True)
        ts = int(time.time())
        agg_out = os.path.join(metrics_dir, f"swarm_aggregate_{ts}.csv")
        agg.to_csv(agg_out, index=False)
        print(f"Wrote {agg_out}")

        skip = agg[(agg["decision"] == "skipped") & (agg["used_packet_agent"].notna())].copy()
        if not skip.empty:
            xfer = pd.crosstab(skip["used_packet_agent"], skip["agent_id"])
            xfer_out = os.path.join(metrics_dir, "cross_agent_transfer.csv")
            xfer.to_csv(xfer_out)
            print(f"Wrote {xfer_out}")

            if "used_packet_model_name" in skip.columns and "local_backbone" in skip.columns:
                bb_xfer = pd.crosstab(skip["used_packet_model_name"], skip["local_backbone"])
                bb_xfer_out = os.path.join(metrics_dir, "cross_backbone_transfer.csv")
                bb_xfer.to_csv(bb_xfer_out)
                print(f"Wrote {bb_xfer_out}")


if __name__ == "__main__":
    main()
