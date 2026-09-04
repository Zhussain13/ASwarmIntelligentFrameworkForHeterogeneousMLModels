"""Run a local multi-agent experiment with shared FakeRedis collective memory.

Supports heterogeneous backbone assignment across agents for cross-model
knowledge-transfer experiments.
"""
import argparse
import csv
import json
import os
import threading
import time
from typing import Dict, List

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


def agent_thread(
    agent_id: str,
    cfg: Config,
    memory: CollectiveMemory,
    video_mode: bool,
    video_path: str,
    max_frames: int,
    delay: float,
    embedding_dim: int,
    backbone_name: str,
    use_pretrained_backbone: bool,
    prefer_peer_transfer: bool,
    metrics_dir: str,
):
    enc = Encoder(
        embedding_dim=embedding_dim,
        backbone_name=backbone_name,
        use_pretrained=use_pretrained_backbone,
    )
    comm = AgentCommunication(
        collective_memory=memory,
        agent_id=agent_id,
        broker_host=cfg.mqtt_broker_host,
        broker_port=cfg.mqtt_broker_port,
        topic=cfg.mqtt_topic,
    )

    time.sleep(delay)
    metrics = []
    frame = 0

    try:
        while frame < max_frames:
            frame += 1
            if video_mode:
                import cv2

                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
                ret, img = cap.read()
                cap.release()
                if not ret:
                    img = (np.random.rand(240, 320, 3) * 255).astype("uint8")
            else:
                base = np.full((240, 320, 3), 128, dtype=np.uint8)
                noise = (np.random.randn(240, 320, 3) * 4).astype(np.int8)
                img = np.clip(base + noise, 0, 255).astype(np.uint8)

            feat_tensor, emb = enc.get_feature_and_embedding(img)

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

            if best_pkt is not None:
                decision = "skipped"
                match_info = {"sim": best_sim, "label": best_pkt.object_label, "packet": best_pkt}
                used_packet_agent = best_pkt.agent_id
                used_packet_ts = best_pkt.timestamp
                used_packet_model_name = best_pkt.model_name
                used_packet_model_version = best_pkt.model_version
            else:
                decision = "heavy"
                time.sleep(cfg.heavy_inference_ms / 1000.0)
                pkt = KnowledgePacket(
                    agent_id=agent_id,
                    timestamp=time.time(),
                    object_label="Person",
                    confidence=0.9,
                    embedding=emb,
                    model_name=backbone_name,
                    model_version="torchvision",
                    model_capacity=_backbone_capacity(backbone_name),
                )
                comm.publish_packet(pkt)

            metrics.append(
                {
                    "agent_id": agent_id,
                    "frame": frame,
                    "decision": decision,
                    "sim": match_info["sim"] if match_info else None,
                    "stored_label": match_info["label"] if match_info else None,
                    "used_packet_agent": used_packet_agent,
                    "used_packet_ts": used_packet_ts,
                    "local_backbone": backbone_name,
                    "local_embedding_dim": embedding_dim,
                    "used_packet_model_name": used_packet_model_name,
                    "used_packet_model_version": used_packet_model_version,
                }
            )

            if decision == "skipped" and feat_tensor is not None and match_info is not None:
                if not hasattr(threading.current_thread(), "adapt_buf"):
                    threading.current_thread().adapt_buf = {"feats": [], "targets": []}
                buf = threading.current_thread().adapt_buf
                buf["feats"].append(feat_tensor)
                buf["targets"].append(match_info["packet"].embedding)
                if len(buf["feats"]) >= cfg.adapt_batch_size:
                    try:
                        loss = enc.adapt_projection(
                            buf["feats"],
                            np.stack(buf["targets"]),
                            epochs=cfg.adapt_epochs,
                            lr=cfg.adapt_lr,
                        )
                        ats = int(time.time())
                        alog = os.path.join(metrics_dir, f"adaptation_log_{agent_id}_{ats}.csv")
                        with open(alog, "a", newline="") as fh:
                            w = csv.writer(fh)
                            w.writerow([time.time(), loss, len(buf["feats"]), backbone_name])
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
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)
    print(f"Agent {agent_id} ({backbone_name}) finished, wrote {out}")


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


def main():
    parser = argparse.ArgumentParser(description="Run local experiment with heterogeneous backbones")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--mode", choices=["synthetic", "video"], default="synthetic")
    parser.add_argument("--video", type=str, default=None, help="Path to video file when using video mode")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--stagger", type=float, default=0.2)
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
    args = parser.parse_args()

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

    manifest = {
        "timestamp": int(time.time()),
        "num_agents": num_agents,
        "mode": args.mode,
        "video": args.video,
        "max_frames": args.max_frames,
        "stagger": args.stagger,
        "hetero_embedding_mode": args.hetero,
        "embedding_dims": embedding_dims,
        "hetero_backbone_mode": args.hetero_backbone,
        "backbones": backbones,
        "prefer_peer_transfer": bool(args.prefer_peer_transfer),
        "use_pretrained_backbone": bool(args.use_pretrained_backbone),
        "similarity_threshold": cfg.similarity_threshold,
        "confidence_threshold": cfg.confidence_threshold,
    }
    _write_manifest(os.path.join(artifacts_dir, "run_manifest.json"), manifest)

    print("Run manifest:")
    print(json.dumps(manifest, indent=2))

    threads = []
    for i in range(num_agents):
        agent_id = f"agent_{i + 1}"
        t = threading.Thread(
            target=agent_thread,
            args=(
                agent_id,
                cfg,
                memory,
                args.mode == "video",
                args.video,
                args.max_frames,
                i * args.stagger,
                embedding_dims[i],
                backbones[i],
                bool(args.use_pretrained_backbone),
                bool(args.prefer_peer_transfer),
                metrics_dir,
            ),
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

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


if __name__ == "__main__":
    main()
