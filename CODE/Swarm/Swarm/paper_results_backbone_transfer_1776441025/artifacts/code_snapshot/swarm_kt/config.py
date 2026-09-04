"""Configuration utilities for the swarm knowledge-transfer system.

Provides a simple programmatic configuration with defaults and optional
override via a JSON config file placed at the repository root named
`swarm_config.json` or via environment variables.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl: int = 30

    # Similarity logic
    similarity_threshold: float = 0.85
    confidence_threshold: float = 0.80
    max_scan: int = 128

    # MQTT
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_topic: str = "swarm/knowledge"

    # Embedding
    embedding_dim: int = 256

    # Simulation tuning (ms saved per heavy inference)
    heavy_inference_ms: int = 120
    # Consensus/trust settings
    consensus_k: int = 1
    consensus_accept_weight: float = 0.9

    # Model provenance (per-agent declared capacity)
    model_name: str = "local_detector"
    model_version: str = "0.1"
    model_capacity: float = 0.5

    # Online adaptation settings
    adapt_batch_size: int = 8
    adapt_epochs: int = 3
    adapt_lr: float = 1e-3

    @staticmethod
    def load(path: Optional[str] = None) -> "Config":
        # Order: explicit path -> env var SWARM_CONFIG_PATH -> default file -> defaults
        cfg_path = path or os.environ.get("SWARM_CONFIG_PATH") or "swarm_config.json"
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as fh:
                raw = json.load(fh)
            return Config(**{k: raw.get(k, v) for k, v in Config().__dict__.items()})
        else:
            return Config()


__all__ = ["Config"]
