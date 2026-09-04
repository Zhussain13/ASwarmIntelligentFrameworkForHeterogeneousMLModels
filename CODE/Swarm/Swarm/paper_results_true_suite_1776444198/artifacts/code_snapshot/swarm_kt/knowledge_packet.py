"""KnowledgePacket data structure

This module defines the KnowledgePacket dataclass used to share detections
between heterogeneous agents. The embedding is serialized in a model-agnostic
way (base64-encoded float32 bytes + shape/dtype metadata) so different
agents can reconstruct and compare vectors reliably.

Academic notes:
- Embeddings should be produced with a shared lightweight encoder (e.g., a
  frozen ResNet18 penultimate layer) and L2-normalized to ensure cross-model
  comparability.
- Embedding size should be fixed (e.g., 256 or 512) across agents.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import json
import base64
from typing import Any, Dict, Optional

import numpy as np


def _encode_embedding(arr: np.ndarray) -> Dict[str, Any]:
    """Encode numpy array embedding to a JSON-serializable dict.

    We use float32 bytes + base64 for compact, lossless cross-language
    transport and to preserve dtype/shape.
    """
    a = np.asarray(arr, dtype=np.float32)
    b = a.tobytes()
    return {
        "b64": base64.b64encode(b).decode("ascii"),
        "dtype": str(a.dtype),
        "shape": a.shape,
    }


def _decode_embedding(payload: Dict[str, Any]) -> np.ndarray:
    """Decode embedding dict back into numpy array."""
    b64 = payload.get("b64")
    if b64 is None:
        # Fallback: maybe embedding is stored as a list
        lst = payload.get("list")
        if lst is None:
            raise ValueError("No embedding found in payload")
        return np.asarray(lst, dtype=np.float32)

    raw = base64.b64decode(b64.encode("ascii"))
    dtype = np.dtype(payload.get("dtype", "float32"))
    shape = tuple(payload.get("shape", (-1,)))
    arr = np.frombuffer(raw, dtype=dtype)
    try:
        arr = arr.reshape(shape)
    except Exception:
        # if shape doesn't match, just return flattened
        arr = arr.ravel()
    return arr


@dataclass
class KnowledgePacket:
    """A packet of knowledge describing a detection from one agent.

    Fields:
    - agent_id: unique string identifier of the sender
    - timestamp: float epoch seconds
    - object_label: semantic label (e.g., "Person")
    - confidence: float in [0, 1]
    - embedding: numpy array (1D) representing visual features

    Serialization uses JSON with embedding encoded as base64 bytes. This
    representation is model-agnostic if all agents use the same encoder
    architecture and preprocessing pipeline for embeddings.
    """

    agent_id: str
    timestamp: float
    object_label: str
    confidence: float
    embedding: np.ndarray
    # provenance / model metadata (optional)
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    # relative model capability score in [0,1] (e.g., measured AP or benchmarked)
    model_capacity: Optional[float] = None
    # optional calibration metadata (e.g., temperature)
    calibration: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Basic validation and normalization
        if not isinstance(self.agent_id, str):
            raise TypeError("agent_id must be a string")
        if not isinstance(self.timestamp, (float, int)):
            raise TypeError("timestamp must be a float (seconds since epoch)")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

        # Ensure embedding is a 1-D float32 numpy array
        arr = np.asarray(self.embedding)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        self.embedding = arr.astype(np.float32)

    def to_dict(self, include_embedding_as_list: bool = False) -> Dict[str, Any]:
        """Create a JSON-friendly dict representation.

        By default embeddings are encoded as base64 float32 bytes. If
        include_embedding_as_list is True, a fallback "list" representation
        is also added (useful for debugging or tiny embeddings).
        """
        d = {
            "agent_id": self.agent_id,
            "timestamp": float(self.timestamp),
            "object_label": self.object_label,
            "confidence": float(self.confidence),
            "embedding": _encode_embedding(self.embedding),
        }
        # optional provenance fields
        if self.model_name is not None:
            d["model_name"] = self.model_name
        if self.model_version is not None:
            d["model_version"] = self.model_version
        if self.model_capacity is not None:
            d["model_capacity"] = float(self.model_capacity)
        if self.calibration is not None:
            d["calibration"] = self.calibration
        if include_embedding_as_list:
            d["embedding"]["list"] = self.embedding.tolist()
        return d

    def to_json(self, **json_kwargs) -> str:
        """Serialize the packet to a JSON string."""
        return json.dumps(self.to_dict(), **json_kwargs)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "KnowledgePacket":
        """Reconstruct a KnowledgePacket from a dict (as produced by to_dict)."""
        emb_payload = data.get("embedding")
        if emb_payload is None:
            raise ValueError("embedding missing in data")
        embedding = _decode_embedding(emb_payload)
        return KnowledgePacket(
            agent_id=str(data["agent_id"]),
            timestamp=float(data["timestamp"]),
            object_label=str(data["object_label"]),
            confidence=float(data["confidence"]),
            embedding=embedding,
            model_name=data.get("model_name"),
            model_version=data.get("model_version"),
            model_capacity=data.get("model_capacity"),
            calibration=data.get("calibration"),
        )

    @staticmethod
    def from_json(j: str) -> "KnowledgePacket":
        d = json.loads(j)
        return KnowledgePacket.from_dict(d)

    def age_seconds(self) -> float:
        """Return age of packet in seconds from creation timestamp to now."""
        return time.time() - float(self.timestamp)

    def __repr__(self) -> str:  # concise for logs
        return (
            f"KnowledgePacket(agent={self.agent_id!r}, label={self.object_label!r}, "
            f"conf={self.confidence:.3f}, ts={self.timestamp:.3f}, "
            f"emb_len={len(self.embedding)})"
        )
