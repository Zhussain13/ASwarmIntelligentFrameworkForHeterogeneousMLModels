"""CollectiveMemory: Redis-backed memory for knowledge packets.

This module implements a portable Redis-based store that keeps recent
KnowledgePacket entries with TTL and provides a redundancy check using
cosine similarity.

Design choices (portable):
- Use simple Redis primitives (SET with EX, LPUSH + LTRIM for recent keys,
  and MGET for batched reads).
- Bounded scan (configurable `max_scan`) to keep checks fast on edge devices.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import List, Optional

import numpy as np
import redis

from .knowledge_packet import KnowledgePacket


logger = logging.getLogger(__name__)


class CollectiveMemory:
    """Redis-backed collective memory with redundancy checking.

    Parameters
    - redis_url: Redis connection URL
    - ttl: time-to-live for each stored packet (seconds)
    - similarity_threshold: cosine similarity threshold for redundancy
    - confidence_threshold: minimum stored confidence to be trusted
    - max_scan: maximum number of recent packets to compare against
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl: int = 30,
        similarity_threshold: float = 0.85,
        confidence_threshold: float = 0.80,
        max_scan: int = 128,
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        # Accept an optional redis client for testability (e.g., fakeredis).
        if redis_client is not None:
            self.redis = redis_client
        else:
            self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl = int(ttl)
        self.similarity_threshold = float(similarity_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.max_scan = int(max_scan)
        self.recent_list_key = "swarm:recent"  # holds keys to recent packet keys

    def _make_key(self) -> str:
        return f"swarm:packet:{uuid.uuid4().hex}"

    def add_packet(self, packet: KnowledgePacket) -> str:
        """Store a KnowledgePacket in Redis and push its key to recent list.

        Returns the Redis key used.
        """
        key = self._make_key()
        j = packet.to_json()
        pipe = self.redis.pipeline()
        pipe.set(key, j, ex=self.ttl)
        # store the key name in the recent list for quick lookup
        pipe.lpush(self.recent_list_key, key)
        pipe.ltrim(self.recent_list_key, 0, self.max_scan - 1)
        pipe.execute()
        logger.debug("Added packet %s (agent=%s, label=%s)", key, packet.agent_id, packet.object_label)
        # Update trust information from packet if provided (initial trust)
        try:
            if packet.model_capacity is not None:
                # store as a float in a Redis hash for trust scores
                self.redis.hset("swarm:trust", packet.agent_id, float(packet.model_capacity))
        except Exception:
            logger.debug("Could not update trust score for agent %s", packet.agent_id)
        return key

    def import_packet_json(self, json_str: str) -> str:
        """Decode JSON into KnowledgePacket and add to memory."""
        pkt = KnowledgePacket.from_json(json_str)
        return self.add_packet(pkt)

    def get_recent_keys(self, n: Optional[int] = None) -> List[str]:
        """Return up to n recent keys (default: max_scan)."""
        if n is None:
            n = self.max_scan
        return self.redis.lrange(self.recent_list_key, 0, max(0, n - 1))

    def get_packets_by_keys(self, keys: List[str]) -> List[KnowledgePacket]:
        """Retrieve packets for provided keys, ignoring missing/malformed entries."""
        if not keys:
            return []
        vals = self.redis.mget(keys)
        pkts: List[KnowledgePacket] = []
        for v in vals:
            if not v:
                continue
            try:
                pkt = KnowledgePacket.from_json(v)
                pkts.append(pkt)
            except Exception:
                logger.exception("Failed to parse packet from Redis value")
        return pkts

    # Trust / reputation helpers
    def get_trust(self, agent_id: str) -> float:
        """Return agent trust score in [0,1]. Defaults to 0.5 if unknown."""
        try:
            v = self.redis.hget("swarm:trust", agent_id)
            if v is None:
                return 0.5
            return float(v)
        except Exception:
            return 0.5

    def update_trust(self, agent_id: str, delta: float) -> float:
        """Atomically adjust an agent's trust score by delta and return new value.

        Trust is clamped to [0.0, 1.0].
        """
        try:
            pipe = self.redis.pipeline()
            pipe.hget("swarm:trust", agent_id)
            old = pipe.execute()[0]
            if old is None:
                new = 0.5 + float(delta)
            else:
                new = float(old) + float(delta)
            new = max(0.0, min(1.0, new))
            self.redis.hset("swarm:trust", agent_id, new)
            return new
        except Exception:
            logger.exception("Failed to update trust for %s", agent_id)
            return 0.5

    def consensus_match(self, current_embedding: np.ndarray, k: int = 1, accept_weight: float = 0.9):
        """Aggregate recent packets and compute a weighted consensus for labels.

        Returns (accepted: bool, info: dict) where info contains:
          - label: chosen label (if accepted)
          - aggregated_score: weighted sum of confidences
          - distinct_agents: number of distinct agents contributing
          - matches: list of (agent_id, sim, confidence, trust)

        This performs the same bounded scan as `is_redundant` but aggregates by label.
        """
        cur = np.asarray(current_embedding, dtype=np.float32).ravel()
        if cur.size == 0:
            return False, None

        cur_norm = np.linalg.norm(cur)
        if cur_norm == 0:
            return False, None
        cur = cur / cur_norm

        keys = self.get_recent_keys(self.max_scan)
        if not keys:
            return False, None

        vals = self.redis.mget(keys)
        clusters = {}
        matches = []
        for v in vals:
            if not v:
                continue
            try:
                pkt = KnowledgePacket.from_json(v)
            except Exception:
                continue

            emb = getattr(pkt, "embedding", None)
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32).ravel()
            if emb.size != cur.size:
                continue
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            embn = emb / emb_norm
            sim = float(np.dot(cur, embn))
            if sim < self.similarity_threshold:
                continue
            trust = self.get_trust(pkt.agent_id)
            weight = trust * float(pkt.confidence)
            matches.append({
                "agent_id": pkt.agent_id,
                "label": pkt.object_label,
                "sim": sim,
                "confidence": float(pkt.confidence),
                "trust": trust,
                "weight": weight,
            })
            clusters.setdefault(pkt.object_label, []).append(matches[-1])

        if not clusters:
            return False, None

        # compute aggregated scores per label
        agg = []
        for label, items in clusters.items():
            agg_score = sum(it["weight"] for it in items)
            distinct_agents = len({it["agent_id"] for it in items})
            agg.append((label, agg_score, distinct_agents, items))

        # choose best label by aggregated_score
        agg.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score, best_distinct, best_items = agg[0]

        accepted = (best_score >= accept_weight) and (best_distinct >= k)
        info = {
            "label": best_label,
            "aggregated_score": float(best_score),
            "distinct_agents": int(best_distinct),
            "matches": best_items,
        }
        return accepted, info

    def is_redundant(self, current_embedding: np.ndarray):
        """Check whether current_embedding matches a trusted recent packet.

        Returns a tuple (is_redundant: bool, match_info: Optional[dict]).
        match_info contains keys: 'sim', 'confidence', 'label' when a match is found.

        If no match is found returns (False, None).

        Matching logic:
        - L2-normalize both vectors
        - Compare cosine similarity
        - If any stored packet has similarity > similarity_threshold AND
          stored_confidence > confidence_threshold, return True

        Notes: This performs a bounded scan of recent elements (max_scan) for
        portability. For large deployments, replace with Redis vector index.
        """
        cur = np.asarray(current_embedding, dtype=np.float32).ravel()
        if cur.size == 0:
            return False, None

        cur_norm = np.linalg.norm(cur)
        if cur_norm == 0:
            return False, None
        cur = cur / cur_norm

        keys = self.get_recent_keys(self.max_scan)
        if not keys:
            return False, None

        # Batched fetch
        vals = self.redis.mget(keys)
        for v in vals:
            if not v:
                continue
            try:
                pkt = KnowledgePacket.from_json(v)
            except Exception:
                # ignore malformed entries
                continue

            emb = getattr(pkt, "embedding", None)
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32).ravel()
            if emb.size != cur.size:
                # embedding dim mismatch — skip this entry
                continue
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            embn = emb / emb_norm
            sim = float(np.dot(cur, embn))
            logger.debug("Comparing sim=%.4f (stored_conf=%.3f)", sim, pkt.confidence)
            if sim >= self.similarity_threshold and pkt.confidence >= self.confidence_threshold:
                logger.info(
                    "Redundant match found: sim=%.3f, stored_conf=%.3f, label=%s",
                    sim,
                    pkt.confidence,
                    pkt.object_label,
                )
                return True, {"sim": sim, "confidence": float(pkt.confidence), "label": pkt.object_label}

        return False, None

    def get_best_match(self, current_embedding: np.ndarray, exclude_agent_id: Optional[str] = None):
        """Return the best matching KnowledgePacket and similarity score.

        Returns (packet: KnowledgePacket, sim: float) or (None, 0.0) if no
        suitable match found (respecting similarity and confidence
        thresholds).
        """
        cur = np.asarray(current_embedding, dtype=np.float32).ravel()
        if cur.size == 0:
            return None, 0.0

        cur_norm = np.linalg.norm(cur)
        if cur_norm == 0:
            return None, 0.0
        cur = cur / cur_norm

        keys = self.get_recent_keys(self.max_scan)
        if not keys:
            return None, 0.0

        vals = self.redis.mget(keys)
        best_pkt = None
        best_sim = 0.0
        for v in vals:
            if not v:
                continue
            try:
                pkt = KnowledgePacket.from_json(v)
            except Exception:
                continue

            if exclude_agent_id is not None and pkt.agent_id == exclude_agent_id:
                continue

            emb = getattr(pkt, "embedding", None)
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32).ravel()
            if emb.size != cur.size:
                continue
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            embn = emb / emb_norm
            sim = float(np.dot(cur, embn))
            if sim >= self.similarity_threshold and pkt.confidence >= self.confidence_threshold:
                if sim > best_sim:
                    best_sim = sim
                    best_pkt = pkt

        if best_pkt is None:
            return None, 0.0
        return best_pkt, float(best_sim)

    def count_recent(self) -> int:
        return self.redis.llen(self.recent_list_key)


__all__ = ["CollectiveMemory"]
