"""AgentCommunication: MQTT-based publish/subscribe for KnowledgePackets.

If paho-mqtt is unavailable, the class runs in local-only mode and still
writes to CollectiveMemory so experiments remain reproducible.
"""
from __future__ import annotations

import logging
from typing import Optional

try:
    import paho.mqtt.client as mqtt
except Exception:  # optional dependency
    mqtt = None

from .knowledge_packet import KnowledgePacket

logger = logging.getLogger(__name__)


class AgentCommunication:
    def __init__(
        self,
        collective_memory,
        agent_id: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic: str = "swarm/knowledge",
        client_id: Optional[str] = None,
    ) -> None:
        self.memory = collective_memory
        self.agent_id = agent_id
        self.broker_host = broker_host
        self.broker_port = int(broker_port)
        self.topic = topic

        self.client = None
        if mqtt is not None:
            self.client = mqtt.Client(client_id=client_id)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message

        self.connected = False
        if self.client is None:
            logger.warning("paho-mqtt not installed; running in local-only mode")
        else:
            try:
                self.client.connect(self.broker_host, self.broker_port, keepalive=60)
                self.client.loop_start()
                self.connected = True
            except Exception:
                logger.warning(
                    "Could not connect to MQTT broker %s:%s - running in local-only mode",
                    self.broker_host,
                    self.broker_port,
                )

    def _on_connect(self, client, userdata, flags, rc):
        logger.info("Connected to MQTT broker %s:%s rc=%s", self.broker_host, self.broker_port, rc)
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            self.memory.import_packet_json(payload)
            logger.debug("Imported packet from topic %s", msg.topic)
        except Exception:
            logger.exception("Failed to handle incoming MQTT message")

    def publish_packet(self, packet: KnowledgePacket) -> None:
        """Publish a KnowledgePacket to the swarm topic (non-blocking)."""
        try:
            self.memory.add_packet(packet)
        except Exception:
            logger.exception("Failed to add packet to local CollectiveMemory")

        if not getattr(self, "connected", False):
            logger.debug("Running in local-only mode; skipping network publish")
            return

        try:
            j = packet.to_json()
            self.client.publish(self.topic, j, qos=0)
            logger.debug("Published packet (agent=%s, label=%s)", packet.agent_id, packet.object_label)
        except Exception:
            logger.exception("Failed to publish KnowledgePacket")

    def stop(self) -> None:
        try:
            if getattr(self, "connected", False):
                self.client.loop_stop()
                self.client.disconnect()
        except Exception:
            logger.exception("Failed to stop MQTT client cleanly")


__all__ = ["AgentCommunication"]
