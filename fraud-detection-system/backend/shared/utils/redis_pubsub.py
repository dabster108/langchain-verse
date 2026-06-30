from typing import Any

from redis import Redis

from shared.utils.serialization import to_json


def publish_event(redis_client: Redis, channel: str, payload: dict[str, Any]) -> int:
    return redis_client.publish(channel, to_json(payload))
