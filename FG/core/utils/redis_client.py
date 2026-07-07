import redis
import os
from urllib.parse import urlparse

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
parsed = urlparse(redis_url)

redis_client = redis.Redis(
    host=parsed.hostname,
    port=parsed.port,
    db=int(parsed.path.lstrip("/") or 0),
    decode_responses=True
)
