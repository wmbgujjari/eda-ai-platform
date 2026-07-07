import json
from datetime import datetime
from FG.core.utils.redis_client import redis_client


def update_progress(
    task_id: str,
    message: str,
    status: str = "in_progress",
    section_id: str = None,
    progress_percent: float = None,
    meta: dict = None,
    publish: bool = True
):
    """
    Updates real-time progress in Redis and optionally publishes via Pub/Sub.
    """

    progress_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "task_id": task_id,
        "section_id": section_id,
        "status": status,
        "message": message,
        "progress_percent": progress_percent,
        "meta": meta or {},
    }

    # 🔹 Store latest snapshot (for UI polling)
    redis_client.set(f"progress:{task_id}", json.dumps(progress_data))

    # 🔹 Publish live event (for LangChain or WebSocket subscribers)
    if publish:
        redis_client.publish("task_progress_channel", json.dumps(progress_data))

    print(f"[{datetime.now()}] [{status}] [{task_id}] {message}")
