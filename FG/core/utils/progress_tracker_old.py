# utils/progress_tracker.py

import redis
import json
import os
from core.utils.redis_client import redis_client


#redis_client = redis.from_url("redis://localhost:6379/0", decode_responses=True)

def update_progress(task_id: str, message: str, status: str = "in_progress"):
    
    print("Connecting to Redis at", os.getenv("REDIS_HOST", "localhost"), os.getenv("REDIS_PORT", 6379))
    print(f"progress:{task_id}")
    """
    Updates the progress of a background task in Redis.

    Args:
        task_id (str): Unique task identifier (usually Celery task ID).
        message (str): Status message to describe the current step.
        status (str): Optional status - 'in_progress', 'completed', 'failed'. Default is 'in_progress'.
        extra (dict): Optional additional data to store with the progress.
    """
    progress_data = {
        "status": status,
        "message": message
    }

    redis_client.set(f"progress:{task_id}", json.dumps(progress_data))
