# utils/progress_tracker.py

from datetime import datetime
from core.utils.progress_tracker import update_progress
from core.utils.audit_logging_utils import insert_audit_log

def tracker_log_and_progress(task_id: str, message: str, status: str = "INFO"):
    """
    Updates progress in Redis and logs to the audit table.

    :param task_id: Unique identifier for the background task.
    :param message: Status message to log and display.
    :param status: Type of log (e.g., INFO, ERROR, WARNING).
    """
    # Update Redis progress
    update_progress(task_id, message)

    # Insert into database audit log
    #insert_audit_log(task_id=task_id, message=message, status=status)

    # Optional: Console log for visibility
    print(f"[{datetime.now()}] [{status}] [{task_id}] {message}")
