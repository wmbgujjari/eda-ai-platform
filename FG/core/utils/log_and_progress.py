from datetime import datetime
from FG.core.utils.progress_tracker import update_progress
# from core.utils.audit_logging_utils import insert_audit_log  # optional DB logging


def tracker_log_and_progress(
    task_id: str,
    message: str,
    status: str = "INFO",
    section_id: str = None,
    progress_percent: float = None,
    meta: dict = None,
    publish: bool = True
):
    """
    Unified live progress + structured audit logging.
    Integrates with Redis (for UI/monitoring) and LangChain (for reasoning).
    """

    # --- Normalize status ---
    normalized_status = status.lower()

    # --- Update live progress (Redis + LangChain Insight) ---
    update_progress(
        task_id=task_id,
        message=message,
        status=normalized_status,
        section_id=section_id,
        progress_percent=progress_percent,
        meta=meta,
        publish=publish,
    )

    # --- Optional DB logging (for audit history) ---
    # insert_audit_log(task_id=task_id, message=message, status=status, meta=meta)

    # --- Console output ---
    print(f"[{datetime.now()}] [{status.upper()}] [{task_id}] {message}")
