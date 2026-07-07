from pydantic import BaseModel
from typing import Optional

class UsecaseSchedule(BaseModel):
    use_case: str
    subtask_name: str
    cron_expr: str
    office_id: str
    filter_type: str                 # "day" or "month"
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    month: Optional[int] = None       # 1-12
    year: Optional[int] = None
    is_active: str
    queue_name: str
    model_name: str