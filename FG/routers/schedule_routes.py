from fastapi import APIRouter, Query, Body
from FG.models.schedule_model import UsecaseSchedule
from FG.services.schedule_db_service import insert_schedule
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schedule", tags=["Schedules"])


@router.post("/insert-schedule")
async def insert_usecase_schedule(
    use_case: str = Query(None),
    subtask_name: str = Query(None),
    cron_expr: str = Query(None),
    office_id: str = Query(None),
    filter_type: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
    month: int = Query(None),
    year: int = Query(None),
    is_active: str = Query("Y"),
    queue_name: str = Query(None),
    model_name: str = Query(None),
    body: dict = Body(None)
):
    """
    Insert schedule. Supports both daywise (start_date/end_date) 
    and monthwise (month/year) filters.
    """
    if body:
        schedule = body
    else:
        schedule = {
            "use_case": use_case,
            "subtask_name": subtask_name,
            "cron_expr": cron_expr,
            "office_id": office_id,
            "filter_type": filter_type,
            "is_active": is_active,
            "queue_name": queue_name,
            "model_name": model_name
        }

        # 🔹 Validate and apply filter based on filter_type
        if filter_type == "day":
            if not start_date or not end_date:
                return {"error": "start_date and end_date are required for daywise schedules."}
            schedule["start_date"] = start_date
            schedule["end_date"] = end_date
            schedule["month"] = None
            schedule["year"] = None
        elif filter_type == "month":
            if not month or not year:
                return {"error": "month and year are required for monthwise schedules."}
            schedule["month"] = month
            schedule["year"] = year
            schedule["start_date"] = None
            schedule["end_date"] = None
        else:
            return {"error": "filter_type must be either 'day' or 'month'."}

    logger.info(f"Inserting schedule: {schedule}")
    result = await insert_schedule(schedule)
    return {"message": "Inserted successfully", "details": result}