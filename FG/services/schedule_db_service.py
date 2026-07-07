import logging
import pandas as pd
import asyncio
from typing import List
from FG.models.schedule_model import UsecaseSchedule
from concurrent.futures import ThreadPoolExecutor
from FG.core.database import get_db_connection, session_pool
from FG.core.utils.log_and_progress import tracker_log_and_progress
from FG.core.query_loader import get_query_by_name
from celery.schedules import crontab

executor = ThreadPoolExecutor()


def insert_schedule_sync(schedule_data: dict):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        
        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO MIS_USER.USECASE_SCHEDULES (
                USE_CASE, SUBTASK_NAME, CRON_EXPR, OFFICE_ID, START_DATE, END_DATE,
                MONTH, YEAR, FILTER_TYPE,
                IS_ACTIVE, QUEUE_NAME, MODEL_NAME
            ) VALUES (
                :1, :2, :3, :4,
                TO_DATE(:5, 'YYYY-MM-DD'),
                TO_DATE(:6, 'YYYY-MM-DD'),
                :7, :8, :9,
                :10, :11, :12
            )
        """
        values = (
            schedule_data.get("use_case"),
            schedule_data.get("subtask_name"),
            schedule_data.get("cron_expr"),
            schedule_data.get("office_id"),
            schedule_data.get("start_date"),  # can be None
            schedule_data.get("end_date"),    # can be None
            schedule_data.get("month"),       # optional for monthwise
            schedule_data.get("year"),        # optional for monthwise
            schedule_data.get("filter_type"), # "day" or "month"
            schedule_data.get("is_active"),
            schedule_data.get("queue_name"),
            schedule_data.get("model_name"),
        )

        cursor.execute(insert_sql, values)
        conn.commit()
        logging.info(f"✅ Schedule inserted successfully for use case: {schedule_data.get('use_case')}")
        return {"message": "✅ Schedule inserted successfully"}

    except Exception as e:
        logging.error(f"🔥 Error inserting schedule for use case {schedule_data.get('use_case')}: {e}")
        return {"message": f"🔥 Error inserting schedule: {str(e)}"}

    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)
            
# 🔄 Async wrapper
async def insert_schedule(schedule_data: dict):
    loop = asyncio.get_event_loop()
    print(schedule_data)
    try:
        result = await loop.run_in_executor(executor, insert_schedule_sync, schedule_data)
        logging.info(f"✔️ Schedule inserted: {schedule_data.use_case}")
        return result
    except Exception as e:
        logging.error(f"❌ Error inserting schedule: {e}")
        return {"message": f"Error inserting schedule: {e}"}            


def parse_cron_expression(expr):
    try:
        parts = expr.strip().split()
        if len(parts) != 5:
            raise ValueError("Invalid cron expression format")
        return crontab(minute=parts[0], hour=parts[1], day_of_month=parts[2], month_of_year=parts[3], day_of_week=parts[4])
    except Exception as e:
        logging.error(f"❌ Invalid cron expression: {expr} - {e}")
        return None

# ✅ Load beat schedule dynamically from Oracle
def load_dynamic_schedules():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = get_query_by_name("SCHEDULE_TASKS")
        cursor.execute(query)
        rows = cursor.fetchall()
        schedule_dict = {}

        for row in rows:
            # Unpack row according to new schema
            (
                use_case, subtask_name, cron_expr, office_id, start_date, end_date,
                month, year, filter_type, queue_name, model_name
            ) = row

            task_name = f"schedule_dynamic_{use_case.lower()}"
            schedule = parse_cron_expression(cron_expr)
            print(filter_type)
            if schedule:
                # Determine args based on filter_type
                if filter_type == "day":
                    args = [use_case, office_id, str(start_date), str(end_date), subtask_name, model_name,filter_type]
                elif filter_type == "month":
                    args = [use_case, office_id, month, year, subtask_name, model_name,filter_type]
                else:
                    args = [use_case, office_id, month, year, subtask_name, model_name,filter_type]

                schedule_dict[task_name] = {
                    "task": "tasks.schedule_tasks.dynamic_schedule_trigger",
                    "schedule": schedule,
                    "options": {"queue": "schedule_queue"},
                    "args": args
                }

        logging.info(f"Loaded dynamic schedules: {list(schedule_dict.keys())}")
        return schedule_dict

    except Exception as e:
        logging.error(f"🔥 Failed loading beat schedule: {e}")
        return {}

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            session_pool.release(conn)
