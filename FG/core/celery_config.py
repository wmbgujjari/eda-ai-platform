from celery import Celery
from celery import current_app
from redbeat import RedBeatScheduler, RedBeatSchedulerEntry
import logging
from FG.services.schedule_db_service import load_dynamic_schedules
import os

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
REDBEAT_REDIS_URL = os.getenv("REDBEAT_REDIS_URL", "redis://localhost:6379/1")

celery = Celery(
    __name__,
    broker=BROKER_URL,
    backend=RESULT_BACKEND
)

celery.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json'],
    timezone='Asia/Kolkata',
    beat_scheduler='redbeat.RedBeatScheduler',
    redbeat_redis_url=REDBEAT_REDIS_URL,
    redbeat_lock_key='redbeat::lock',
    beat_max_loop_interval=30,
    task_routes={
        "tasks.consumption_tasks.*": {"queue": "consumption_queue"},
        "tasks.newconnection_daywise_tasks.*": {"queue": "newconnection_daywise_queue"},
        "tasks.schedule_tasks.*": {"queue": "schedule_queue"},
        "tasks.dynamic_tasks.*": {"queue": "dynamic_queue"},
        "tasks.segmentation_tasks.*": {"queue": "segmentation_queue"},
        "tasks.demand_tasks.*": {"queue": "demand_queue"},
        "tasks.revenue_tasks.*": {"queue": "revenue_queue"},
        "tasks.langchain_tasks.*": {"queue": "langchain_queue"},
        "tasks.daily_demand_tasks.*": {"queue": "daily_demand_queue"},
    },
)

celery.autodiscover_tasks([
    "tasks.consumption_tasks",
    "tasks.newconnection_daywise_tasks",
    "tasks.schedule_tasks",
    "tasks.dynamic_tasks",
    "tasks.segmentation_tasks",
    "tasks.demand_tasks",
    "tasks.revenue_tasks",
    "tasks.langchain_tasks",
    "tasks.daily_demand_tasks"
], force=True)

# ✅ Initial load from DB
try:
    celery.conf.beat_schedule = load_dynamic_schedules()
    logging.info("✅ Celery Beat schedules loaded successfully from Oracle.")
except Exception as e:
    logging.error(f"🔥 Failed to load Celery Beat schedules: {e}")
    celery.conf.beat_schedule = {}

# --------------------------------------------------------
# 🔁 Background Task: Refresh Beat schedules dynamically
# --------------------------------------------------------
@celery.task(name="FG.core.celery_config.refresh_schedules")
def refresh_beat_schedules():
    """Refresh Celery Beat schedules from DB and clean stale RedBeat entries (except locks)."""
    try:
        print("🔄 Refreshing Celery Beat schedules from DB...")
        new_schedules = load_dynamic_schedules()
        current_names = set(new_schedules.keys())

        added, updated, removed = 0, 0, 0

        import redis
        redis_conn = redis.Redis.from_url(celery.conf.redbeat_redis_url)

        # --- Add or update valid schedules ---
        for name, sched in new_schedules.items():
            try:
                entry = RedBeatSchedulerEntry(
                    name=name,
                    task=sched["task"],
                    schedule=sched["schedule"],
                    args=sched.get("args", []),
                    kwargs=sched.get("kwargs", {}),
                    options=sched.get("options", {}),
                    app=celery,
                )

                # Check directly in Redis
                if redis_conn.exists(entry.key):
                    updated += 1
                    logging.info(f"🌀 Updating existing schedule: {name}")
                else:
                    added += 1
                    logging.info(f"🆕 Adding new schedule: {name}")

                entry.save()

            except Exception as e:
                logging.error(f"⚠️ Failed updating schedule {name}: {e}")

        # --- Remove stale RedBeat entries except locks ---
        for key in redis_conn.scan_iter(match="redbeat:*"):
            key_decoded = key.decode()

            # Skip active or lock keys
            if any(skip in key_decoded for skip in ["lock", "schedule"]):
                continue

            # Extract schedule name
            key_name = key_decoded.split(":")[-1]
            if key_name not in current_names:
                redis_conn.delete(key)
                removed += 1
                logging.info(f"🗑 Removed stale RedBeat entry: {key_decoded}")

        logging.info(
            f"✅ Refreshed schedules from DB — Added: {added}, Updated: {updated}, Removed: {removed}"
        )
        print(
            f"✅ Refreshed schedules from DB — Added: {added}, Updated: {updated}, Removed: {removed}"
        )

    except Exception as e:
        logging.error(f"🔥 Failed refreshing beat schedules: {e}")

# --------------------------------------------------------
# ⏰ Schedule: Auto-refresh every 5 minutes
# --------------------------------------------------------
celery.conf.beat_schedule["refresh_dynamic_schedules"] = {
    "task": "FG.core.celery_config.refresh_schedules",
    "schedule": 100.0,  # Every 5 minutes
}
