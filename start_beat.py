import os
import subprocess

# Set PYTHONPATH
os.environ["PYTHONPATH"] = r"D:\EDAFrameWork\EDAProject"

# Optional: Print confirmation
print("🔧 PYTHONPATH set to:", os.environ["PYTHONPATH"])

# Remove old celerybeat-schedule file if exists
SCHEDULE_PATH = r"D:\EDAFrameWork\EDAProject\celerybeat-schedule"
if os.path.exists(SCHEDULE_PATH):
    print("🧹 Removing existing Celery Beat schedule file...")
    os.remove(SCHEDULE_PATH)

# Start Celery Beat
print("🚀 Starting Celery Beat...")
subprocess.call([
    "celery", "-A", "core.celery_config", "beat", "--loglevel=info"
])
