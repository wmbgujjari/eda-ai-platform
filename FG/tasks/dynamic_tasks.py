
from FG.services.dynamic_data_service import execute_dynamic_query
from FG.core.config import THRESHOLD_MB # Config constants
import pandas as pd
from FG.core.database import get_db_connection
from celery import shared_task
from celery import shared_task
from FG.core.database import get_db_connection
from FG.services.train_dynamic import run_automl_training
from FG.core.utils.log_and_progress import tracker_log_and_progress
import logging
from FG.core.database import get_db_connection, session_pool
from typing import List
from celery import Celery
from datetime import datetime


celery = Celery(
    "FG.tasks.dynamic_tasks",
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json']
)

@shared_task
def insert_section_data(section_id: str, records: List[dict],task_id: str):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        cursor = conn.cursor()
        insert_sql = """
            INSERT INTO MIS_USER.NSC_DAYWISE_PREDICTIONS (
                SECTION_ID, DAY, PREDICTED_VALUE, MODEL_VERSION, TRAINING_DATE
            )
            VALUES (:1, :2, :3, :4, :5)
        """

        data_tuples = [
            (
                rec["SECTION_ID"],
                datetime.strptime(rec["DATE"], "%Y-%m-%d").date() if isinstance(rec["DATE"], str) else rec["DATE"],
                rec["PREDICTED_CONNECTION_COUNT"],
                rec["MODEL_VERSION"],
                datetime.strptime(rec["TRAINING_DATE"], "%Y-%m-%d").date() if isinstance(rec["TRAINING_DATE"], str) else rec["TRAINING_DATE"],
            )
            for rec in records
        ]

        cursor.executemany(insert_sql, data_tuples)
        conn.commit()
        logging.info(f"✅ Inserted predicted data for section {section_id} successfully.")
        tracker_log_and_progress(task_id, f"✅ Inserted predicted data for section {section_id} successfully.")

    except Exception as e:
        logging.error(f"🔥 Error inserting predicted data for section {section_id}: {e}")
        tracker_log_and_progress(task_id, f"🔥 Error inserting predicted data for section {section_id}: {e}","failed")
    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)

@shared_task
def insert_accuracy_log(section_id, model_version, training_date, trained_with, r2, rmse,task_id):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        cursor = conn.cursor()
        training_date = datetime.strptime(training_date, "%Y-%m-%d").date()
        insert_sql = """
            INSERT INTO MIS_USER.MODEL_TRAINING_LOGS (
                SECTION_ID, MODEL_VERSION, TRAINING_DATE, TRAINED_WITH, R2_SCORE, RMSE
            )
            VALUES (:1, :2, TO_DATE(:3, 'YYYY-MM-DD'), :4, :5, :6)
        """

        cursor.execute(insert_sql, (section_id, model_version, training_date, trained_with, r2, rmse))
        conn.commit()
        logging.info(f"📘 Logged model training details for section {section_id}")
        tracker_log_and_progress(task_id, f"📘 Logged model training details for section {section_id}")
    except Exception as e:
        logging.error(f"❌ Failed to log model info for section {section_id}: {e}")
        tracker_log_and_progress(task_id, f"❌ Failed to log model info for section {section_id}: {e}","failed")
    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)



@shared_task
def fetch_data_and_train(section_ids: List[str],task_id: str, query: str, feature_columns: list, target_column: str):
    try:
        tracker_log_and_progress(task_id, "🚀 Dynamic training pipeline started...")

        # Load data from the query
        df = execute_dynamic_query(query)

        if df.empty:
            tracker_log_and_progress(task_id, "⚠️ No data found for the provided query", "failed")
            return "No data for training."

        # Train model (AutoML or selected model)
        section_data_map, section_accuracy_map = run_automl_training(
            df, feature_columns, target_column, task_id
        )

        tracker_log_and_progress(task_id, "✅ Training complete. Preparing DB insertions...")

        # Insert predictions
        for section_id, records in section_data_map.items():
            insert_section_data.apply_async(
                args=[section_id, records, task_id],
                queue="dynamic_queue"  # Replace or reuse queue
            )
            tracker_log_and_progress(task_id, f"📤 Insert queued for section: {section_id}")

        # Insert accuracy logs
        for section_id, metrics in section_accuracy_map.items():
            insert_accuracy_log.apply_async(
                kwargs={
                    "section_id": section_id,
                    "model_version": metrics["MODEL_VERSION"],
                    "training_date": metrics["TRAINING_DATE"],
                    "trained_with": metrics["TRAINED_WITH"],
                    "r2": metrics["R2_SCORE"],
                    "rmse": metrics["RMSE"],
                    "task_id": task_id
                },
                queue="dynamic_queue"
            )
            tracker_log_and_progress(task_id, f"📝 Accuracy log queued for section: {section_id}")

        tracker_log_and_progress(task_id, "🎯 All dynamic training tasks completed.", "completed")
        return f"Dynamic training completed. Task ID: {task_id}"

    except Exception as e:
        logging.error(f"🔥 Error in dynamic_train_pipeline: {str(e)}")
        tracker_log_and_progress(task_id, f"🔥 Error during training: {str(e)}", "failed")
        return f"Error: {str(e)}"