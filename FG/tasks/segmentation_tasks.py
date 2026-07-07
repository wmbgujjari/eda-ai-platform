import asyncio
import logging
from typing import List
from celery import shared_task
from FG.core.utils.log_and_progress import tracker_log_and_progress
from FG.services.train_segmentation import load_data, run_segmentation_pipeline
from FG.core.database import get_db_connection, session_pool
from celery import Celery
import requests
logger = logging.getLogger(__name__)

celery = Celery(
    "FG.tasks.segmentation_tasks",
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json']
)


@shared_task(name="FG.tasks.segmentation_tasks.schedule_train_segmentation")
def schedule_train_segmentation(start_date: str, end_date: str, office_id: str,model_name: str):
    logger.info("✅ Beat triggered schedule_train_newconnection")
    # 🔁 You can dynamically calculate date range here
    #end_date = "2024-12-31"#datetime.now().strftime("%Y-%m-%d")
    #start_date = "2024-08-01"#(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")  # last 30 days
    #office_id = "803-10"  # Replace with dynamic logic or loop if needed

    url = f"http://127.0.0.1:8000/consumption/train/{start_date}/{end_date}/{office_id}/{model_name}"

    try:
        response = requests.post(url)
        print(f"Triggered training: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error triggering training: {e}")



@shared_task
def fetch_and_train_segmentation(section_ids: List[str], task_id: str, start_date: str, end_date: str, model_name: str):
    """
    Centralized Segmentation Task:
    1. Fetch data
    2. Run segmentation (rule-based + clustering + hybrid)
    3. Queue insertion tasks for consumer segments & segment forecasts
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 1. Fetch data
        df, file_size = loop.run_until_complete(
            load_data(section_ids, task_id, start_date, end_date)
        )
        logging.info("📥 Data fetched for segmentation: %s rows", len(df))
        if df.empty:
            logging.warning("⚠️ No data fetched for section(s): %s", section_ids)
            return "No data found for segmentation."

        tracker_log_and_progress(task_id, f"🚀 Starting segmentation for sections: {section_ids}")

        # 2. Run segmentation pipeline
        consumer_df = run_segmentation_pipeline(df, file_size, task_id, model_name)

        logging.info("✅ Segmentation completed. Starting insertion...")
        tracker_log_and_progress(task_id, "✅ Segmentation completed. Starting insertion...")

        # 3. Queue insertion of consumer-level segments
        if not consumer_df.empty:
            for section_id in section_ids:
                records = consumer_df[consumer_df["SECTION_ID"] == section_id].to_dict(orient="records")
                if records:
                    insert_consumer_segments.apply_async(
                        args=[section_id, records, task_id],
                        queue="segmentation_queue"
                    )
                    logging.info(f"📤 Queued consumer segments for section: {section_id}, records: {len(records)}")
                    tracker_log_and_progress(task_id, f"📤 Queued consumer segments for section: {section_id}, records: {len(records)}")
        tracker_log_and_progress(task_id, f"🎯 Segmentation training + insertion completed for sections {section_ids}", "completed")
        return f"Segmentation done. Insertion started for sections: {section_ids}"

    except Exception as e:
        logging.error(f"🔥 Error in segmentation for sections {section_ids}: {e}")
        tracker_log_and_progress(task_id, f"🔥 Error in segmentation for sections {section_ids}: {e}", "failed")
        return f"Error: {str(e)}"

    finally:
        loop.close()

@shared_task
def insert_consumer_segments(section_id: str, records: list, task_id: str):
    """
    Insert consumer segmentation results into DB
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            logging.error("❌ Could not get database connection")
            return "DB connection failed"

        cursor = conn.cursor()
        if not records:
            logging.warning(f"⚠️ No records to insert for section {section_id}")
            return "No records to insert."

        logging.info(f"📥 Preparing {len(records)} records for section {section_id}")

        insert_query = """
            INSERT INTO mis_user.consumer_segments (
                consumer_id,
                office_id,
                cat_code,
                seg_rule,
                kmean_cluster_id,
                cluster_label,
                seg_hybrid,
                bill_units,
                billmonth,
                billyear,
                amount_collected,
                arrears,
                subsidy                
            )
            VALUES (
                :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13
            )
        """

        # Prepare tuples from records
        data_tuples = [
            (
                rec["CONSUMER_ID"],
                rec["SECTION_ID"],
                rec["CAT_CODE"],
                rec["SEG_RULE"],
                rec["CLUSTER_ID"],
                rec["CLUSTER_LABEL"],
                rec["SEG_HYBRID"],
                rec["BILL_UNITS"],
                rec["BILL_MONTH"],
                rec["BILL_YEAR"],
                rec["AMOUNT_COLLECTED"],
                rec["ARREARS"],
                rec["SUBSIDY"]                
            )
            for rec in records
        ]

        cursor.executemany(insert_query, data_tuples)
        conn.commit()

        logging.info(f"✅ Inserted {cursor.rowcount} records into consumer_segments")
        return f"Inserted {cursor.rowcount} records."

    except Exception as e:
        if conn:
            conn.rollback()
        logging.error(f"❌ Error while inserting consumer_segments: {e}")
        return f"Error: {e}"

    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)