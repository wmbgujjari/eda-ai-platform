import logging
import pandas as pd
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from FG.core.database import get_db_connection, session_pool
#from core.database_uat import get_uat_db_connection, uat_session_pool
from FG.core.utils.log_and_progress import tracker_log_and_progress
from FG.core.utils.common_db_service import fetch_data_sync
from FG.core.query_loader import get_query_by_name

executor = ThreadPoolExecutor()

async def fetch_segmentation_data(section_ids: list[str], task_id: str, start_date: str, end_date: str):
    #queries = load_queries()
    base_query = get_query_by_name("SEGMENTATION_QUERY")

    # Format section IDs as SQL string list
    formatted_section_ids = ', '.join(f"'{sid}'" for sid in section_ids)

    # Inject parameters
    query = base_query.format(
        section_ids=formatted_section_ids,
        start_date=start_date,
        end_date=end_date
    )
    print(query)
    tracker_log_and_progress(task_id, "✅ Query Execution Started for New Connection Data")

    loop = asyncio.get_event_loop()
    try:
        df = await loop.run_in_executor(executor, fetch_data_sync, query)
        if df is not None:
            logging.info(f"✅ New Connection Data: {len(df)} rows fetched")
            tracker_log_and_progress(task_id, f"✅ New Connection Data: {len(df)} rows fetched")
        else:
            tracker_log_and_progress(task_id, f"❌ No new connection data found.")
            logging.warning("❌ No data returned for new connections.")
    except Exception as e:
        logging.error(f"❌ Exception in fetch_new_connection_data: {e}")
        tracker_log_and_progress(task_id, f"❌ Query execution failed: {e}")

    return df
