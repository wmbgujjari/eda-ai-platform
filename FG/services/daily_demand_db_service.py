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

async def fetch_daily_demand_data(section_ids: str, task_id: str, start_date: str, end_date: str):

    #queries = load_queries()
    base_query = get_query_by_name("DAILY_DEMAND_FORECAST_QUERY")

    # Format section IDs for SQL
    #formatted_section_ids = ', '.join(f"'{sid}'" for sid in section_ids)

    # Inject parameters
    query = base_query.format(
        section_ids=section_ids,
        start_date=start_date,
        end_date=end_date
    )
    print(query)
    tracker_log_and_progress(task_id, "✅ Query Execution Started Fetching rows from Oracle")

    loop = asyncio.get_event_loop()
    try:
        df = await loop.run_in_executor(executor, fetch_data_sync, query)
        if df is not None:
            logging.info(f"✅ Fetched {len(df)} rows from Oracle")
            tracker_log_and_progress(task_id, f"✅ Fetched {len(df)} rows from Oracle")
        else:
            tracker_log_and_progress(task_id, f"❌ No data fetched.")
            logging.warning("❌ No data returned from Oracle.")
    except Exception as e:
        logging.error(f"❌ Exception in async fetch_from_oracle: {e}")
        tracker_log_and_progress(task_id, f"❌ Issue in Query fetch failed: {e}")

    return df

async def fetch_day_wise_weather_data(task_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    #queries = load_queries()
    base_query = get_query_by_name("WEATHER_MASTER_FUTURE_DATES_QUERY")

    # Inject parameters
    query = base_query.format(
        start_date=start_date,
        end_date=end_date
    )
    print(query)

    loop = asyncio.get_event_loop()
    try:
        df_weather = await loop.run_in_executor(executor, fetch_data_sync, query)
        if df_weather is not None:
            logging.info(f"✅ Fetched {len(df_weather)} rows from Oracle")
            tracker_log_and_progress(task_id, f"✅ Fetched {len(df_weather)} rows from Oracle")
        else:
            #tracker_log_and_progress(f"❌ No data fetched.")
            logging.warning("❌ No data returned from df_weather.")
            tracker_log_and_progress(task_id, f"❌ No data fetched for df_weather.")
        df_weather["DATE"] = pd.to_datetime(df_weather["WEATHER_DATE"])            
    except Exception as e:
        logging.error(f"❌ Exception in async fetch_from_oracle: {e}")
        tracker_log_and_progress(task_id, f"❌ Issue in df_weather Query fetch failed: {e}")

    return df_weather
  