import logging
import pandas as pd
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from core.database import get_db_connection, session_pool
from core.database_uat import get_uat_db_connection, uat_session_pool
from utils.log_and_progress import tracker_log_and_progress
from core.query_loader import get_query_by_name

executor = ThreadPoolExecutor()

def fetch_data_sync(query: str):
    conn = None
    cursor = None
    try:
        #if db_source == "dev":
        #    conn = get_db_connection()
        #elif db_source == "uat":
        conn = get_db_connection()
        if conn is None:
            return None
        
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

        return df

    except Exception as e:
        logging.error(f"Oracle data fetch failed: {e}")
        return None

    finally:
        if cursor:
            cursor.close()
        if conn:
            #if db_source == "dev":
            session_pool.release(conn)
            #elif db_source == "uat":  
            #    uat_session_pool.release(conn)
                



async def fetch_consumer_consumption_data(section_ids: list[str], task_id: str, start_date: str, end_date: str):
    #queries = load_queries()
    base_query = get_query_by_name("CONSUMPTION_FORECAST_QUERY")

    # Format section IDs for SQL
    formatted_section_ids = ', '.join(f"'{sid}'" for sid in section_ids)

    # Inject parameters
    query = base_query.format(
        section_ids=formatted_section_ids,
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


async def fetch_new_connection_data(section_ids: list[str], task_id: str, start_date: str, end_date: str):
    #queries = load_queries()
    base_query = get_query_by_name("NEW_CONNECTION_QUERY")

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


async def fetch_from_oracle_daywise(section_id: str):
    formatted_section_ids = ', '.join(f"'{sid}'" for sid in section_id)
    query = f"""
SELECT
    om2.ID AS SECTION_ID,
    TO_CHAR(TO_DATE(TO_CHAR(NAA.ENE_DATE), 'YYYYMMDD'), 'YYYY-MM-DD HH24:MI:SS') AS DAYS,
    WRD.TEMPERATURE,
    WRD.HUMIDITY,
    WRD.WIND_SPEED,
    WRD.HOLIDAY,
    WRD.HOLIDAY_NAME,
    SUM(NAA.KWH_IMP) AS ACTUAL_CONSUMPTION
FROM
    mis_user.AMI_NETWORK_AREA_AGGREGATION NAA
INNER JOIN
    mis_user.WEATHER_RAW_DATA WRD
ON
    TO_DATE(NAA.ENE_DATE, 'YYYYMMDD') = TRUNC(WRD.DAY)
INNER JOIN
    AMI_OFFICE_M aom ON NAA.OFF_ID = aom.ID
INNER JOIN
    COMMON_USER.OFFICE_MASTER om2 ON TO_CHAR(aom.OFFICE_CODE) = TO_CHAR(om2.OLD_UCODE)
WHERE
    TO_DATE(NAA.ENE_DATE, 'YYYYMMDD') BETWEEN TO_DATE('2024-12-01', 'YYYY-MM-DD') AND TO_DATE('2024-12-31', 'YYYY-MM-DD')
    AND aom.RECORD_STATUS = 1
    AND om2.RECORD_STATUS = 1
    AND om2.ID IN ({formatted_section_ids})
    GROUP BY om2.ID,
    WRD.TEMPERATURE,
    WRD.HUMIDITY,
    WRD.WIND_SPEED,
    WRD.HOLIDAY,
    WRD.HOLIDAY_NAME,
    TO_CHAR(TO_DATE(TO_CHAR(NAA.ENE_DATE), 'YYYYMMDD'), 'YYYY-MM-DD HH24:MI:SS') 
            """   
    print(query)
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(executor, fetch_data_sync,query)
    if df is not None:
        logging.info(f"✅ Fetched {len(df)} rows from Oracle")
    return df


async def fetch_division_ids(officeId: str) -> List[str]:
    #803-10 for consumptiom prediction
    base_query = get_query_by_name("OFFICD_ID")
   
    query = base_query.format(
        officeId=officeId
    )   

#    query = f"""
#    select id 
#from common_user.office_master where record_status=1 
#and parentoffice_id='{officeId}' 
#start with id ='{officeId}'  connect by prior id= parentoffice_id 
#    """
    loop = asyncio.get_event_loop()
    division_rows = await loop.run_in_executor(executor, fetch_data_sync, query)
    
    if division_rows is not None:
        logging.info(f"✅ Fetched {len(division_rows)} rows from Oracle")
    else:
        logging.warning("⚠️ No division rows fetched.")
        return []

    # Extract the list of IDs
    division_ids = division_rows
    print("🔁 Division IDs:", division_ids)
    return division_ids

async def fetch_section_ids(division_id: str)-> List[str]:

    base_query = get_query_by_name("DIVISION_IDS")

    query = base_query.format(
        division_id=division_id
    )     
#    query = f"""
#SELECT ID FROM COMMON_USER.OFFICE_MASTER om WHERE 
#ID IN (
#          SELECT ID
#          FROM COMMON_USER.OFFICE_MASTER OMF
#          WHERE RECORD_STATUS = 1
#            START WITH ID = '{division_id}'
#            CONNECT BY PRIOR ID = PARENTOFFICE_ID
#      )
#ORDER BY ID """    
    print(f"🟡 Query being run for division {division_id}:\n{query}")
    loop = asyncio.get_event_loop()
    section_id = await loop.run_in_executor(executor, fetch_data_sync,query)
    #print("selectionId's",section_id)
    if section_id is not None:
        logging.info(f"✅ Fetched {len(section_id)} rows from Oracle")
    return section_id["ID"].tolist()
    
    
async def get_division_section_map(officeId: str) -> dict:
    print("in divisions")
    division_df = await fetch_division_ids(officeId)
    division_ids = division_df["ID"].tolist()
    #division_ids = await fetch_division_ids()  # Fetch division IDs for the given circle
    division_section_map = {}

    for division_id in division_ids:
        section_ids = await fetch_section_ids(division_id)
        print("section_id",section_ids)
        if section_ids:
            division_section_map[division_id] = section_ids

    return division_section_map    

