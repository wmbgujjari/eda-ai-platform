import logging
import pandas as pd
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from FG.core.database import get_db_connection, session_pool
#from core.database_uat import get_uat_db_connection, uat_session_pool
from FG.core.utils.log_and_progress import tracker_log_and_progress
from FG.core.query_loader import get_query_by_name
from contextlib import contextmanager

executor = ThreadPoolExecutor()



@contextmanager
def safe_db_connection():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        yield cursor
    except Exception as e:
        logging.error(f"❌ Error using DB connection: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)


# ✅ Step 2: Generic fetch method using the context manager
def fetch_data_sync(query: str) -> pd.DataFrame | None:
    """
    Executes the given SQL query and returns the result as a DataFrame.
    Uses safe_db_connection to ensure proper resource management.
    """
    try:
        with safe_db_connection() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            logging.info(f"✅ Retrieved {len(df)} rows")
            return df
    except Exception as e:
        logging.error(f"❌ Oracle data fetch failed in fetch_data_sync(): {e}")
        return None

#def fetch_data_sync(query: str):
#    conn = None
#    cursor = None
#    try:
        #if db_source == "dev":
        #    conn = get_db_connection()
        #elif db_source == "uat":
#        conn = get_db_connection()
#        if conn is None:
#            return None
#        
#        cursor = conn.cursor()
#        cursor.execute(query)
#        rows = cursor.fetchall()
#        columns = [col[0] for col in cursor.description]
#        df = pd.DataFrame(rows, columns=columns)
#
#        return df
#
#    except Exception as e:
#        logging.error(f"Oracle data fetch failed: {e}")
#        return None
#
#    finally:
#        if cursor:
#            cursor.close()
#        if conn:
            #if db_source == "dev":
#            session_pool.release(conn)
            #elif db_source == "uat":  
            #    uat_session_pool.release(conn)
                

async def fetch_division_ids(officeId: str) -> List[str]:
    #803-10 for consumptiom prediction
    base_query = get_query_by_name("OFFICD_ID")
   
    query = base_query.format(
        officeId=officeId
    )   
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

