import logging
from datetime import datetime
from FG.core.database import get_db_connection, session_pool  # Make sure you have this
from typing import Optional

def insert_audit_log(module: str, message: str, status: str):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            logging.error("❌ Database connection failed.")
            return

        cursor = conn.cursor()

        query = """
            INSERT INTO EDA_AUDIT_LOGS (LOG_TIME, MODEL, MESSAGE, STATUS)
            VALUES (:1, :2, :3, :4)
        """
        cursor.execute(query, (datetime.now(), module, message, status))
        conn.commit()

    except Exception as e:
        logging.error(f"❌ Failed to insert audit log: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)

