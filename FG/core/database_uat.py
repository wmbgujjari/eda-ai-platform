#import cx_Oracle
import oracledb
import logging

oracledb.init_oracle_client(lib_dir=None)
# Session pool configuration parameters
UAT_ORACLE_CONFIG = {
    "user": "MIS_USER",
    "password": "FG@MISnpdb",
    "dsn": "10.100.21.20:1521/NPPDBMIS",
    "min": 2,
    "max": 10,
    "increment": 1,
    "encoding": "UTF-8"
}

# Initialize session pool globally
try:
    uat_session_pool = oracledb.SessionPool(
        user=UAT_ORACLE_CONFIG["user"],
        password=UAT_ORACLE_CONFIG["password"],
        dsn=UAT_ORACLE_CONFIG["dsn"],
        min=UAT_ORACLE_CONFIG["min"],
        max=UAT_ORACLE_CONFIG["max"],
        increment=UAT_ORACLE_CONFIG["increment"],
        encoding=UAT_ORACLE_CONFIG["encoding"],
        threaded=True,  # Enable for multithreaded apps
        getmode=oracledb.SPOOL_ATTRVAL_WAIT  # Wait if no connections are available
    )
    logging.info("✅ Oracle session pool created successfully.")

except oracledb.DatabaseError as e:
    logging.error(f"❌ Failed to create Oracle session pool: {e}")
    uat_session_pool = None


def get_uat_db_connection():
    try:
        if uat_session_pool is None:
            logging.error("❌ Session pool is not initialized.")
            return None
        conn = uat_session_pool.acquire()
        return conn
    except oracledb.DatabaseError as e:
        logging.error(f"❌ Error acquiring DB connection from session pool: {e}")
        return None
