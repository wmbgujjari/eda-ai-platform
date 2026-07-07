#import cx_Oracle
import oracledb
import logging

oracledb.init_oracle_client(lib_dir=None)

# Session pool configuration parameters
ORACLE_CONFIG = {
    "user": "MIS_USER",
    "password": "bpdcl123",
    "dsn": "10.10.69.192:1521/MISDB",
    "min": 2,
    "max": 10,
    "increment": 1,
    "encoding": "UTF-8"
}

# Initialize session pool globally
try:
    session_pool = oracledb.SessionPool(
        user=ORACLE_CONFIG["user"],
        password=ORACLE_CONFIG["password"],
        dsn=ORACLE_CONFIG["dsn"],
        min=ORACLE_CONFIG["min"],
        max=ORACLE_CONFIG["max"],
        increment=ORACLE_CONFIG["increment"],
        encoding=ORACLE_CONFIG["encoding"],
        threaded=True,  # Enable for multithreaded apps
        getmode=oracledb.SPOOL_ATTRVAL_WAIT  # Wait if no connections are available
    )
    logging.info("✅ Oracle session pool created successfully.")

except oracledb.DatabaseError as e:
    logging.error(f"❌ Failed to create Oracle session pool: {e}")
    session_pool = None


def get_db_connection():
    try:
        if session_pool is None:
            logging.error("❌ Session pool is not initialized.")
            return None
        conn = session_pool.acquire()
        return conn
    except oracledb.DatabaseError as e:
        logging.error(f"❌ Error acquiring DB connection from session pool: {e}")
        return None
