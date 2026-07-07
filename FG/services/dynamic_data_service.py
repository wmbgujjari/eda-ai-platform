import pandas as pd
from FG.core.utils.validator import is_safe_sql
from FG.core.database import get_db_connection  # Existing pooled Oracle function

def execute_dynamic_query(query: str) -> pd.DataFrame:
    if not is_safe_sql(query):
        raise ValueError("Only safe SELECT queries allowed.")
    
    print(f"Executing dynamic query: {query}")
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df
