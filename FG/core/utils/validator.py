def is_safe_sql(query: str) -> bool:
    lower = query.strip().lower()
    return lower.startswith("select") and "drop" not in lower and "delete" not in lower
