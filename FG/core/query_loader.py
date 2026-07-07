import re
import os

def get_query_by_name(name: str, file_name="Oracle_Queries.sql") -> str:
    # Get absolute path to the 'core' folder regardless of working directory
    base_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the 'core/' folder
    file_path = os.path.join(base_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Query file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Regex to extract the named query
    pattern = re.compile(rf"-- name:\s*{re.escape(name)}\s*\n(.*?)(?=\n-- name:|\Z)", re.DOTALL)
    match = pattern.search(content)

    if not match:
        raise KeyError(f"Query '{name}' not found in file '{file_path}'")

    return match.group(1).strip()
