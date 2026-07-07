import os

import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "models/pandas/"
PYSPARK_MODEL_PATH = "models/pyspark/spark_consumption_model"
PYSPARK_METADATA_PATH="models/pyspark/spark_consumption_model.pkl"
THRESHOLD_MB = 50
file_path = "data/consumption.csv"

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY","").strip()
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM","HS256").strip()
#EXPECTED_USER_ID = os.getenv("EXPECTED_USER_ID","").strip()
#EXPECTED_ROLE = os.getenv("EXPECTED_ROLE","").strip()
EXPECTED_USER_ID='69d8bb2e-a0a06073-294c684c-2a7a5dcf'
EXPECTED_ROLE='ROLE_BIHAR'
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this will be inside /tasks
#MODEL_PATH = os.path.join(BASE_DIR, "models", "pandas")