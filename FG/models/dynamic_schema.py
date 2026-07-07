from pydantic import BaseModel
from typing import List

class QueryPreviewRequest(BaseModel):
    query: str

class DynamicTrainRequest(BaseModel):
    query: str
    feature_columns: List[str]
    target_column: str
    officeId: str
