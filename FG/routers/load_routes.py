from fastapi import APIRouter
from FG.tasks.load_tasks import train_load_model, predict_load

router = APIRouter(prefix="/load", tags=["Load Forecasting"])

@router.post("/train")
async def train():
    train_load_model.delay()
    return {"message": "Load Forecasting Training Started"}

@router.post("/predict")
async def predict():
    result = predict_load.delay()
    return result.get()
