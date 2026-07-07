import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

MODEL_PATH = "models/load_model.pkl"

def train_load_model(df):
    """Train Load Forecasting model using XGBoost."""
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    model = XGBRegressor()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model
