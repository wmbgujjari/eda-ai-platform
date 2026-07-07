# ml_engine/inference/evaluation.py

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from datetime import date

def evaluate_section_performance(
        section_id,
        model_name,
        train_df,
        trained_with,
        y_true,
        y_pred
    ):
    """
    Compute metrics (R2, RMSE) and return the accuracy map for DB insertion.
    """

    r2_sec = round(r2_score(y_true, y_pred), 4)
    rmse_sec = round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)

    accuracy_map = {
        section_id: {
            "MODEL_VERSION": f"Daily_Demand_{model_name}_{train_df['DAYS'].max().strftime('%Y%m%d')}",
            "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
            "TRAINED_WITH": trained_with,
            "R2_SCORE": r2_sec,
            "RMSE": rmse_sec,
            "DASHBOARD": "Daily_Demand",
        }
    }

    return accuracy_map, r2_sec, rmse_sec
