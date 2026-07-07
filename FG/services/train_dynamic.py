from pycaret.classification import setup, compare_models, predict_model
from pycaret.regression import setup, compare_models, pull, save_model,predict_model,finalize_model
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from datetime import date,datetime
from FG.core.utils.log_and_progress import tracker_log_and_progress
import numpy as np
import logging

def run_automl_training(df: pd.DataFrame, feature_columns: list, target_column: str,task_id: str):
    # Setup AutoML
    automl_setup = setup(
        data=df,
        target=target_column,
        feature_selection=True,
        fold_strategy="kfold",
        remove_outliers=False,
        use_gpu=False,
        session_id=123,
        verbose=False
    )

    # Train best model using AutoML
    best_model = compare_models()

        # ✅ Step 4: Evaluate & Log
    leaderboard = pull()
    logging.info(f"AutoML Leaderboard:\n{leaderboard}")
       
    logging.info(f"Best Model: {best_model}")
    final_model = finalize_model(best_model) 
    logging.info(f"Best Model: {final_model}")

    # Predict on full data
    predictions_df = predict_model(best_model, data=df)

    tracker_log_and_progress(task_id, f"✅ Training complete with Model: {type(best_model).__name__}")

    # Add section ID to predictions (if not already present)
    if 'SECTION_ID' not in df.columns:
        raise ValueError("SECTION_ID column missing from input data")

    if 'DAY' not in df.columns:
        raise ValueError("DAY column missing from input data")    

    predictions_df["SECTION_ID"] = df["SECTION_ID"]
    predictions_df["ACTUAL"] = df[target_column]
    predictions_df["PREDICTED"] = predictions_df["prediction_label"]

    section_data_map = {}
    section_accuracy_map = {}

    for section_id in predictions_df["SECTION_ID"].unique():
        section_df = predictions_df[predictions_df["SECTION_ID"] == section_id]

        # Prepare list of rows for insertion
        records = []
        for _, row in section_df.iterrows():
            records.append({
                "SECTION_ID": section_id,
                "DAY": row["DAY"],
                "ACTUAL": row["ACTUAL"],
                "PREDICTED_VALUE": row["PREDICTED"],
                "MODEL_VERSION": f"dynamic_model_{row['DAY'].strftime('%Y%m%d')}",
                "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
            })

        section_data_map[str(section_id)] = records

        # Calculate R2 and RMSE for accuracy
        y_true = section_df["ACTUAL"]
        y_pred = section_df["PREDICTED"]
        if len(y_true) > 1:  # R2 needs at least two data points
            r2 = r2_score(y_true, y_pred)
        else:
            r2 = 0.0
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        section_accuracy_map[str(section_id)] = {
            "MODEL_VERSION": f"dynamic_model_{row['DAY'].strftime('%Y%m%d')}",
            "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
            "TRAINED_WITH": str(type(best_model).__name__),
            "R2_SCORE": round(r2, 4),
            "RMSE": round(rmse, 4),
        }

        print(f"Section {section_id} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
        tracker_log_and_progress(task_id, f"Section {section_id} - R²: {r2:.4f}, RMSE: {rmse:.4f}")   # Return section-wise data map
    return section_data_map, section_accuracy_map
