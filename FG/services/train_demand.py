import os
import pandas as pd
import numpy as np
import joblib
import logging
from scipy.stats import zscore
from FG.core.config import file_path, THRESHOLD_MB,MODEL_PATH,PYSPARK_MODEL_PATH,PYSPARK_METADATA_PATH
import lightgbm as lgb  
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from FG.core.utils.log_and_progress import tracker_log_and_progress
from FG.services.consumption_db_service import fetch_demand_consumption_monthly_data_sectionwise,fetch_demand_weather_data
from fastapi import HTTPException
from pycaret.regression import setup, compare_models, pull, save_model, finalize_model

def load_model(size_mb: float):
    try:
        if size_mb <= THRESHOLD_MB:
            path = MODEL_PATH
            if os.path.exists(path):
                model_bundle = joblib.load(path)
                return {
                    "model": model_bundle.get("model"),
                    "scaler": model_bundle.get("scaler"),
                    "feature_columns": model_bundle.get("feature_columns"),
                    "trained_with": "pandas"
                }
            else:
                logging.warning("Pandas model path does not exist.")
                return None

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

async def load_data(section_id: str,task_id: str,train_year: str,trian_month: str):
    try:
        logging.info("File not found. Fetching from Oracle.")
        df = await fetch_demand_consumption_monthly_data_sectionwise(section_id,task_id,train_year,trian_month)
        print("loding df")
        df_weather = await fetch_demand_weather_data(task_id,train_year,trian_month)
        print("loaiding df_weather")
        if df is None:
            raise Exception("fetch_demand_consumption_monthly_data() returned None")
        
        if df_weather is None:
            raise Exception("fetch_demand_weather_data() returned None")
        
        logging.info(f"Return Fetched {len(df)} rows from Oracle")
        tracker_log_and_progress(task_id, f"Return Fetched {len(df)} rows from Oracle")
        file_size = df.memory_usage(deep=True).sum() / (1024 ** 2)

        # Compute memory usage only for Pandas DataFrame
        if isinstance(df, pd.DataFrame):
            file_size = df.memory_usage(deep=True).sum() / (1024 ** 2)
        else:
            # Spark DataFrame – estimate size or set as unknown
            file_size = None

        return df,df_weather, file_size
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail="Failed to load data")


def train_with_pandas(df, df_weather, model_dir: str):
    try:
        logging.info("Starting Pandas training with demand + weather data...")

        # --- Ensure input is DataFrame ---
        train_df, features, target,le = process_data(df, df_weather)
        if len(train_df) < 12:
            logging.warning("Skipping training: insufficient data (<12 records).")
            return None

        X = train_df[features].copy()
        y = train_df[target]

        X.columns = X.columns.str.replace(r"\s+", "_", regex=True)

        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10,
            random_state=42,
            force_row_wise=True
        )

        # Grid Search
        grid_search = GridSearchCV(model, {
            'n_estimators': [100],
            'max_depth': [3],
            'learning_rate': [0.1],
            'subsample': [1],
            'colsample_bytree': [1],
            'num_leaves': [7],
            'min_data_in_leaf': [5, 10]
        }, cv=3, n_jobs=-1, verbose=0)

        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        model_file = f"models/pandas/model_lgbm_office.pkl"
        # Save model
        #joblib.dump(best_model, model_file)

        logging.info("✅ Section-wise training completed")

        return {
            "model": best_model,
            "train_df": train_df,
            "features": features,
            "trained_by": "LGBMRegressor",
            "target_col": target
        }

    except Exception as e:
        logging.error(f"Pandas training failed: {e}")
        raise


def train_with_automl(df, df_weather, model_dir: str):
    try:
        logging.info("🚀 Starting AutoML training with demand + weather data...")

        # --- Ensure input is DataFrame ---
        train_df, features, target, le = process_data(df, df_weather)

        train_df = train_df.dropna(subset=features + [target])
        # DROP BILLDATE before AutoML

        numeric_features = [f for f in features if f != "SECTION_ID_En"]
        categorical_features = ["SECTION_ID_En"]

        s = setup(
        data=train_df,
        target=target,
        session_id=42,
        use_gpu=False,
        fold_strategy="kfold", # or "timeseries"
        numeric_features=numeric_features,
        #categorical_features=categorical_features,
        remove_outliers=False,
        verbose=False
        )
        best_model = compare_models(
            sort="RMSE",  # Or "R2"
            fold=3,
            turbo=True  # Quick run, set False for full search
        )

        finalized_model = finalize_model(best_model)

        # Save model
        #model_path = f"{model_dir}/automl_best_model"
        #save_model(finalized_model, model_path)

        logging.info(f"✅ AutoML training completed, best model: {best_model}")

        return {
            "model": finalized_model,
            "train_df": train_df,
            "features": features,
            "trained_by": "PyCaret_AutoML",
            "target_col": target,
            "le_section_id": le
        }

    except Exception as e:
        logging.error(f"❌ AutoML training failed: {e}")
        raise

def process_data(df, df_weather):
    # --- Ensure input is DataFrame ---
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if isinstance(df_weather, list):
        df_weather = pd.DataFrame(df_weather)

    # --- Ensure target column naming ---
    df = df.rename(columns={"CONSUMPTION": "ACTUAL_CONSUMPTION"})

    # --- Ensure SECTION_ID present ---
    if "SECTION_ID" not in df.columns:
        raise ValueError("SECTION_ID column missing in input data")

    # --- Merge Weather Monthly ---
    df_weather["DATE"] = pd.to_datetime(df_weather["WEATHER_DATE"])
    df_weather["BILLMONTH"] = df_weather["DATE"].dt.month
    df_weather["BILLYEAR"] = df_weather["DATE"].dt.year
    monthly_weather = (
        df_weather.groupby(["BILLYEAR", "BILLMONTH"])
        .mean(numeric_only=True)
        .reset_index()
    )

    df_merged = pd.merge(df, monthly_weather,
                         on=["BILLMONTH", "BILLYEAR"], how="left")

    # --- Feature Engineering ---
    df_merged["Section_Monthly_Avg"] = (
        df_merged.groupby(["SECTION_ID", "BILLMONTH"])["ACTUAL_CONSUMPTION"]
        .transform("mean")
        .round(2)
    )
    df_merged["Consumption_vs_Avg"] = (
        df_merged["ACTUAL_CONSUMPTION"] - df_merged["Section_Monthly_Avg"]
    )

    df_merged['BILLDATE'] = pd.to_datetime(dict(
        year=df_merged['BILLYEAR'],
        month=df_merged['BILLMONTH'],
        day=1
    ))
    df_merged["rain_cloud_inter"] = (
        df_merged["RAIN_CHANCE"] * df_merged["CLOUD"]
    )

    # Previous month consumption
    df_merged['Previous_month_Consumption'] = (
        df_merged.groupby('SECTION_ID')["ACTUAL_CONSUMPTION"]
        .shift(1).fillna(0)
    )

    # Sort
    df_sorted = df_merged.sort_values(['SECTION_ID', 'BILLDATE'])

    # Lags & rolling features
    weather_cols = ['TEMPERATURE', 'HUMIDITY', 'RAIN_CHANCE', 'PRESSURE', 'CLOUD']
    for col in weather_cols:
        for lag in [1, 2, 3]:
            df_sorted[f'{col}_lag{lag}'] = df_sorted.groupby('SECTION_ID')[col].shift(lag)
        for window in [2, 3]:
            df_sorted[f'{col}_roll{window}'] = (
                df_sorted.groupby('SECTION_ID')[col]
                .rolling(window).mean()
                .reset_index(0, drop=True)
            )

    # Train data
    train_df = df_sorted.dropna()

    if len(train_df) < 12:
        return None, None, None

    # Label encode SECTION_ID
    le = LabelEncoder()
    train_df["SECTION_ID_En"] = le.fit_transform(train_df["SECTION_ID"].astype(str))

    # Feature set
    features = [
        "SECTION_ID_En", "Consumption_vs_Avg", "Previous_month_Consumption",
        "TEMPERATURE_lag1", "RAIN_CHANCE_lag1", "PRESSURE_lag1", "CLOUD_lag1",
        "TEMPERATURE_roll2", "RAIN_CHANCE_roll2", "PRESSURE_roll2", "CLOUD_roll2",
        "TEMPERATURE_roll3", "RAIN_CHANCE_roll3", "PRESSURE_roll3"
    ]
    target = "ACTUAL_CONSUMPTION"

    return train_df, features, target,le


def train_with_pandas_old(df, df_weather, model_dir: str):
    try:
        logging.info("Starting Pandas training with demand + weather data...")

        # --- Ensure input is DataFrame ---
        if isinstance(df, list):
            df = pd.DataFrame(df)
        if isinstance(df_weather, list):
            df_weather = pd.DataFrame(df_weather)

        # --- Ensure target column naming ---
        df = df.rename(columns={"CONSUMPTION": "ACTUAL_CONSUMPTION"})

        # --- Ensure SECTION_ID present ---
        if "SECTION_ID" not in df.columns:
            raise ValueError("SECTION_ID column missing in input data")

        # --- Merge Weather Monthly ---
        df_weather["DATE"] = pd.to_datetime(df_weather["WEATHER_DATE"])
        df_weather["BILLMONTH"] = df_weather["DATE"].dt.month
        df_weather["BILLYEAR"] = df_weather["DATE"].dt.year
        monthly_weather = (
            df_weather.groupby(["BILLYEAR", "BILLMONTH"])
            .mean(numeric_only=True)
            .reset_index()
        )

        df_merged = pd.merge(df, monthly_weather,
                             on=["BILLMONTH", "BILLYEAR"], how="left")

        # --- Feature Engineering ---
        df_merged["Section_Monthly_Avg"] = (
            df_merged.groupby(["SECTION_ID", "BILLMONTH"])["ACTUAL_CONSUMPTION"]
            .transform("mean")
            .round(2)
        )
        df_merged["Consumption_vs_Avg"] = (
            df_merged["ACTUAL_CONSUMPTION"] - df_merged["Section_Monthly_Avg"]
        )

        df_merged['BILLDATE'] = pd.to_datetime(dict(
            year=df_merged['BILLYEAR'],
            month=df_merged['BILLMONTH'],
            day=1
        ))
        df_merged["rain_cloud_inter"] = (
            df_merged["RAIN_CHANCE"] * df_merged["CLOUD"]
        )

        # Previous month consumption
        df_merged['Previous_month_Consumption'] = (
            df_merged.groupby('SECTION_ID')["ACTUAL_CONSUMPTION"]
            .shift(1).fillna(0)
        )

        # Sort
        df_sorted = df_merged.sort_values(['SECTION_ID', 'BILLDATE'])

        # Lags & rolling features
        weather_cols = ['TEMPERATURE', 'HUMIDITY', 'RAIN_CHANCE', 'PRESSURE', 'CLOUD']
        for col in weather_cols:
            for lag in [1, 2, 3]:
                df_sorted[f'{col}_lag{lag}'] = df_sorted.groupby('SECTION_ID')[col].shift(lag)
            for window in [2, 3]:
                df_sorted[f'{col}_roll{window}'] = (
                    df_sorted.groupby('SECTION_ID')[col]
                    .rolling(window).mean()
                    .reset_index(0, drop=True)
                )

        # Train data
        train_df = df_sorted.dropna()

        # Label encode SECTION_ID
        le = LabelEncoder()
        train_df["SECTION_ID_En"] = le.fit_transform(train_df["SECTION_ID"].astype(str))
        #joblib.dump(le, f"{model_dir}/section_label_encoder.pkl")

        # Feature set
        features = [
            "SECTION_ID_En", "Consumption_vs_Avg", "Previous_month_Consumption",
            "TEMPERATURE_lag1", "RAIN_CHANCE_lag1", "PRESSURE_lag1", "CLOUD_lag1",
            "TEMPERATURE_roll2", "RAIN_CHANCE_roll2", "PRESSURE_roll2", "CLOUD_roll2",
            "TEMPERATURE_roll3", "RAIN_CHANCE_roll3", "PRESSURE_roll3"
        ]
        target = "ACTUAL_CONSUMPTION"

        if len(train_df) < 12:
            logging.warning("Skipping training: insufficient data (<12 records).")
            return None

        X = train_df[features].copy()
        y = train_df[target]

        X.columns = X.columns.str.replace(r"\s+", "_", regex=True)

        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10,
            random_state=42,
            force_row_wise=True
        )

        # Grid Search
        grid_search = GridSearchCV(model, {
            'n_estimators': [100],
            'max_depth': [3],
            'learning_rate': [0.1],
            'subsample': [1],
            'colsample_bytree': [1],
            'num_leaves': [7],
            'min_data_in_leaf': [5, 10]
        }, cv=3, n_jobs=-1, verbose=0)

        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        model_file = f"models/pandas/model_lgbm_office.pkl"
        # Save model
        #joblib.dump(best_model, model_file)

        logging.info("✅ Section-wise training completed")

        return {
            "model": best_model,
            "train_df": train_df,
            "features": features,
            "trained_by": "LGBMRegressor",
            "target_col": target
        }

    except Exception as e:
        logging.error(f"Pandas training failed: {e}")
        raise

