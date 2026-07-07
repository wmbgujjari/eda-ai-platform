import os
import pandas as pd
import numpy as np
import joblib
import logging
from scipy.stats import zscore
from core.config import file_path, THRESHOLD_MB,MODEL_PATH,PYSPARK_MODEL_PATH,PYSPARK_METADATA_PATH
import lightgbm as lgb  
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from core.utils.log_and_progress import tracker_log_and_progress
from services.daily_demand_db_service import fetch_daily_demand_data,fetch_day_wise_weather_data
from fastapi import HTTPException
from pycaret.regression import setup, compare_models, pull, save_model, finalize_model
from core.utils.FGautoML import get_regression_model
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.tensorflow
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


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

async def load_data(section_id: str,task_id: str,start_date: str,end_date: str):
    try:
        logging.info("File not found. Fetching from Oracle.")
        df = await fetch_daily_demand_data(section_id,task_id,start_date,end_date)
        print("loding df")
        df_weather = await fetch_day_wise_weather_data(task_id,start_date,end_date)
        print("loaiding df_weather")
        if df is None:
            raise Exception("fetch_daily_demand_data() returned None")
        
        if df_weather is None:
            raise Exception("fetch_day_wise_weather_data() returned None")
        
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


def train_lstm_model(df, df_weather, model_name: str, task_id: str, section_id: str):
    """
    Train or fine-tune an LSTM model for section-level time-series prediction.
    Returns a unified model bundle consistent with other training methods.
    """
    import mlflow
    import mlflow.tensorflow

    logging.info(f"🚀 Starting Deep Learning training for section: {section_id}")
    tracker_log_and_progress(task_id, f"🚀 Starting Deep Learning training for section: {section_id}")

    # ---- Step 1️⃣: Common preprocessing ----
    train_df, features, target, le = process_data(df, df_weather)

    logging.info(f"🧩 Preprocessing completed for section {section_id}. Features: {features}")
    tracker_log_and_progress(task_id, f"🧩 Preprocessing completed for section {section_id}. Features count: {len(features)}")

    X = train_df[features].copy()
    y = train_df[target].copy()

    # ---- Step 2️⃣: Scale the target variable for LSTM ----
    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(y.values.reshape(-1, 1))

    window_size = 7
    X_seq, y_seq = [], []
    for i in range(window_size, len(scaled_y)):
        X_seq.append(scaled_y[i - window_size:i, 0])
        y_seq.append(scaled_y[i, 0])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    X_seq = np.reshape(X_seq, (X_seq.shape[0], X_seq.shape[1], 1))

    # ---- Step 3️⃣: Create or load model ----
    model_file = os.path.join(MODEL_PATH, f"dl_model_section_{section_id}.keras")

    logging.info(f"🧠 Creating NEW LSTM model for section {section_id}")
    tracker_log_and_progress(task_id, f"🧠 Creating NEW LSTM model (non-incremental)")

    model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], 1)),
            Dropout(0.2),
            LSTM(64),
            Dense(1)
        ])
    model.compile(optimizer="adam", loss="mse")

    # ---- Step 4️⃣: Train with MLflow ----
    # ✅ Ensure any previous run is closed before starting a new one
    if mlflow.active_run() is not None:
        mlflow.end_run()

    try:
        with mlflow.start_run(run_name=f"LSTM_Section_{section_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",nested=True):
            mlflow.tensorflow.autolog()

            callbacks = [
                EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                ModelCheckpoint(model_file, monitor='val_loss',save_best_only=True)
            ]

            history = model.fit(
                X_seq, y_seq,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )

            # Save model
            model.save(model_file)
            tracker_log_and_progress(task_id, f"✅ LSTM model saved successfully for {section_id}")

            mlflow.log_param("section_id", section_id)
            mlflow.log_param("window_size", window_size)
            mlflow.log_metric("final_loss", history.history['loss'][-1])
            mlflow.log_artifact(model_file)

    finally:
        # ✅ Always ensure MLflow run is closed
        if mlflow.active_run() is not None:
            mlflow.end_run()

    # ---- Step 5️⃣: Unified Return ----
    model_bundle = {
        "model": model,
        "train_df": train_df,
        "features": features,
        "trained_by": model_name.upper(),
        "target_col": target,
        "le": le,
        "is_lstm": True,
        "window_size": window_size
    }

    logging.info(f"✅ LSTM model training completed and returned for section: {section_id}")
    tracker_log_and_progress(task_id, f"✅ LSTM model training completed for section: {section_id}")

    return model_bundle

def train_with_pandas(df, df_weather, model_name: str):
    try:
        logging.info(f"Starting training with {model_name } Revenue Collections + weather data...")

        # --- Ensure input is DataFrame ---
        train_df, features, target,le = process_data(df, df_weather)
        if len(train_df) < 12:
            logging.warning("Skipping training: insufficient data (<12 records).")
            return None

        X = train_df[features].copy()
        y = train_df[target]

        X.columns = X.columns.str.replace(r"\s+", "_", regex=True)

        base_model, param_grid = get_regression_model(model_name)

        # --- Grid Search (if applicable) ---
        if param_grid:
            logging.info(f"🔍 Running GridSearchCV for {model_name}...")
            grid_search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1, verbose=0)
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            logging.info(f"🏆 Best Params for {model_name}: {grid_search.best_params_}")
        else:
            base_model.fit(X, y)
            best_model = base_model
        
        model_file = f"models/pandas/model_lgbm_office.pkl"
        # Save model
        #joblib.dump(best_model, model_file)

        logging.info("✅ Section-wise training completed")

        return {
            "model": best_model,
            "train_df": train_df,
            "features": features,
            "trained_by": model_name.upper(),
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
    """
    Preprocess daily demand and weather data for model training.
    Returns: train_df, features, target, le
    """

    # --- Ensure DataFrame types ---
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if isinstance(df_weather, list):
        df_weather = pd.DataFrame(df_weather)

    # --- Validate required columns ---
    required_cols = ["SECTION_ID", "DAYS", "ACTUAL_CONSUMPTION"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing column '{col}' in demand data")

    # --- Convert date columns ---
    df["DAYS"] = pd.to_datetime(df["DAYS"])
    df_weather["WEATHER_DATE"] = pd.to_datetime(df_weather["WEATHER_DATE"])

    # --- Merge Weather Data (daily) ---
    df_merged = pd.merge(
        df,
        df_weather,
        left_on=["DAYS"],
        right_on=["WEATHER_DATE"],
        how="left"
    )

    # --- Feature Engineering ---
    df_merged = df_merged.sort_values(["SECTION_ID", "DAYS"])

    # Daily rolling features (for weather)
    weather_cols = ["TEMPERATURE", "HUMIDITY", "RAIN_CHANCE", "PRESSURE", "CLOUD"]
    for col in weather_cols:
        # Lags
        for lag in [1, 2, 3, 7]:
            df_merged[f"{col}_lag{lag}"] = (
                df_merged.groupby("SECTION_ID")[col].shift(lag)
            )

        # Rolling averages
        for window in [3, 7]:
            df_merged[f"{col}_roll{window}"] = (
                df_merged.groupby("SECTION_ID")[col]
                .rolling(window)
                .mean()
                .reset_index(0, drop=True)
            )

    # --- Previous Day Consumption ---
    df_merged["Prev_Day_Consumption"] = (
        df_merged.groupby("SECTION_ID")["ACTUAL_CONSUMPTION"].shift(1)
    )

    # --- Rolling Average of Consumption ---
    df_merged["Rolling_3Day_Consumption"] = (
        df_merged.groupby("SECTION_ID")["ACTUAL_CONSUMPTION"]
        .rolling(3)
        .mean()
        .reset_index(0, drop=True)
    )

    df_merged["Rolling_7Day_Consumption"] = (
        df_merged.groupby("SECTION_ID")["ACTUAL_CONSUMPTION"]
        .rolling(7)
        .mean()
        .reset_index(0, drop=True)
    )

    # --- Interaction features ---
    df_merged["Rain_Cloud_Interaction"] = (
        df_merged["RAIN_CHANCE"] * df_merged["CLOUD"]
    )

    # --- Drop nulls from lag features ---
    train_df = df_merged.dropna().reset_index(drop=True)

    # --- Encode SECTION_ID ---
    le = LabelEncoder()
    train_df["SECTION_ID_En"] = le.fit_transform(train_df["SECTION_ID"].astype(str))

    # --- Define features and target ---
    features = [
        "SECTION_ID_En",
        "Prev_Day_Consumption",
        "Rolling_3Day_Consumption",
        "Rolling_7Day_Consumption",
        "TEMPERATURE_lag1", "TEMPERATURE_lag3",
        "RAIN_CHANCE_lag1", "RAIN_CHANCE_lag3",
        "PRESSURE_lag1", "CLOUD_lag1",
        "Rain_Cloud_Interaction",
        "TEMPERATURE_roll3", "RAIN_CHANCE_roll3",
        "PRESSURE_roll3", "CLOUD_roll3"
    ]

    target = "ACTUAL_CONSUMPTION"

    # --- Return ---
    return train_df, features, target, le


