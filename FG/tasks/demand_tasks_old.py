from celery import shared_task
from celery import Celery
import pandas as pd
import numpy as np
import pandas as pd
from datetime import date,datetime
import logging
from services.train_demand import train_with_pandas,load_data,train_with_automl
from core.config import MODEL_PATH,THRESHOLD_MB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import asyncio
from typing import List
from core.utils.log_and_progress import tracker_log_and_progress
import requests
from core.utils.common_db_service import fetch_data_sync
from core.query_loader import get_query_by_name

from core.database_uat import get_uat_db_connection, uat_session_pool
logger = logging.getLogger(__name__)


celery = Celery(
    "tasks.demand_tasks",
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json']
)

@shared_task(name="tasks.demand_tasks.schedule_train_demand")
def schedule_train_demand(train_year: str, train_month: str, office_id: str):
    logger.info("✅ Beat triggered schedule_train_demand_tasks")
    # 🔁 You can dynamically calculate date range here
    #end_date = "2024-12-31"#datetime.now().strftime("%Y-%m-%d")
    #start_date = "2024-08-01"#(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")  # last 30 days
    #office_id = "803-10"  # Replace with dynamic logic or loop if needed

    url = f"http://127.0.0.1:8000/demand/train/{train_year}/{train_month}/{office_id}"

    try:
        response = requests.post(url)
        print(f"Triggered training: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error triggering training: {e}")


@shared_task
def fetch_data_and_train_section_wise(section_ids: List[str], task_id: str, train_year: str, train_month: str, model_name: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        for section_id in section_ids:
            try:
                # 1. Load data per section
                df, df_weather, file_size = loop.run_until_complete(
                    load_data(section_id, task_id, train_year, train_month)
                )

                if df.empty:
                    logging.warning(f"⚠️ No data fetched for section: {section_id}")
                    tracker_log_and_progress(task_id, f"⚠️ No data fetched for section: {section_id}")
                    continue

                df_dict = df.to_dict(orient="records")
                logging.info(f"🚀 Starting training for section: {section_id}")
                tracker_log_and_progress(task_id, f"🚀 Starting training for section: {section_id}")

                # 2. Train or increment model (predict + metrics)
                section_data_map, section_accuracy_map = select_or_increment_model(
                    df_dict, df_weather, task_id, model_name
                )

                # 3. Insert predictions + metrics
                if isinstance(section_data_map, dict) and section_id in section_data_map:
                    records = section_data_map[section_id]
                    insert_section_data.apply_async(
                        args=[section_id, records, task_id],
                        queue="demand_queue"
                    )
                    logging.info(f"📤 Queued insertion for section: {section_id}, records: {len(records)}")

                if isinstance(section_accuracy_map, dict) and section_id in section_accuracy_map:
                    metrics = section_accuracy_map[section_id]
                    insert_accuracy_log.apply_async(
                        kwargs={
                            "section_id": section_id,
                            "model_version": metrics["MODEL_VERSION"],
                            "training_date": metrics["TRAINING_DATE"],
                            "trained_with": metrics["TRAINED_WITH"],
                            "r2": metrics["R2_SCORE"],
                            "rmse": metrics["RMSE"],
                            "task_id": task_id
                        },
                        queue="demand_queue"
                    )
                    logging.info(f"📝 Queued accuracy log for section: {section_id}")
                    tracker_log_and_progress(task_id, f"📝 Queued accuracy log for section: {section_id}")

                tracker_log_and_progress(task_id, f"✅ Section training done: {section_id}", "completed")

            except Exception as sec_err:
                logging.error(f"🔥 Error training section {section_id}: {sec_err}")
                tracker_log_and_progress(task_id, f"🔥 Error training section {section_id}: {sec_err}", "failed")

        return f"✅ Section-wise training done for sections: {section_ids}"

    except Exception as e:
        logging.error(f"🔥 Error in fetch_data_and_train: {e}")
        tracker_log_and_progress(task_id, f"🔥 Error in fetch_data_and_train: {e}", "failed")
        return f"Error: {str(e)}"

    finally:
        loop.close()


@shared_task
def fetch_data_and_train(section_ids: List[str], task_id: str, train_year: str, train_month: str, model_name: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 1. Load division data
        df, df_weather, file_size = loop.run_until_complete(
            load_data(section_ids, task_id, train_year, train_month)
        )

        if df.empty:
            logging.warning("⚠️ No data fetched for section(s): %s", section_ids)
            return "No data found for training."

        df_dict = df.to_dict(orient="records")
        logging.info("🚀 Starting training for division with section(s): %s", section_ids)
        tracker_log_and_progress(task_id, f"🚀 Starting training for division with section(s): {section_ids}")

        # 2. Train or increment model (this internally does prediction + metrics)
        section_data_map, section_accuracy_map = select_or_increment_model(
            df_dict, df_weather,task_id, model_name
        )

        # 3. Insert predictions + metrics
        if isinstance(section_data_map, dict) and section_data_map:
            logging.info("✅ Training complete. Starting insertion...")
            tracker_log_and_progress(task_id, "✅ Training complete. Starting insertion...")

            # Insert predicted section-wise data
            for section_id, records in section_data_map.items():
                insert_section_data.apply_async(
                    args=[section_id, records, task_id],
                    queue="demand_queue"
                )
                logging.info(f"📤 Queued insertion for section: {section_id}, records: {len(records)}")

            # Insert section-wise metrics
            for section_id, metrics in section_accuracy_map.items():
                insert_accuracy_log.apply_async(
                    kwargs={
                        "section_id": section_id,
                        "model_version": metrics["MODEL_VERSION"],
                        "training_date": metrics["TRAINING_DATE"],
                        "trained_with": metrics["TRAINED_WITH"],
                        "r2": metrics["R2_SCORE"],
                        "rmse": metrics["RMSE"],
                        "task_id": task_id
                    },
                    queue="demand_queue"
                )
                logging.info(f"📝 Queued accuracy log for section: {section_id}")
                tracker_log_and_progress(task_id, f"📝 Queued accuracy log for section: {section_id}")

        else:
            logging.warning(f"⚠️ Skipping insertion due to training issue: {section_data_map}")
            tracker_log_and_progress(task_id, f"⚠️ Skipping insertion due to training issue: {section_data_map}", "failed")

        tracker_log_and_progress(task_id, "✅ Division training done. Insertion queued for all sections.", "completed")
        return f"Division training done. Insertion started for sections: {section_ids}"

    except Exception as e:
        logging.error(f"🔥 Error in fetch_data_and_train for sections {section_ids}: {e}")
        tracker_log_and_progress(task_id, f"🔥 Error in fetch_data_and_train for sections {section_ids}: {e}", "failed")
        return f"Error: {str(e)}"

    finally:
        loop.close()


@shared_task
def select_or_increment_model_new(df, df_weather, task_id, model_name: str):
    try:
        # --- Decide training approach ---
        if model_name.lower() == "automl":
            logging.info("Full training using AutoML")
            tracker_log_and_progress(task_id, "Full training using AutoML")
            model_bundle = train_with_automl(df, df_weather, model_name)
            trained_with = "automl"
        else:
            logging.info("Full training using Pandas")
            tracker_log_and_progress(task_id, "Full training using Pandas")
            model_bundle = train_with_pandas(df, df_weather, model_name)
            trained_with = "pandas"

        # --- Generate training metrics & future prediction ---
        section_data_map, section_accuracy_map = predict_demand(
            model_bundle=model_bundle,
            trained_with=trained_with,
            task_id=task_id
        )

        logging.info(f"✅ Training and prediction completed for task {task_id}")
        return section_data_map, section_accuracy_map

    except Exception as e:
        logging.error(f"🔥 Error in select_or_increment_model: {e}")
        raise

def predict_demand(model_bundle, trained_with: str, task_id: str):
    """
    Generate predictions and accuracy metrics for demand forecasting.

    Args:
        model_bundle: dict containing trained model, features, train_df etc.
        trained_with: "pandas" or "automl"
        task_id: for logging progress
    Returns:
        section_data_map, section_accuracy_map
    """
    try:
        # --- Common: Fetch model ---
        print(trained_with)
        if trained_with == "automl":
            model = model_bundle.get("model")
            train_df = model_bundle.get("train_df")
            features = model_bundle.get("features")
            target_col = "ACTUAL_CONSUMPTION"
        else:
            model = model_bundle["model"]
            train_df = model_bundle["train_df"]
            features = model_bundle["features"]
            target_col = model_bundle["target_col"]

        section_id = train_df["SECTION_ID"].iloc[0]

        # --- Training metrics ---
        X_train = train_df[features]
        y_train = train_df[target_col]

        if trained_with == "automl":
            # PyCaret predict
            from pycaret.regression import predict_model
            preds_train_df = predict_model(model, data=train_df)
            preds_train = preds_train_df['prediction_label']
        else:
            preds_train = model.predict(X_train)

        r2 = r2_score(y_train, preds_train)
        rmse = np.sqrt(mean_squared_error(y_train, preds_train))
        logging.info(f"Training metrics: R2={r2:.4f}, RMSE={rmse:.4f}")

         #Section-wise accuracy map
        section_accuracy_map = {
            section_id: {
                "MODEL_VERSION": f"Demand_{train_df['BILLMONTH'].max()}_{train_df['BILLYEAR'].max()}",
                "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
                "TRAINED_WITH": trained_with,
                "R2_SCORE": round(r2, 4),
                "RMSE": round(rmse, 4),
            }
        }
        tracker_log_and_progress(task_id, f"Section {section_id} - R²: {r2:.4f}, RMSE: {rmse:.4f}")

        # --- Future Prediction ---
        last_year = train_df["BILLYEAR"].max()
        last_month = train_df.loc[train_df["BILLYEAR"] == last_year, "BILLMONTH"].max()
        last_date = pd.to_datetime(f"{last_year}-{last_month:02d}-01")

        start_date = last_date - pd.DateOffset(months=5)
        end_date = last_date + pd.DateOffset(months=6)

        # Fetch future weather data
        df_weather = fetch_demand_future_weather_data(
            task_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        future_months = pd.date_range(start=start_date, end=end_date, freq="MS")

        # Build input for prediction
        future_df, section_accuracy_map = generate_future_prediction_input_avghistory(
            train_df, df_weather, future_months, features, model
        )

        # --- Generate section-wise output ---
        section_data_map = {}
        for sec_id in future_df["SECTION_ID"].unique():
            section_data_map[sec_id] = [
                {
                    "BILLMONTH": int(row["BILLMONTH"]) if pd.notnull(row["BILLMONTH"]) else None,
                    "BILLYEAR": int(row["BILLYEAR"]) if pd.notnull(row["BILLYEAR"]) else None,
                    "OFFICE_ID": str(row["SECTION_ID"]).strip() if pd.notnull(row["SECTION_ID"]) else None,
                    "CONSUMPTION": safe_float(row.get("ACTUAL_CONSUMPTION", 0)),
                    "Predicted_kWh": safe_float(row.get("PREDICTED_VALUE", 0)),
                    "MODEL_NAME": "automl_model" if trained_with=="automl" else "modeldemandfromUIavghistorylatest",
                    "VERSION": "1",
                }
                for _, row in future_df[future_df["SECTION_ID"] == sec_id].iterrows()
            ]

        return section_data_map, section_accuracy_map

    except Exception as e:
        logging.error(f"🔥 Error in demand prediction: {e}")
        raise


@shared_task
def select_or_increment_model(df, df_weather, task_id, model_name: str):
    try:
        # --- Decide training approach ---
        if model_name == "AutoML":
            logging.info("Full training using AutoML")
            tracker_log_and_progress(task_id, "Full training using AutoML")
            model_bundle = train_with_automl(df, df_weather, model_name)
            trained_with = "automl"
        else:
            logging.info("Full training using Pandas")
            tracker_log_and_progress(task_id, "Full training using Pandas")
            model_bundle = train_with_pandas(df, df_weather, model_name)
            trained_with = "pandas"

        # --- Extract common objects ---
        model = model_bundle.get("model")
        train_df = model_bundle.get("train_df")
        features = model_bundle.get("features")
        le_section_id= model_bundle.get("le_section_id")
        target_col = "ACTUAL_CONSUMPTION" if trained_with == "automl" else model_bundle.get("target_col")

        # --- Section-wise metrics ---
        section_accuracy_map = {}
        if trained_with == "automl":
            from pycaret.regression import predict_model
            section_accuracy_map = {}
            for sec_id, group in train_df.groupby("SECTION_ID"):
                preds_df = predict_model(model, data=group)  # full DataFrame
                y_true = group[target_col]
                y_pred = preds_df["prediction_label"]
                r2_sec = r2_score(y_true, y_pred)
                rmse_sec = np.sqrt(mean_squared_error(y_true, y_pred))
                section_accuracy_map[sec_id] = {
                    "MODEL_VERSION": f"Demand_{train_df['BILLMONTH'].max()}_{train_df['BILLYEAR'].max()}",
                    "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
                    "TRAINED_WITH": trained_with,
                    "R2_SCORE": round(r2_sec, 4),
                    "RMSE": round(rmse_sec, 4),
                }
                tracker_log_and_progress(task_id, f"Section {sec_id} - R²: {r2_sec:.4f}, RMSE: {rmse_sec:.4f}")
        else:
            for sec_id, group in train_df.groupby("SECTION_ID"):
                y_true = group[target_col]
                y_pred = model.predict(group[features])
                r2_sec = r2_score(y_true, y_pred)
                rmse_sec = np.sqrt(mean_squared_error(y_true, y_pred))
                section_accuracy_map[sec_id] = {
                    "MODEL_VERSION": f"Demand_{train_df['BILLMONTH'].max()}_{train_df['BILLYEAR'].max()}",
                    "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
                    "TRAINED_WITH": trained_with,
                    "R2_SCORE": round(r2_sec, 4),
                    "RMSE": round(rmse_sec, 4),
                }
                tracker_log_and_progress(task_id, f"Section {sec_id} - R²: {r2_sec:.4f}, RMSE: {rmse_sec:.4f}")

        # --- Future Prediction (last 5 months → next 6 months) ---
        last_year = train_df["BILLYEAR"].max()
        last_month = train_df.loc[train_df["BILLYEAR"] == last_year, "BILLMONTH"].max()
        last_date = pd.to_datetime(f"{last_year}-{last_month:02d}-01")

        start_date = last_date - pd.DateOffset(months=5)
        end_date   = last_date + pd.DateOffset(months=6)

        df_weather = fetch_demand_future_weather_data(
            task_id,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        future_months = pd.date_range(start=start_date, end=end_date, freq="MS")

        future_df = generate_future_prediction_input_avghistory(
            train_df, df_weather, future_months, features, model,le_section_id
        )

        # --- Build section-wise prediction output ---
        section_data_map = {
            sec_id: [
                {
                    "BILLMONTH": int(row["BILLMONTH"]) if pd.notnull(row["BILLMONTH"]) else None,
                    "BILLYEAR": int(row["BILLYEAR"]) if pd.notnull(row["BILLYEAR"]) else None,
                    "OFFICE_ID": str(row["SECTION_ID"]).strip() if pd.notnull(row["SECTION_ID"]) else None,
                    "CONSUMPTION": safe_float(row.get("ACTUAL_CONSUMPTION", 0)),
                    "Predicted_kWh": safe_float(row.get("PREDICTED_VALUE", 0)),
                    "MODEL_NAME": "automl_model" if trained_with == "automl" else "modeldemandfromUIavghistorylatest",
                    "VERSION": "1",
                }
                for _, row in future_df[future_df["SECTION_ID"] == sec_id].iterrows()
            ]
            for sec_id in future_df["SECTION_ID"].unique()
        }

        return section_data_map, section_accuracy_map

    except Exception as e:
        logging.error(f"🔥 Error in select_or_increment_model: {e}")
        raise

@shared_task
def select_or_increment_model_old(df, df_weather, task_id, model_name: str):
    try:
        # --- Decide training approach ---
        if model_name == "AutoML":
            logging.info("Full training using AutoML")
            tracker_log_and_progress(task_id, "Full training using AutoML")
            model_bundle = train_with_automl(df, df_weather, model_name)
            trained_with = "automl"
        else:
            logging.info("Full training using Pandas")
            tracker_log_and_progress(task_id, "Full training using Pandas")
            model_bundle = train_with_pandas(df, df_weather, model_name)
            trained_with = "pandas"

        if trained_with == "automl":
            # AutoML flow
            model = model_bundle.get("model")
            train_df = model_bundle.get("train_df")
            features = model_bundle.get("features")
            target_col = "ACTUAL_CONSUMPTION"
            le= model_bundle.get("label_encoder")

        else:
            # Pandas flow
            model = model_bundle["model"]
            train_df = model_bundle["train_df"]
            features = model_bundle["features"]
            target_col = model_bundle["target_col"]
            section_id = train_df["SECTION_ID"].iloc[0]
            # --- Training Metrics ---
            X = train_df[features]
            y = train_df[target_col]
            preds_train = model.predict(X)

            r2 = r2_score(y, preds_train)
            rmse = np.sqrt(mean_squared_error(y, preds_train))
            logging.info(f"Training metrics: R2={r2:.4f}, RMSE={rmse:.4f}")

            # Section-wise metrics
            section_accuracy_map = {}
            for section_id, group in train_df.groupby("SECTION_ID"):
                y_true = group[target_col]
                y_pred = model.predict(group[features])
                r2_sec = r2_score(y_true, y_pred)
                rmse_sec = np.sqrt(mean_squared_error(y_true, y_pred))
                section_accuracy_map[section_id] = {
                    "MODEL_VERSION": f"Demand_{train_df['BILLMONTH'].max()}_{train_df['BILLYEAR'].max()}",
                    "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
                    "TRAINED_WITH": trained_with,
                    "R2_SCORE": round(r2_sec, 4),
                    "RMSE": round(rmse_sec, 4),
                }
                tracker_log_and_progress(task_id, f"Section {section_id} - R²: {r2_sec:.4f}, RMSE: {rmse_sec:.4f}")
            #section_accuracy_map = {
            #    section_id: {
            #        "MODEL_VERSION": f"Demand_{train_df['BILLMONTH'].max()}_{train_df['BILLYEAR'].max()}",
            #        "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
            #        "TRAINED_WITH": trained_with,
            #        "R2_SCORE": round(r2, 4),
            #        "RMSE": round(rmse, 4),
            #    }
            #}
            #tracker_log_and_progress(task_id, f"Section {section_id} - R²: {r2:.4f}, RMSE: {rmse:.4f}")

            # --- Future Prediction (last 5 months → next 6 months) ---
        last_year = train_df["BILLYEAR"].max()
        last_month = train_df.loc[train_df["BILLYEAR"] == last_year, "BILLMONTH"].max()
        last_date = pd.to_datetime(f"{last_year}-{last_month:02d}-01")

        start_date = last_date - pd.DateOffset(months=5)
        end_date   = last_date + pd.DateOffset(months=6)

        df_weather = fetch_demand_future_weather_data(task_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        future_months = pd.date_range(start=start_date, end=end_date, freq="MS")

        future_df = generate_future_prediction_input_avghistory(train_df, df_weather, future_months, features, model,le)

            # Build section-wise output
        section_data_map = {
            section_id: [
                {
                    "BILLMONTH": int(row["BILLMONTH"]) if pd.notnull(row["BILLMONTH"]) else None,
                    "BILLYEAR": int(row["BILLYEAR"]) if pd.notnull(row["BILLYEAR"]) else None,
                    "OFFICE_ID": str(row["SECTION_ID"]).strip() if pd.notnull(row["SECTION_ID"]) else None,
                    "CONSUMPTION": safe_float(row.get("ACTUAL_CONSUMPTION", 0)),
                    "Predicted_kWh": safe_float(row.get("PREDICTED_VALUE", 0)),
                    "MODEL_NAME": "modeldemandfromUIavghistorylatest",
                    "VERSION": "1",
                }
                for _, row in future_df[future_df["SECTION_ID"] == section_id].iterrows()
           ]
           for sec_id in future_df["SECTION_ID"].unique()
        }

        return section_data_map, section_accuracy_map

    except Exception as e:
        logging.error(f"🔥 Error in select_or_increment_model: {e}")
        raise


def fetch_demand_future_weather_data(task_id: str, start_date: str, end_date: str):
    # Get query template from property/config
    base_query = get_query_by_name("WEATHER_MASTER_FUTURE_DATES_QUERY")

    # Inject parameters
    query = base_query.format(start_date=start_date, end_date=end_date)

    df = fetch_data_sync(query)  # Your existing synchronous fetch
    if df is not None:
        logging.info(f"Fetched {len(df)} rows of weather data for {start_date} → {end_date}")
        tracker_log_and_progress(task_id, f"Fetched {len(df)} rows of weather data for {start_date} → {end_date}")
    else:
        logging.warning("No weather data returned")
        tracker_log_and_progress(task_id, f"❌ No data fetched.")
        df = pd.DataFrame()
    return df



def generate_future_prediction_input_avghistory_new(
    train_df, df_weather, future_months, features, model, trained_with="pandas"
):
    """
    Hybrid forecast generator with smoothing and section-wise metrics:
    - Autoregressive with rolling mean for stability
    - Blended with historical monthly average
    - Includes actual consumption if available
    - Weather lag and rolling features supported
    - Returns future predictions + section metrics + metadata
    """
    import numpy as np
    import pandas as pd
    import logging
    from collections import defaultdict
    from sklearn.metrics import r2_score, mean_squared_error
    from datetime import date

    alpha = 0.85
    rolling_window = 3
    future_dfs = []
    section_accuracy_map = {}

    # --- Initialize rolling consumption per section ---
    rolling_prev = defaultdict(list)
    for sec, group in train_df.groupby("SECTION_ID"):
        rolling_prev[sec] = group["ACTUAL_CONSUMPTION"].dropna().tolist()[-rolling_window:]

    # --- Prepare weather ---
    if "BILLYEAR" not in df_weather.columns or "BILLMONTH" not in df_weather.columns:
        if "WEATHER_DATE" in df_weather.columns:
            df_weather["WEATHER_DATE"] = pd.to_datetime(df_weather["WEATHER_DATE"])
            df_weather["BILLYEAR"] = df_weather["WEATHER_DATE"].dt.year
            df_weather["BILLMONTH"] = df_weather["WEATHER_DATE"].dt.month
        else:
            raise ValueError("df_weather must have either WEATHER_DATE or BILLYEAR/BILLMONTH")

    monthly_weather = df_weather.groupby(["BILLYEAR", "BILLMONTH"]).mean(numeric_only=True).reset_index()
    weather_cols = ["TEMPERATURE", "HUMIDITY", "RAIN_CHANCE", "PRESSURE", "CLOUD"]

    weather_history = {
        sec: {col: list(train_df[train_df["SECTION_ID"] == sec][col].dropna().values) for col in weather_cols}
        for sec in train_df["SECTION_ID"].unique()
    }

    # Precompute section monthly averages
    section_month_avg = train_df.groupby(["SECTION_ID", "BILLMONTH"])["ACTUAL_CONSUMPTION"].mean().to_dict()

    # --- Generate future predictions month by month ---
    for dt in future_months:
        year, month = dt.year, dt.month

        month_weather = monthly_weather[
            (monthly_weather["BILLYEAR"] == year) & (monthly_weather["BILLMONTH"] == month)
        ]
        if month_weather.empty:
            month_weather = pd.DataFrame([{col: 0 for col in weather_cols + ["BILLYEAR", "BILLMONTH"]}])

        rows = []
        for sec in train_df["SECTION_ID"].unique():
            last_row = train_df[train_df["SECTION_ID"] == sec].iloc[-1].copy()
            row = last_row.copy()

            row["BILLYEAR"] = year
            row["BILLMONTH"] = month

            # Inject actual if exists
            actual_available = (train_df["SECTION_ID"] == sec) & \
                               (train_df["BILLYEAR"] == year) & \
                               (train_df["BILLMONTH"] == month)
            row["ACTUAL_CONSUMPTION"] = train_df.loc[actual_available, "ACTUAL_CONSUMPTION"].values[0] \
                                        if train_df[actual_available].shape[0] > 0 else None

            # Rolling autoregressive previous consumption
            row["Previous_month_Consumption"] = np.mean(rolling_prev[sec][-rolling_window:]) if rolling_prev[sec] else 0

            # Section monthly average
            row["Section_Monthly_Avg"] = section_month_avg.get((sec, month), row.get("Section_Monthly_Avg", 0))
            row["Consumption_vs_Avg"] = row["Previous_month_Consumption"] - row["Section_Monthly_Avg"]

            # Weather + lag/rolling features
            for col in weather_cols:
                val = month_weather[col].values[0] if col in month_weather.columns else 0
                row[col] = val
                weather_history[sec][col].append(val)

                for lag in [1, 2, 3]:
                    key = f"{col}_lag{lag}"
                    row[key] = weather_history[sec][col][-lag-1] if len(weather_history[sec][col]) > lag else 0

                for window in [2, 3]:
                    key = f"{col}_roll{window}"
                    row[key] = np.mean(weather_history[sec][col][-window:]) if len(weather_history[sec][col]) >= window else 0

            rows.append(row)

        fdf = pd.DataFrame(rows)

        # --- Predict ---
        preds = model.predict(fdf[features])

        # Hybrid blending: autoregressive + historical monthly average
        blended_preds = []
        for sec, pred in zip(fdf["SECTION_ID"], preds):
            hist_avg = section_month_avg.get((sec, month), pred)
            blended_preds.append(alpha * pred + (1 - alpha) * hist_avg)

        fdf["PREDICTED_VALUE"] = blended_preds

        # Update rolling_prev
        for sec, pred in zip(fdf["SECTION_ID"], blended_preds):
            rolling_prev[sec].append(pred)
            if len(rolling_prev[sec]) > rolling_window:
                rolling_prev[sec] = rolling_prev[sec][-rolling_window:]

        # --- Compute section metrics AFTER features exist ---
        for sec in fdf["SECTION_ID"].unique():
            sec_rows = fdf[fdf["SECTION_ID"] == sec]
            X_sec = sec_rows[features]
            y_sec = sec_rows["ACTUAL_CONSUMPTION"]

            try:
                if trained_with == "automl":
                    from pycaret.regression import predict_model
                    preds_sec_df = predict_model(model, data=X_sec)
                    preds_sec = preds_sec_df["Label"]
                else:
                    preds_sec = model.predict(X_sec)

                r2 = r2_score(y_sec, preds_sec)
                rmse = np.sqrt(mean_squared_error(y_sec, preds_sec))
            except Exception as e:
                logging.warning(f"Metrics computation failed for section {sec}: {e}")
                r2 = None
                rmse = None

            # --- Section accuracy map ---
            section_accuracy_map[sec] = {
                "MODEL_VERSION": f"Demand_{fdf['BILLMONTH'].max()}_{fdf['BILLYEAR'].max()}",
                "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
                "TRAINED_WITH": trained_with,
                "R2_SCORE": round(r2, 4) if r2 is not None else None,
                "RMSE": round(rmse, 4) if rmse is not None else None,
            }

        future_dfs.append(fdf)

    # Return combined future predictions + section metrics
    return pd.concat(future_dfs, ignore_index=True), section_accuracy_map


from collections import deque
import numpy as np
import pandas as pd
import logging

def generate_future_prediction_input_avghistory_pandas(
    train_df, df_weather, future_months, features, model
):
    """
    Hybrid forecast generator with smoothing:
    - Autoregressive with rolling mean for stability
    - Blended with historical monthly average
    - Includes actual consumption if available
    - Weather lag and rolling features supported
    """

    import numpy as np
    import pandas as pd
    from collections import defaultdict

    alpha = 0.85
    rolling_window = 3
    future_dfs = []
    print("enter into generate_future_prediction_input_avghistory")
    # Initialize rolling_prev map per section
    rolling_prev = defaultdict(list)
    for sec, group in train_df.groupby("SECTION_ID"):
        rolling_prev[sec] = group["ACTUAL_CONSUMPTION"].dropna().tolist()[-rolling_window:]

    # Precompute section monthly averages (year + month to avoid collisions)
    section_month_avg = (
        train_df.groupby(["SECTION_ID", "BILLYEAR", "BILLMONTH"])["ACTUAL_CONSUMPTION"]
        .mean()
        .to_dict()
    )

    # Ensure BILLYEAR & BILLMONTH exist in weather
    if "BILLYEAR" not in df_weather.columns or "BILLMONTH" not in df_weather.columns:
        if "WEATHER_DATE" in df_weather.columns:
            df_weather["WEATHER_DATE"] = pd.to_datetime(df_weather["WEATHER_DATE"])
            df_weather["BILLYEAR"] = df_weather["WEATHER_DATE"].dt.year
            df_weather["BILLMONTH"] = df_weather["WEATHER_DATE"].dt.month
        else:
            raise ValueError("df_weather must have either WEATHER_DATE or BILLYEAR/BILLMONTH")

    # Pre-aggregate weather monthly
    monthly_weather = (
        df_weather.groupby(["BILLYEAR", "BILLMONTH"]).mean(numeric_only=True).reset_index()
    )

    weather_cols = ["TEMPERATURE", "HUMIDITY", "RAIN_CHANCE", "PRESSURE", "CLOUD"]

    # Track weather history for lag/rolling features
    weather_history = {
        sec: {col: list(train_df[train_df["SECTION_ID"] == sec][col].dropna().values) for col in weather_cols}
        for sec in train_df["SECTION_ID"].unique()
    }

    for dt in future_months:
        year, month = dt.year, dt.month

        # Get monthly weather
        month_weather = monthly_weather[
            (monthly_weather["BILLYEAR"] == year) & (monthly_weather["BILLMONTH"] == month)
        ]
        if month_weather.empty:
            month_weather = pd.DataFrame(
                [{col: 0 for col in weather_cols + ["BILLYEAR", "BILLMONTH"]}]
            )

        rows = []
        for sec in train_df["SECTION_ID"].unique():
            last_row = train_df[train_df["SECTION_ID"] == sec].iloc[-1].copy()
            row = last_row.copy()

            row["BILLYEAR"] = year
            row["BILLMONTH"] = month
            # Inject actual if exists
            # ✅ Inject actual if exists
            actual_available = (
                (train_df["SECTION_ID"] == sec) &
                (train_df["BILLYEAR"] == year) &
                (train_df["BILLMONTH"] == month)
            )
            if train_df[actual_available].shape[0] > 0:
                row["ACTUAL_CONSUMPTION"] = train_df.loc[actual_available, "ACTUAL_CONSUMPTION"].values[0]
            else:
                row["ACTUAL_CONSUMPTION"] = None

            # Previous consumption: rolling autoregressive mean
            row["Previous_month_Consumption"] = (
                np.mean(rolling_prev[sec][-rolling_window:]) if rolling_prev[sec] else 0
            )

            # Section monthly average
            row["Section_Monthly_Avg"] = section_month_avg.get((sec, year, month), 0)

            # Consumption vs average
            row["Consumption_vs_Avg"] = (
                row["Previous_month_Consumption"] - row["Section_Monthly_Avg"]
            )

            # Weather base + lag/rolling
            for col in weather_cols:
                val = month_weather[col].values[0] if col in month_weather.columns else 0
                row[col] = val
                weather_history[sec][col].append(val)

                # Lag features
                for lag in [1, 2, 3]:
                    row[f"{col}_lag{lag}"] = (
                        weather_history[sec][col][-lag - 1]
                        if len(weather_history[sec][col]) > lag
                        else 0
                    )

                # Rolling means
                for window in [2, 3]:
                    row[f"{col}_roll{window}"] = (
                        np.mean(weather_history[sec][col][-window:])
                        if len(weather_history[sec][col]) >= window
                        else 0
                    )

            rows.append(row)

        fdf = pd.DataFrame(rows)
        print("future df")

        features_to_use = [f for f in features if f != "BILLDATE"]

        # Only select columns your model expects
        fdf_model = fdf[features_to_use]

        # Check what features model was trained on
        # Check what features model was trained on
        if hasattr(model, "feature_names_in_"):
            fnames = model.feature_names_in_
            if isinstance(fnames, (list, tuple)):
                model_features = list(fnames)
            else:
                model_features = fnames.tolist()
            print("🔎 Model was trained on:", model_features)

        elif hasattr(model, "features"):  # for PyCaret wrapped models
            model_features = list(model.features)
            print("🔎 Model was trained on:", model_features)

        else:
            try:
                from pycaret.regression import get_config
                model_features = get_config("X_train").columns.tolist()
                print("🔎 Model was trained on:", model_features)
            except:
                model_features = features  # fallback
                print("⚠️ Using fallback features:", model_features)

        # Now check alignment before prediction
        print("🔎 fdf_model columns:", fdf_model.columns.tolist())

        missing_cols = set(model_features) - set(fdf_model.columns)
        extra_cols   = set(fdf_model.columns) - set(model_features)

        print("❌ Missing cols:", missing_cols)
        print("⚠️ Extra cols:", extra_cols)
        # Predict
        preds = model.predict(fdf_model)
        print("after prediction")
        # Hybrid blending: autoregressive (rolling) + historical average
        blended_preds = []
        for sec, pred in zip(fdf["SECTION_ID"], preds):
            hist_avg = section_month_avg.get((sec, year, month), pred)
            blended_preds.append(alpha * pred + (1 - alpha) * hist_avg)

        fdf["PREDICTED_VALUE"] = blended_preds

        # Update rolling_prev for autoregressive smoothing
        for sec, pred in zip(fdf["SECTION_ID"], blended_preds):
            rolling_prev[sec].append(pred)
            if len(rolling_prev[sec]) > rolling_window:
                rolling_prev[sec] = rolling_prev[sec][-rolling_window:]

        # Append results
        future_dfs.append(fdf)

    return pd.concat(future_dfs, ignore_index=True)


def generate_future_prediction_input_avghistory(
    train_df, df_weather, future_months, features, model, le_section_id
):
    """
    Hybrid forecast generator with smoothing:
    - Autoregressive with rolling mean for stability
    - Blended with historical monthly average
    - Includes actual consumption only for reference (NOT in model input)
    - Weather lag and rolling features supported
    - SECTION_ID_En uses the same LabelEncoder as training
    """

    import numpy as np
    import pandas as pd
    from collections import defaultdict

    alpha = 0.85
    rolling_window = 3
    future_dfs = []

    # Initialize rolling_prev map per section
    rolling_prev = defaultdict(list)
    for sec, group in train_df.groupby("SECTION_ID"):
        rolling_prev[sec] = group["ACTUAL_CONSUMPTION"].dropna().tolist()[-rolling_window:]

    # Precompute section monthly averages
    section_month_avg = (
        train_df.groupby(["SECTION_ID", "BILLYEAR", "BILLMONTH"])["ACTUAL_CONSUMPTION"]
        .mean()
        .to_dict()
    )

    # Ensure BILLYEAR & BILLMONTH exist in weather
    if "BILLYEAR" not in df_weather.columns or "BILLMONTH" not in df_weather.columns:
        if "WEATHER_DATE" in df_weather.columns:
            df_weather["WEATHER_DATE"] = pd.to_datetime(df_weather["WEATHER_DATE"])
            df_weather["BILLYEAR"] = df_weather["WEATHER_DATE"].dt.year
            df_weather["BILLMONTH"] = df_weather["WEATHER_DATE"].dt.month
        else:
            raise ValueError("df_weather must have either WEATHER_DATE or BILLYEAR/BILLMONTH")

    # Pre-aggregate weather monthly
    monthly_weather = (
        df_weather.groupby(["BILLYEAR", "BILLMONTH"]).mean(numeric_only=True).reset_index()
    )
    weather_cols = ["TEMPERATURE", "HUMIDITY", "RAIN_CHANCE", "PRESSURE", "CLOUD"]

    # Track weather history for lag/rolling features
    weather_history = {
        sec: {col: list(train_df[train_df["SECTION_ID"] == sec][col].dropna().values) for col in weather_cols}
        for sec in train_df["SECTION_ID"].unique()
    }

    for dt in future_months:
        year, month = dt.year, dt.month

        # Get monthly weather
        month_weather = monthly_weather[
            (monthly_weather["BILLYEAR"] == year) & (monthly_weather["BILLMONTH"] == month)
        ]
        if month_weather.empty:
            month_weather = pd.DataFrame(
                [{col: 0 for col in weather_cols + ["BILLYEAR", "BILLMONTH"]}]
            )

        rows = []
        for sec in train_df["SECTION_ID"].unique():
            last_row = train_df[train_df["SECTION_ID"] == sec].iloc[-1].copy()
            row = last_row.copy()

            row["BILLYEAR"] = year
            row["BILLMONTH"] = month

            # Keep actual consumption for reference only
            actual_available = (train_df["SECTION_ID"] == sec) & \
                               (train_df["BILLYEAR"] == year) & \
                               (train_df["BILLMONTH"] == month)
            row["ACTUAL_CONSUMPTION"] = (
                train_df.loc[actual_available, "ACTUAL_CONSUMPTION"].values[0]
                if train_df[actual_available].shape[0] > 0
                else None
            )

            # Previous consumption: rolling autoregressive mean
            row["Previous_month_Consumption"] = (
                np.mean(rolling_prev[sec][-rolling_window:]) if rolling_prev[sec] else 0
            )

            # Section monthly average
            row["Section_Monthly_Avg"] = section_month_avg.get((sec, year, month), 0)

            # Consumption vs average
            row["Consumption_vs_Avg"] = (
                row["Previous_month_Consumption"] - row["Section_Monthly_Avg"]
            )

            # Weather features + lags/rolling
            for col in weather_cols:
                val = month_weather[col].values[0] if col in month_weather.columns else 0
                row[col] = val
                weather_history[sec][col].append(val)

                # Lag features
                for lag in [1, 2, 3]:
                    row[f"{col}_lag{lag}"] = (
                        weather_history[sec][col][-lag - 1]
                        if len(weather_history[sec][col]) > lag
                        else 0
                    )

                # Rolling means
                for window in [2, 3]:
                    row[f"{col}_roll{window}"] = (
                        np.mean(weather_history[sec][col][-window:])
                        if len(weather_history[sec][col]) >= window
                        else 0
                    )

            rows.append(row)

        fdf = pd.DataFrame(rows)
        print("future df before alignment")

        # -------------------------------
        # Align features with trained model
        # -------------------------------

        # Get model features safely
        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
        elif hasattr(model, "features"):
            model_features = list(model.features)
        else:
            try:
                from pycaret.regression import get_config
                model_features = get_config("X_train").columns.tolist()
            except:
                model_features = [c for c in fdf.columns if c != "ACTUAL_CONSUMPTION"]

        # ✅ Ensure target column is removed
        if "ACTUAL_CONSUMPTION" in model_features:
            model_features.remove("ACTUAL_CONSUMPTION")

        # Start with prediction features only
        fdf_model = fdf.copy()
        if "ACTUAL_CONSUMPTION" in fdf_model.columns:
            fdf_model = fdf_model.drop(columns=["ACTUAL_CONSUMPTION"])

        # Inject missing columns
        for col in model_features:
            if col not in fdf_model.columns:
                if col == "BILLDATE":
                    fdf_model[col] = pd.NaT
                elif col in ["SECTION_ID", "SECTION_ID_En"]:
                    fdf_model[col] = None
                else:
                    fdf_model[col] = 0

        # Keep only aligned features
        fdf_model = fdf_model[model_features]

        # Encode SECTION_ID if needed
        if le_section_id is not None and "SECTION_ID_En" in fdf_model.columns:
            fdf_model["SECTION_ID_En"] = le_section_id.transform(fdf["SECTION_ID"].astype(str))

        print("🔎 Model was trained on:", model_features)
        print("🔎 fdf_model columns aligned for prediction:", fdf_model.columns.tolist())
        print("🔎 fdf_model head:\n", fdf_model.head())

        # Predict
        preds = model.predict(fdf_model)
        print("after prediction")

        # Hybrid blending
        blended_preds = []
        for sec, pred in zip(fdf["SECTION_ID"], preds):
            hist_avg = section_month_avg.get((sec, year, month), pred)
            blended_preds.append(alpha * pred + (1 - alpha) * hist_avg)

        fdf["PREDICTED_VALUE"] = blended_preds

        # Update rolling_prev
        for sec, pred in zip(fdf["SECTION_ID"], blended_preds):
            rolling_prev[sec].append(pred)
            if len(rolling_prev[sec]) > rolling_window:
                rolling_prev[sec] = rolling_prev[sec][-rolling_window:]

        future_dfs.append(fdf)

    return pd.concat(future_dfs, ignore_index=True)


def generate_future_prediction_input(train_df, df_weather, future_months, features, model):
    """
    Iteratively generate future predictions for each section.
    Includes autoregressive consumption + weather lag/rolling features.
    """
    future_dfs = []

    # Initialize prev_consumption_map
    prev_consumption_map = {
        sec: group["ACTUAL_CONSUMPTION"].iloc[-1]
        for sec, group in train_df.groupby("SECTION_ID")
    }

    # Precompute section monthly averages
    section_month_avg = (
        train_df.groupby(["SECTION_ID", "BILLMONTH"])["ACTUAL_CONSUMPTION"]
        .mean()
        .to_dict()
    )

    # Ensure weather has year/month
    if "BILLYEAR" not in df_weather.columns or "BILLMONTH" not in df_weather.columns:
        if "WEATHER_DATE" in df_weather.columns:
            df_weather["WEATHER_DATE"] = pd.to_datetime(df_weather["WEATHER_DATE"])
            df_weather["BILLYEAR"] = df_weather["WEATHER_DATE"].dt.year
            df_weather["BILLMONTH"] = df_weather["WEATHER_DATE"].dt.month
        else:
            raise ValueError("df_weather must have either WEATHER_DATE or BILLYEAR/BILLMONTH")

    # Monthly weather
    monthly_weather = (
        df_weather.groupby(["BILLYEAR", "BILLMONTH"]).mean(numeric_only=True).reset_index()
    )

    # Track weather history per section (for lags)
    weather_cols = ['TEMPERATURE', 'HUMIDITY', 'RAIN_CHANCE', 'PRESSURE', 'CLOUD']
    weather_history = {
        sec: {col: list(train_df[train_df["SECTION_ID"] == sec][col].dropna().values) 
              for col in weather_cols}
        for sec in train_df["SECTION_ID"].unique()
    }

    for dt in future_months:
        year, month = dt.year, dt.month
        month_weather = monthly_weather[
            (monthly_weather["BILLYEAR"] == year) & (monthly_weather["BILLMONTH"] == month)
        ]
        if month_weather.empty:
            month_weather = pd.DataFrame([{col: 0 for col in weather_cols + ["BILLYEAR","BILLMONTH"]}])

        rows = []
        for sec in train_df["SECTION_ID"].unique():
            last_row = train_df[train_df["SECTION_ID"] == sec].iloc[-1].copy()
            row = last_row.copy()

            row["BILLYEAR"] = year
            row["BILLMONTH"] = month
            # Actual consumption if exists
            actual_available = (
                (train_df["SECTION_ID"] == sec) &
                (train_df["BILLYEAR"] == year) &
                (train_df["BILLMONTH"] == month)
            )
            if train_df[actual_available].shape[0] > 0:
                row["ACTUAL_CONSUMPTION"] = train_df.loc[actual_available, "ACTUAL_CONSUMPTION"].values[0]
            else:
                row["ACTUAL_CONSUMPTION"] = None

            row["Previous_month_Consumption"] = prev_consumption_map.get(sec, 0)
            row["Section_Monthly_Avg"] = section_month_avg.get((sec, month), row["Section_Monthly_Avg"])
            row["Consumption_vs_Avg"] = row["Previous_month_Consumption"] - row["Section_Monthly_Avg"]

            # Base weather
            for col in weather_cols:
                val = month_weather[col].values[0] if col in month_weather.columns else 0
                row[col] = val
                weather_history[sec][col].append(val)

                # Add lag features
                for lag in [1, 2, 3]:
                    key = f"{col}_lag{lag}"
                    row[key] = weather_history[sec][col][-lag-1] if len(weather_history[sec][col]) > lag else 0

                # Add rolling means
                for window in [2, 3]:
                    key = f"{col}_roll{window}"
                    row[key] = np.mean(weather_history[sec][col][-window:]) if len(weather_history[sec][col]) >= window else 0

            rows.append(row)

        fdf = pd.DataFrame(rows)

        # Predict
        preds = model.predict(fdf[features])
        fdf["PREDICTED_VALUE"] = preds

        # Update prev_consumption_map
        for sec, pred in zip(fdf["SECTION_ID"], fdf["PREDICTED_VALUE"]):
            prev_consumption_map[sec] = pred

        # Append results
        future_dfs.append(fdf)

    return pd.concat(future_dfs, ignore_index=True)




@shared_task
def insert_accuracy_log(section_id, model_version, training_date, trained_with, r2, rmse,task_id: str):
    conn = None
    cursor = None
    try:
        conn = get_uat_db_connection()
        if conn is None:
            return None

        cursor = conn.cursor()
        training_date = datetime.strptime(training_date, "%Y-%m-%d").date()
        insert_sql = """
            INSERT INTO MIS_USER.MODEL_TRAINING_LOGS (
                SECTION_ID, MODEL_VERSION, TRAINING_DATE, TRAINED_WITH, R2_SCORE, RMSE
            )
            VALUES (:1, :2, TO_DATE(:3, 'YYYY-MM-DD'), :4, :5, :6)
        """

        cursor.execute(insert_sql, (section_id, model_version, training_date, trained_with, r2, rmse))
        conn.commit()
        logging.info(f"📘 Logged model training details for section {section_id}")
        tracker_log_and_progress(task_id,f"📘 Logged model training details for section {section_id}")
    except Exception as e:
        logging.error(f"❌ Failed to log model info for section {section_id}: {e}")
        tracker_log_and_progress(task_id,f"❌ Failed to log model info for section {section_id}: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            uat_session_pool.release(conn)

def safe_float(x):
    try:
        return float(x) if pd.notnull(x) else None
    except:
        return None

@shared_task
def insert_section_data(section_id: str, records: List[dict], task_id: str):
    print("Start insert ")
    conn = None
    cursor = None
    try:
        conn = get_uat_db_connection()
        if conn is None:
            return None

        cursor = conn.cursor()

        # ✅ Step 1: Delete previous prediction records for this section
        delete_sql = """
            DELETE FROM MIS_USER.CONSUMPTION_PREDICTION
            WHERE office_id = :office_id
        """
        cursor.execute(delete_sql, {"office_id": section_id})
        conn.commit()
        logging.info(f"🧹 Deleted previous predictions for section {section_id}")
        tracker_log_and_progress(task_id, f"🧹 Deleted previous predictions for section {section_id}")

        # ✅ Step 2: Prepare new records for insertion
        insert_sql = """
            INSERT INTO MIS_USER.CONSUMPTION_PREDICTION (
                conspt_month, conspt_year, office_id, actual, predicted, model, record_status
            )
            VALUES (:1, :2, :3, :4, :5, :6, :7)
        """

        data_tuples = [
            (
                int(rec["BILLMONTH"]) if rec.get("BILLMONTH") else None,
                int(rec["BILLYEAR"]) if rec.get("BILLYEAR") else None,
                str(rec["OFFICE_ID"]).strip() if rec.get("OFFICE_ID") else None,
                safe_float(rec.get("CONSUMPTION")),
                safe_float(rec.get("Predicted_kWh")),
                str(rec["MODEL_NAME"]).strip() if rec.get("MODEL_NAME") else None,
                "1"  # record_status
            )
            for rec in records
        ]

        # ✅ Step 3: Insert new predictions
        cursor.executemany(insert_sql, data_tuples)
        conn.commit()

        logging.info(f"✅ Inserted predicted data for section {section_id} successfully.")
        tracker_log_and_progress(task_id, f"✅ Inserted predicted data for section {section_id} successfully.")

    except Exception as e:
        logging.error(f"🔥 Error inserting predicted data for section {section_id}: {e}")
        tracker_log_and_progress(task_id, f"🔥 Error inserting predicted data for section {section_id}: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            uat_session_pool.release(conn)


