from celery import shared_task
from celery import Celery
import pandas as pd
import numpy as np
import pandas as pd
from datetime import date,datetime
import logging
from FG.services.train_newconnection_daywise import train_with_AutoML
from FG.ml_engine.train.train_daily_demand import train_with_pandas,load_data,train_with_automl,train_dl_model
from FG.core.config import MODEL_PATH,THRESHOLD_MB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import asyncio
from sklearn.preprocessing import MinMaxScaler
from typing import List
from FG.core.utils.log_and_progress import tracker_log_and_progress
import requests
from FG.core.utils.common_db_service import fetch_data_sync
from FG.core.query_loader import get_query_by_name
import mlflow
from FG.core.utils.progress_listener import ProgressListener
from FG.tasks.langchain_tasks import log_reasoning_to_langchain,save_reasoning_log_to_db
from FG.core.database import get_db_connection, session_pool
from FG.core.mlflow_config import setup_mlflow
from FG.ml_engine.inference.daily_demand.dd_dl_inference import dl_inference
from FG.ml_engine.inference.daily_demand.dd_classical_inference import classical_inference
from FG.ml_engine.inference.daily_demand.dd_automl_inference import automl_inference
from FG.ml_engine.inference.daily_demand.dd_evaluation import evaluate_section_performance


#from core.database_uat import get_uat_db_connection, uat_session_pool
logger = logging.getLogger(__name__)


celery = Celery(
    "tasks.daily_demand_tasks",
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json']
)

@shared_task(name="FG.tasks.daily_demand_tasks.schedule_train_daily_demand")
def schedule_train_daily_demand(start_date: str, end_date: str, office_id: str,model_name: str):
    logger.info("✅ Beat triggered schedule_train_daily_demand_tasks")
    # 🔁 You can dynamically calculate date range here
    #end_date = "2024-12-31"#datetime.now().strftime("%Y-%m-%d")
    #start_date = "2024-08-01"#(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")  # last 30 days
    #office_id = "803-10"  # Replace with dynamic logic or loop if needed

    url = f"http://127.0.0.1:8000/daily_demand/train/{start_date}/{end_date}/{office_id}/{model_name}"

    try:
        response = requests.post(url)
        print(f"Triggered training: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error triggering training: {e}")

@shared_task
def fetch_data_and_train(section_ids: List[str], task_id: str, start_date: str, end_date: str, model_name: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    mlflow = setup_mlflow()
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Parent run for the full job
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", nested=False) as parent_run:
        try:
            mlflow.log_params({
                "model_name": model_name,
                "start_date": start_date,
                "end_date": end_date,
                "num_sections": len(section_ids)
            })

            tracker_log_and_progress(task_id, "🚀 MLflow parent run started & LangChain monitoring initialized")

            for section_id in section_ids:
                try:
                    # 🧠 NEW: Nested MLflow run for each section
                    with mlflow.start_run(run_name=f"Section_{section_id}", nested=True):
                        log_reasoning_to_langchain.apply_async(
                            args=[task_id, f"Loading data for section {section_id}", "Data Fetching"],
                            queue="langchain_queue"
                        )

                        df, df_weather, file_size = loop.run_until_complete(
                            load_data(section_id, task_id, start_date, end_date)
                        )

                        if df.empty:
                            msg = f"⚠️ No data fetched for section: {section_id}"
                            logging.warning(msg)
                            tracker_log_and_progress(task_id, msg)
                            mlflow.log_param("training_status", "skipped")
                            continue

                        df_dict = df.to_dict(orient="records")
                        tracker_log_and_progress(task_id, f"🚀 Training started for section {section_id}")

                        section_data_map, section_accuracy_map = select_or_increment_model(
                            df_dict, df_weather, task_id, model_name, section_id
                        )

                        # --- Log metrics safely inside nested run
                        if isinstance(section_accuracy_map, dict) and section_id in section_accuracy_map:
                            metrics = section_accuracy_map[section_id]
                            mlflow.log_metrics({
                                "r2": metrics["R2_SCORE"],
                                "rmse": metrics["RMSE"]
                            })
                            mlflow.log_params({
                                "trained_with": metrics["TRAINED_WITH"],
                                "model_version": metrics["MODEL_VERSION"]
                            })

                        # --- Data insertion (unchanged)
                        if isinstance(section_data_map, dict) and section_id in section_data_map:
                            records = section_data_map[section_id]
                            insert_section_data.apply_async(
                                args=[section_id, records, task_id, model_name],
                                queue="daily_demand_queue"
                            )

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
                                    "dashboard": metrics["DASHBOARD"],
                                    "task_id": task_id,
                                },
                                queue="daily_demand_queue"
                            )

                        mlflow.log_param("training_status", "completed")
                        tracker_log_and_progress(task_id, f"✅ Completed section training for {section_id}", "completed")

                except Exception as sec_err:
                    err_msg = f"🔥 Error training section {section_id}: {sec_err}"
                    logging.error(err_msg)
                    tracker_log_and_progress(task_id, err_msg, "failed")
                    mlflow.log_param("training_status", "failed")
                    mlflow.log_param("error_message", str(sec_err))

            # ✅ Log only once at the end for global completion
            mlflow.log_param("pipeline_status", "completed")
            tracker_log_and_progress(task_id, "🏁 All section training tasks completed successfully.", "completed")

        except Exception as e:
            err_msg = f"🔥 Global training error: {e}"
            logging.error(err_msg)
            tracker_log_and_progress(task_id, err_msg, "failed")
            mlflow.log_param("pipeline_status", "failed")

        finally:
            loop.close()
            mlflow.end_run()
            tracker_log_and_progress(task_id, "🧾 MLflow run ended.")
            logging.info("Training process finished and MLflow run closed.")

@shared_task
def fetch_data_and_train_division_wise(section_ids: List[str], task_id: str, train_year: str, train_month: str, model_name: str):
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
            df_dict, df_weather,task_id, model_name,section_ids
        )

        # 3. Insert predictions + metrics
        if isinstance(section_data_map, dict) and section_data_map:
            logging.info("✅ Training complete. Starting insertion...")
            tracker_log_and_progress(task_id, "✅ Training complete. Starting insertion...")

            # Insert predicted section-wise data
            for section_id, records in section_data_map.items():
                insert_section_data.apply_async(
                    args=[section_id, records, task_id],
                    queue="daily_demand_queue"
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
                        "dashboard": metrics["DASHBOARD"]
                    },
                    queue="daily_demand_queue"
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
def select_or_increment_model(df, df_weather, task_id, model_name: str, section_id: str):
    """
    Train or incrementally update the model for a given section_id.
    Logs metrics & parameters to MLflow (nested run).
    Tracks reasoning logs via LangChain.
    """
    try:
        # 🎯 Initialize LangChain progress listener for this section
        langchain_listener = ProgressListener(task_id)
        langchain_listener.start_listening(interval=10)

        # 🔹 Start MLflow nested run for this section
        with mlflow.start_run(run_name=f"Section_{section_id}_{model_name}", nested=True):
            tracker_log_and_progress(task_id, f"⚙️ Started model training for section {section_id}")

            # --- Model Selection Logic ---
            if model_name.lower() in [ "lstm","gru", "cnn"]:
                trained_with = model_name.lower()
                tracker_log_and_progress(task_id, "⚙️ Training using LSTM")
                model_bundle = train_dl_model(df, df_weather, model_name, task_id, section_id)

            elif model_name == "AutoML":
                trained_with = "automl"
                tracker_log_and_progress(task_id, "⚙️ Training using AutoML")
                model_bundle = train_with_automl(df, df_weather, model_name)

            else:
                trained_with = "pandas"
                tracker_log_and_progress(task_id, "⚙️ Training using Pandas")
                model_bundle = train_with_pandas(df, df_weather, model_name)

            section_accuracy_map = {}

            # --- Extract model components ---
            model = model_bundle["model"]
            train_df = model_bundle["train_df"]
            features = model_bundle["features"]
            target_col = model_bundle.get("target_col", "COLLECTION")
            section_id = str(train_df["SECTION_ID"].iloc[0])

            # -----------------------------------------
            #  🔥 Unified Inference Logic (New)
            # -----------------------------------------
            if trained_with in ["lstm", "gru", "cnn"]:

                # Extract DL parameters from bundle
                window_size = model_bundle["window_size"]
                scaler = model_bundle["scaler"]        # IMPORTANT: Use training scaler
                le = model_bundle.get("le", None)      # optional for some models

                # Call central DL inference
                y_true, y_pred = dl_inference(model=model,train_df=train_df,target_col=target_col,window_size=window_size,scaler=scaler,task_id=task_id)

            elif trained_with == "automl":
                y_true, y_pred = automl_inference(model, train_df, target_col)

            else:
                # Pandas / LGBM
                y_true, y_pred = classical_inference(model, train_df, features, target_col)

            # --- Compute Metrics ---
            section_accuracy_map, r2_sec, rmse_sec = evaluate_section_performance(section_id,model_name,train_df,trained_with,y_true,y_pred)

            # --- Log to MLflow ---
            mlflow.log_params({
                "section_id": section_id,
                "trained_with": trained_with,
                "feature_count": len(features),
            })
            mlflow.log_metrics({
                "r2_score": r2_sec,
                "rmse": rmse_sec,
            })
            mlflow.log_param("training_status", "completed")

            msg = f"📊 Section {section_id}: R²={r2_sec:.4f}, RMSE={rmse_sec:.4f}"
            tracker_log_and_progress(task_id, msg)
            langchain_listener.summarize_progress(msg)

            # ✅ Send LangChain reasoning logs asynchronously
            log_reasoning_to_langchain.apply_async(
                args=[task_id, msg, "Model Summary"],
                queue="langchain_queue"
            )

            # --- Future Predictions ---
            last_date = pd.to_datetime(train_df["DAYS"].max())
            start_date = last_date
            end_date = last_date + pd.DateOffset(days=30)

            df_weather_future = fetch_demand_future_weather_data(
                task_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            future_days = pd.date_range(start=start_date + pd.Timedelta(days=1), end=end_date, freq="D")

            # Generate section-wise future predictions
            if trained_with == "automl":
                le_section_id = model_bundle.get("le_section_id")
                future_df = generate_future_prediction_input_avghistory_automl(
                    train_df, df_weather_future, future_days, features, model, le_section_id
                )
            elif trained_with in ["lstm", "gru", "cnn"]:
                future_df = generate_future_prediction_input_lstm(
                    train_df, df_weather_future, future_days, features, model, window_size, le, task_id
                )
            else:
                future_df = generate_future_prediction_input_avghistory(
                    train_df, df_weather_future, future_days, features, model
                )

            # --- Map for Database Insert ---
            section_data_map = {
                section_id: [
                    {
                        "DAY": (
                            row["DAYS"].strftime("%Y-%m-%d")
                            if "DAYS" in row and pd.notnull(row["DAYS"])
                            else None
                        ),
                        "SECTION_ID": str(row["SECTION_ID"]).strip() if pd.notnull(row["SECTION_ID"]) else None,
                        "ACTUAL_CONSUMPTION": safe_float(row.get("ACTUAL_CONSUMPTION", 0)),
                        "PREDICTED_CONSUMPTION": safe_float(row.get("PREDICTED_VALUE", 0)),
                        "MODEL_NAME": f"Daily_Demand_{model_name}_{train_df['DAYS'].max().strftime('%Y%m%d')}",
                        "VERSION": "1",
                    }
                    for _, row in future_df[future_df["SECTION_ID"] == section_id].iterrows()
                ]
            }

            mlflow.log_param("future_days_generated", len(future_days))
            tracker_log_and_progress(task_id, f"✅ Completed future prediction for section {section_id}", "completed")

            # ✅ Log reasoning to LangChain
            log_reasoning_to_langchain.apply_async(
                args=[task_id, f"✅ Completed training and prediction for section {section_id}", "Model Completion"],
                queue="langchain_queue"
            )

            return section_data_map, section_accuracy_map

    except Exception as e:
        err_msg = f"🔥 Error in select_or_incremental_model: {e}"
        logging.error(err_msg)
        tracker_log_and_progress(task_id, err_msg, "failed")

        # Safe MLflow error logging
        try:
            mlflow.log_param("training_status", "failed")
        except Exception:
            pass

        # LangChain error log
        log_reasoning_to_langchain.apply_async(
            args=[task_id, err_msg, "Model Error"],
            queue="langchain_queue"
        )
        raise

@shared_task
def select_or_increment_model_old(df, df_weather, task_id, model_name: str):
    try:
        # --- Decide training approach ---
        if model_name == "AutoML":
            logging.info("Full training using AutoML")
            tracker_log_and_progress(task_id, "Full training using AutoML")
            #model_bundle = train_with_AutoML(df)
            trained_with = "automl"
        else:
            logging.info("Full training using Pandas")
            tracker_log_and_progress(task_id, "Full training using Pandas")
            model_bundle = train_with_pandas(df, df_weather, model_name)
            trained_with = "pandas"

        if trained_with == "automl":
            # AutoML flow
            model = model_bundle.get("model")
            #feature_columns = model_bundle.get("feature_columns")

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
            #section_accuracy_map = {}
            #for section_id, group in train_df.groupby("SECTION_ID"):
            #    y_true = group[target_col]
            #    y_pred = model.predict(group[features])
            #    r2_sec = r2_score(y_true, y_pred)
            #    rmse_sec = np.sqrt(mean_squared_error(y_true, y_pred))
            #    section_accuracy_map[section_id] = {
            #        "MODEL_VERSION": f"Demand_{train_df['BILLMONTH'].max()}_{train_df['BILLYEAR'].max()}",
            #        "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
            #        "TRAINED_WITH": trained_with,
            #        "R2_SCORE": round(r2_sec, 4),
            #        "RMSE": round(rmse_sec, 4),
            #    }
            #    tracker_log_and_progress(task_id, f"Section {section_id} - R²: {r2_sec:.4f}, RMSE: {rmse_sec:.4f}")
            section_accuracy_map = {
                section_id: {
                    "MODEL_VERSION": f"Revenue_{model_name}_{train_df['REV_MONTH'].max()}_{train_df['REV_YEAR'].max()}",
                    "TRAINING_DATE": date.today().strftime("%Y-%m-%d"),
                    "TRAINED_WITH": trained_with,
                    "R2_SCORE": round(r2, 4),
                    "RMSE": round(rmse, 4),
                    "DASHBOARD":"Revenue",
                }
            }
            tracker_log_and_progress(task_id, f"Section {section_id} - R²: {r2:.4f}, RMSE: {rmse:.4f}")

            # --- Future Prediction (last 5 months → next 6 months) ---
            last_year = train_df["BILLYEAR"].max()
            last_month = train_df.loc[train_df["BILLYEAR"] == last_year, "BILLMONTH"].max()
            last_date = pd.to_datetime(f"{last_year}-{last_month:02d}-01")

            start_date = last_date - pd.DateOffset(months=5)
            end_date   = last_date + pd.DateOffset(months=6)

            df_weather = fetch_demand_future_weather_data(task_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            future_months = pd.date_range(start=start_date, end=end_date, freq="MS")

            future_df = generate_future_prediction_input_avghistory(train_df, df_weather, future_months, features, model)

            # Build section-wise output
            section_data_map = {
                section_id: [
                    {
                        "BILLMONTH": int(row["BILLMONTH"]) if pd.notnull(row["BILLMONTH"]) else None,
                        "BILLYEAR": int(row["BILLYEAR"]) if pd.notnull(row["BILLYEAR"]) else None,
                        "OFFICE_ID": str(row["SECTION_ID"]).strip() if pd.notnull(row["SECTION_ID"]) else None,
                        "CONSUMPTION": safe_float(row.get("ACTUAL_CONSUMPTION", 0)),
                        "Predicted_kWh": safe_float(row.get("PREDICTED_VALUE", 0)),
                        "MODEL_NAME": "automl_model" if trained_with == "automl" else f"Revenue_{model_name}_{train_df['REV_MONTH'].max()}_{train_df['REV_YEAR'].max()}",
                        "VERSION": "1",
                    }
                    for _, row in future_df[future_df["SECTION_ID"] == section_id].iterrows()
                ]
                #for sec_id in future_df["SECTION_ID"].unique()
            }

            return section_data_map, section_accuracy_map

    except Exception as e:
        logging.error(f"🔥 Error in select_or_increment_model: {e}")
        raise


def fetch_demand_future_weather_data(task_id: str, start_date: str, end_date: str):
    """
    Fetch weather data for given date range, log in tracker, and record metrics in MLflow.
    """
    try:
        # --- Prepare SQL ---
        base_query = get_query_by_name("WEATHER_MASTER_FUTURE_DATES_QUERY")
        query = base_query.format(start_date=start_date, end_date=end_date)

        # --- Fetch data ---
        df = fetch_data_sync(query)

        if df is not None and not df.empty:
            row_count = len(df)
            tracker_log_and_progress(task_id, f"✅ Fetched {row_count} weather records ({start_date} → {end_date})")
            logging.info(f"Fetched {row_count} rows of weather data for {start_date} → {end_date}")

            if mlflow.active_run() is not None:
                mlflow.end_run()
            # --- Log to MLflow ---
            with mlflow.start_run(run_name=f"WeatherFetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}", nested=True):
                mlflow.log_param("task_id", task_id)
                mlflow.log_param("date_range", f"{start_date} → {end_date}")
                mlflow.log_metric("weather_row_count", row_count)

            # --- Optional: LangChain reasoning log ---
            try:
                log_reasoning_to_langchain(
                    f"Weather data fetch successful for {start_date} to {end_date} with {row_count} rows."
                )
            except Exception as lc_err:
                logging.warning(f"LangChain log skipped: {lc_err}")

        else:
            tracker_log_and_progress(task_id, "⚠️ No weather data fetched for the given date range.")
            logging.warning("No weather data returned.")
            df = pd.DataFrame()

            # Log to MLflow
            if mlflow.active_run() is not None:
                mlflow.end_run()

            with mlflow.start_run(run_name="WeatherFetch_NoData", nested=True):
                mlflow.log_param("task_id", task_id)
                mlflow.log_param("date_range", f"{start_date} → {end_date}")
                mlflow.log_metric("weather_row_count", 0)

    except Exception as e:
        logging.error(f"❌ Error fetching future weather data: {e}")
        tracker_log_and_progress(task_id, f"❌ Error fetching weather data: {e}")
        df = pd.DataFrame()

        # Optional reasoning log
        try:
            log_reasoning_to_langchain(f"Error fetching weather data: {str(e)}")
        except:
            pass

    return df


import numpy as np
import pandas as pd
import logging
from datetime import timedelta

def generate_future_prediction_input_lstm(
    train_df: pd.DataFrame,
    df_weather_future: pd.DataFrame,
    future_days: pd.DatetimeIndex,
    features: list,
    model,
    window_size: int,
    le,
    task_id: str = None
):
    """
    Generate future daily predictions for LSTM using ONLY the target-column sequence,
    while still computing engineered features for output consistency.
    """

    logging.info("🔹 Starting LSTM-based daily prediction generation (with engineered features)")

    if task_id:
        tracker_log_and_progress(task_id, "🔹 Starting LSTM-based daily prediction generation (engineered features)")

    # --- Sort input and ensure datetime ---
    train_df = train_df.sort_values(["SECTION_ID", "DAYS"])
    df_weather_future["WEATHER_DATE"] = pd.to_datetime(df_weather_future["WEATHER_DATE"])
    df_weather_future = df_weather_future.set_index("WEATHER_DATE")

    # --- Initialize future predictions ---
    future_predictions = []
    section_id = train_df["SECTION_ID"].iloc[0]
    section_id_en = le.transform([str(section_id)])[0]

    # ================================================================
    # 🔥 CRITICAL FIX: Create LSTM sequence ONLY from target column
    # ================================================================
    target_values = train_df["ACTUAL_CONSUMPTION"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(target_values)

    last_sequence = scaled_y[-window_size:].copy()       # shape: (window_size, 1)
    current_sequence = last_sequence.copy()
    # ================================================================

    # --- Iterate over each future day ---
    for future_day in future_days:

        if task_id:
            tracker_log_and_progress(task_id, f"📅 Predicting for {future_day.strftime('%Y-%m-%d')}")

        # --- Merge weather values ---
        if future_day in df_weather_future.index:
            w_row = df_weather_future.loc[future_day]
        else:
            w_row = df_weather_future.iloc[-1]

        # --- Base new record ---
        last_known = train_df.iloc[-1].copy()

        new_row = {
            "SECTION_ID": section_id,
            "DAYS": future_day,
            "SECTION_ID_En": section_id_en,
        }

        for col in ["TEMPERATURE", "HUMIDITY", "RAIN_CHANCE", "PRESSURE", "CLOUD"]:
            new_row[col] = w_row.get(col, last_known.get(col, np.nan))

        # ==== Build lagged & rolling features ====

        temp_df = pd.concat(
            [train_df, pd.DataFrame(future_predictions)],
            ignore_index=True
        ).sort_values("DAYS")

        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

        for col in ["TEMPERATURE", "HUMIDITY", "RAIN_CHANCE", "PRESSURE", "CLOUD"]:
            for lag in [1, 2, 3, 7]:
                temp_df[f"{col}_lag{lag}"] = temp_df.groupby("SECTION_ID")[col].shift(lag)

            for window in [3, 7]:
                temp_df[f"{col}_roll{window}"] = (
                    temp_df.groupby("SECTION_ID")[col]
                    .rolling(window)
                    .mean()
                    .reset_index(0, drop=True)
                )

        # Consumption-based features
        temp_df["Prev_Day_Consumption"] = temp_df.groupby("SECTION_ID")["ACTUAL_CONSUMPTION"].shift(1)
        temp_df["Rolling_3Day_Consumption"] = (
            temp_df.groupby("SECTION_ID")["ACTUAL_CONSUMPTION"].rolling(3).mean().reset_index(0, drop=True)
        )
        temp_df["Rolling_7Day_Consumption"] = (
            temp_df.groupby("SECTION_ID")["ACTUAL_CONSUMPTION"].rolling(7).mean().reset_index(0, drop=True)
        )

        temp_df["Rain_Cloud_Interaction"] = temp_df["RAIN_CHANCE"] * temp_df["CLOUD"]

        # --- Select final row (for prediction)
        current_row = temp_df.iloc[-1].copy()

        # ================================================================
        # 🔥 CRITICAL FIX: Build correct LSTM input
        # ================================================================
        input_seq = current_sequence.reshape(1, window_size, 1)
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        pred_value = scaler.inverse_transform([[pred_scaled]])[0][0]
        # ================================================================

        # Assign predicted value so rolling features work
        current_row["ACTUAL_CONSUMPTION"] = pred_value
        current_row["PREDICTED_VALUE"] = pred_value

        # Store prediction
        future_predictions.append(
            current_row[train_df.columns.intersection(temp_df.columns)].to_dict()
        )

        # ================================================================
        # 🔥 CRITICAL FIX: Update sequence using ONLY predicted target
        # ================================================================
        next_scaled = scaler.transform([[pred_value]])  # shape = (1,1)
        current_sequence = np.vstack([current_sequence[1:], next_scaled])
        # ================================================================

    # Combine output
    future_df = pd.DataFrame(future_predictions)

    if task_id:
        tracker_log_and_progress(
            task_id,
            f"✅ Completed LSTM-based daily future predictions ({len(future_df)} days)"
        )
        log_reasoning_to_langchain.apply_async(
            args=[task_id, f"✅ LSTM future prediction complete for {len(future_df)} days", "Forecast Completion"],
            queue="langchain_queue"
        )

    logging.info("✅ Future daily predictions generated successfully with feature consistency")
    return future_df


def generate_future_prediction_input_avghistory_automl(
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


from collections import deque
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

def generate_future_prediction_input_avghistory(
    train_df, df_weather, future_months, features, model, task_id=None
):
    """
    Hybrid forecast generator with smoothing + LangChain reasoning logs:
    - Autoregressive with rolling mean for stability
    - Blended with historical monthly average
    - Includes actual consumption if available
    - Weather lag and rolling features supported
    - Logs LangChain reasoning updates per month
    """

    alpha = 0.85
    rolling_window = 3
    future_dfs = []

    logging.info("🔹 Starting future prediction generation with LangChain reasoning")
    if task_id:
        tracker_log_and_progress(task_id, "🔹 Starting future prediction generation")

    # ✅ Initialize rolling previous values per section
    rolling_prev = defaultdict(list)
    for sec, group in train_df.groupby("SECTION_ID"):
        rolling_prev[sec] = group["COLLECTION"].dropna().tolist()[-rolling_window:]

    # ✅ Compute monthly averages per section
    section_month_avg = (
        train_df.groupby(["SECTION_ID", "REV_MONTH"])["COLLECTION"].mean().to_dict()
    )

    # ✅ Ensure REV_YEAR / REV_MONTH exist in weather data
    if "REV_YEAR" not in df_weather.columns or "REV_MONTH" not in df_weather.columns:
        if "WEATHER_DATE" in df_weather.columns:
            df_weather["WEATHER_DATE"] = pd.to_datetime(df_weather["WEATHER_DATE"])
            df_weather["REV_YEAR"] = df_weather["WEATHER_DATE"].dt.year
            df_weather["REV_MONTH"] = df_weather["WEATHER_DATE"].dt.month
        else:
            raise ValueError("df_weather must have either WEATHER_DATE or REV_YEAR/REV_MONTH")

    # ✅ Monthly aggregation of weather data
    monthly_weather = (
        df_weather.groupby(["REV_YEAR", "REV_MONTH"])
        .mean(numeric_only=True)
        .reset_index()
    )

    weather_cols = ["TEMPERATURE", "HUMIDITY", "RAIN_CHANCE", "PRESSURE", "CLOUD"]

    # Track weather history per section for lag and rolling
    weather_history = {
        sec: {
            col: list(train_df[train_df["SECTION_ID"] == sec][col].dropna().values)
            for col in weather_cols
        }
        for sec in train_df["SECTION_ID"].unique()
    }

    # ✅ Iterate through each future month
    for dt in future_months:
        year, month = dt.year, dt.month
        month_label = f"{year}-{month:02d}"
        tracker_log_and_progress(task_id, f"📅 Generating predictions for {month_label}")

        # ✅ Log reasoning start for month
        if task_id:
            log_reasoning_to_langchain.apply_async(
                args=[
                    task_id,
                    f"Generating future predictions for {month_label}",
                    "Forecast Progress",
                ],
                queue="langchain_queue",
            )

        # Get monthly weather
        month_weather = monthly_weather[
            (monthly_weather["REV_YEAR"] == year)
            & (monthly_weather["REV_MONTH"] == month)
        ]
        if month_weather.empty:
            month_weather = pd.DataFrame(
                [{col: 0 for col in weather_cols + ["REV_YEAR", "REV_MONTH"]}]
            )

        rows = []
        for sec in train_df["SECTION_ID"].unique():
            last_row = train_df[train_df["SECTION_ID"] == sec].iloc[-1].copy()
            row = last_row.copy()
            row["REV_YEAR"] = year
            row["REV_MONTH"] = month

            # Actual consumption if available
            actual_available = (
                (train_df["SECTION_ID"] == sec)
                & (train_df["REV_YEAR"] == year)
                & (train_df["REV_MONTH"] == month)
            )
            if train_df[actual_available].shape[0] > 0:
                row["COLLECTION"] = train_df.loc[
                    actual_available, "COLLECTION"
                ].values[0]
            else:
                row["COLLECTION"] = None

            # Rolling mean
            row["Previous_month_Consumption"] = (
                np.mean(rolling_prev[sec][-rolling_window:])
                if rolling_prev[sec]
                else 0
            )

            # Sectional average and difference
            row["Section_Monthly_Avg"] = section_month_avg.get(
                (sec, month), row.get("Section_Monthly_Avg", 0)
            )
            row["Consumption_vs_Avg"] = (
                row["Previous_month_Consumption"] - row["Section_Monthly_Avg"]
            )

            # Weather and lag features
            for col in weather_cols:
                val = month_weather[col].values[0] if col in month_weather.columns else 0
                row[col] = val
                weather_history[sec][col].append(val)

                # Lag features
                for lag in [1, 2, 3]:
                    key = f"{col}_lag{lag}"
                    row[key] = (
                        weather_history[sec][col][-lag - 1]
                        if len(weather_history[sec][col]) > lag
                        else 0
                    )

                # Rolling mean features
                for window in [2, 3]:
                    key = f"{col}_roll{window}"
                    row[key] = (
                        np.mean(weather_history[sec][col][-window:])
                        if len(weather_history[sec][col]) >= window
                        else 0
                    )

            rows.append(row)

        fdf = pd.DataFrame(rows)

        # ✅ Predict using model
        preds = model.predict(fdf[features])

        # ✅ Hybrid blending
        blended_preds = []
        for sec, pred in zip(fdf["SECTION_ID"], preds):
            hist_avg = section_month_avg.get((sec, month), pred)
            blended_preds.append(alpha * pred + (1 - alpha) * hist_avg)

        fdf["PREDICTED_VALUE"] = blended_preds

        # ✅ Update rolling_prev
        for sec, pred in zip(fdf["SECTION_ID"], blended_preds):
            rolling_prev[sec].append(pred)
            if len(rolling_prev[sec]) > rolling_window:
                rolling_prev[sec] = rolling_prev[sec][-rolling_window:]

        future_dfs.append(fdf)

        # ✅ Log reasoning after prediction for month
        if task_id:
            summary_msg = (
                f"Completed predictions for {month_label} "
                f"({len(fdf)} sections). Average forecast: {np.mean(blended_preds):.2f}"
            )
            log_reasoning_to_langchain.apply_async(
                args=[task_id, summary_msg, "Forecast Summary"],
                queue="langchain_queue",
            )
            tracker_log_and_progress(task_id, summary_msg)

    final_df = pd.concat(future_dfs, ignore_index=True)

    if task_id:
        log_reasoning_to_langchain.apply_async(
            args=[task_id, "✅ All future months predicted successfully", "Forecast Completion"],
            queue="langchain_queue",
        )
        tracker_log_and_progress(task_id, "✅ All future predictions generated", "completed")

    logging.info("✅ Future prediction generation completed")
    return final_df



def safe_float(x):
    """Safely convert to float if possible, else return None."""
    try:
        return float(x) if pd.notnull(x) else None
    except Exception:
        return None


@shared_task(bind=True)
def insert_accuracy_log(self, section_id, model_version, training_date, trained_with, r2, rmse, dashboard, task_id: str):
    """
    Inserts model training performance logs into MODEL_TRAINING_LOGS table.
    Also triggers reasoning log for LangChain summarization.
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            msg = "❌ UAT DB connection not available."
            tracker_log_and_progress(task_id, msg)

            log_reasoning_to_langchain.apply_async(args=[task_id, msg, "connection Phase"], queue="langchain_queue")
            return None

        cursor = conn.cursor()
        training_date_obj = datetime.strptime(training_date, "%Y-%m-%d").date()

        # --- ✅ Delete existing record before inserting
        delete_sql = """
            DELETE FROM MIS_USER.MODEL_TRAINING_LOGS
            WHERE SECTION_ID = :section_id AND MODEL_VERSION = :model_version
        """
        cursor.execute(delete_sql, {"section_id": section_id, "model_version": model_version})
        conn.commit()

        tracker_log_and_progress(task_id, f"🗑️ Deleted existing logs for Section {section_id}, Version {model_version}")

        insert_sql = """
            INSERT INTO MIS_USER.MODEL_TRAINING_LOGS (
                SECTION_ID, MODEL_VERSION, TRAINING_DATE, TRAINED_WITH, R2_SCORE, RMSE, DASHBOARD
            )
            VALUES (:section_id, :model_version, TO_DATE(:training_date, 'YYYY-MM-DD'), 
                    :trained_with, :r2, :rmse, :dashboard)
        """

        cursor.execute(insert_sql, {
            "section_id": section_id,
            "model_version": model_version,
            "training_date": training_date_obj.strftime("%Y-%m-%d"),
            "trained_with": trained_with,
            "r2": r2,
            "rmse": rmse,
            "dashboard": dashboard,
        })
        conn.commit()

        msg = f"📘 Logged model training details for section {section_id}"
        logging.info(msg)
        tracker_log_and_progress(task_id, msg)

        log_reasoning_to_langchain.apply_async(args=[task_id, msg, "Logged Model Training Phase"], queue="langchain_queue")

    except Exception as e:
        err_msg = f"❌ Failed to log model info for section {section_id}: {e}"
        logging.error(err_msg)
        tracker_log_and_progress(task_id, err_msg)
        log_reasoning_to_langchain.apply_async(args=[task_id, err_msg, "Error Phase"], queue="langchain_queue")
    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)

from datetime import datetime

@shared_task
def insert_section_data(section_id: str, records: List[dict], task_id: str, model_name: str):
    """
    Inserts section-wise prediction data into the REVENUE_PREDICTION table.
    Deletes previous entries for the same section and model before inserting new data.
    Finally, triggers LangChain summary logging to 'langchain_queue'.
    """
    conn = None
    cursor = None

    try:
        # Start logging
        log_reasoning_to_langchain.apply_async(
            args=[task_id, f"🟦 Starting data insertion for section {section_id}", "Data Insertion"],
            queue="langchain_queue"
        )

        conn = get_db_connection()
        if conn is None:
            raise ConnectionError("❌ UAT database connection could not be established.")

        cursor = conn.cursor()

        # Determine model name
        model_name_full = str(records[0].get("MODEL_NAME", model_name)).strip() if records else model_name

        # ---------------------------------------------------------
        # Step 1: Delete previous records
        # ---------------------------------------------------------
        delete_sql = """
            DELETE FROM MIS_USER.POWER_CONSUMPTION_PREDICTIONS
            WHERE SECTION_ID = :SECTION_ID AND model = :model_name
        """
        cursor.execute(delete_sql, {"SECTION_ID": section_id, "model_name": model_name_full})
        conn.commit()

        msg = f"🧹 Deleted previous predictions for section {section_id} (model={model_name_full})"
        logging.info(msg)
        tracker_log_and_progress(task_id, msg)
        log_reasoning_to_langchain.apply_async(args=[task_id, msg, "Cleanup Phase"], queue="langchain_queue")


        # ---------------------------------------------------------
        # Step 2: Prepare new records (DATE FIX APPLIED HERE)
        # ---------------------------------------------------------
        insert_sql = """
            INSERT INTO MIS_USER.POWER_CONSUMPTION_PREDICTIONS (
                DAY, SECTION_ID, actual, predicted_value, model_version, record_status
            )
            VALUES (:1, :2, :3, :4, :5, :6)
        """

        data_tuples = []

        for rec in records:
            # --- Convert DAY from string → datetime.date ---
            day_value = rec.get("DAY")
            if day_value:
                # "2025-11-28" → datetime.date(2025,11,28)
                day_value = datetime.strptime(day_value, "%Y-%m-%d").date()
            else:
                day_value = None

            # Build tuple
            data_tuples.append((
                day_value,
                str(rec.get("SECTION_ID")).strip() if rec.get("SECTION_ID") else None,
                safe_float(rec.get("ACTUAL_CONSUMPTION")),
                safe_float(rec.get("PREDICTED_CONSUMPTION")),
                str(rec.get("MODEL_NAME", model_name_full)).strip(),
                "1"
            ))

        # ---------------------------------------------------------
        # Step 3: Insert records
        # ---------------------------------------------------------
        cursor.executemany(insert_sql, data_tuples)
        conn.commit()

        msg = f"✅ Inserted {len(data_tuples)} predicted records for section {section_id} successfully."
        logging.info(msg)
        tracker_log_and_progress(task_id, msg)
        log_reasoning_to_langchain.apply_async(args=[task_id, msg, "Data Insertion Phase"], queue="langchain_queue")

        # ---------------------------------------------------------
        # Step 4: Trigger LangChain summary
        # ---------------------------------------------------------
        log_reasoning_to_langchain.apply_async(
            args=[task_id, "🧠 Generating reasoning summary for this section.", "Post Training Summary"],
            queue="langchain_queue"
        )

        save_reasoning_log_to_db.apply_async(
            args=[model_name_full, section_id, task_id],
            queue="langchain_queue"
        )

        msg = f"🧩 Reasoning summary saved for section {section_id}, task_id={task_id}."
        logging.info(msg)
        tracker_log_and_progress(task_id, msg)

    except Exception as e:
        error_msg = f"🔥 Error inserting predicted data for section {section_id}: {e}"
        logging.error(error_msg)
        tracker_log_and_progress(task_id, error_msg)
        log_reasoning_to_langchain.apply_async(args=[task_id, error_msg, "Error Phase"], queue="langchain_queue")

        if conn:
            conn.rollback()

    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)

        log_reasoning_to_langchain.apply_async(
            args=[task_id, f"🏁 Completed section data insertion for {section_id}", "Completion Phase"],
            queue="langchain_queue"
        )
