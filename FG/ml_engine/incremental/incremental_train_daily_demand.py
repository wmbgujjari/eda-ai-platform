# services/incremental_train.py
import os
import logging
import numpy as np
import mlflow
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

from FG.core.config import MODEL_PATH
from FG.core.utils.log_and_progress import tracker_log_and_progress
from FG.ml_engine.train.train_daily_demand import process_data, train_dl_model as full_train  # full_train fallback
from models import MODEL_MAP
from FG.ml_engine.incremental.utils import load_state, save_state, create_sequences_from_scaled

def incremental_train_dl_model(df, df_weather, model_name: str, task_id: str, section_id: str,
                               window_size: int = 30, fine_tune_epochs: int = 2,
                               batch_size: int = 32, save_versioned: bool = True,
                               mlflow_experiment: str = "DailyDemand_Training"):
    model_key = model_name.strip().lower()

    if model_key not in MODEL_MAP:
        raise ValueError(f"Unsupported DL model `{model_name}`")

    tracker_log_and_progress(task_id, f"🚀 Starting incremental DL training for section: {section_id} using {model_key}")
    logging.info(f"Starting incremental DL training: {model_key} section {section_id}")

    # Preprocess (shared)
    train_df, features, target_col, le = process_data(df, df_weather)
    tracker_log_and_progress(task_id, f"🧩 Preprocessing completed for section: {section_id}. Rows: {len(train_df)}")
    logging.info(f"Preprocessing done rows={len(train_df)}")

    # Model file locations (one folder per model/section)
    model_dir_for_section = os.path.join(MODEL_PATH, model_key, f"section_{section_id}")
    os.makedirs(model_dir_for_section, exist_ok=True)
    base_model_file = os.path.join(model_dir_for_section, f"dl_model_{model_key}_section_{section_id}.keras")

    # If no saved model -> fallback to full training
    if not os.path.exists(base_model_file):
        logging.info("No base model found; performing full training.")
        return full_train(df, df_weather, model_name, task_id, section_id,
                          window_size=window_size, epochs=50, batch_size=32)

    # load Keras model via model class or direct load
    ModelClass = MODEL_MAP[model_key]
    model_instance = ModelClass(model_name=model_key)
    # We will load weights into model_instance.model for incremental training
    model_instance.load(base_model_file)

    # load persisted state (scaler + last_window)
    scaler, last_window_scaled = load_state(section_id, model_key)

    # if no state, build from train_df and persist (so next incremental run works)
    if scaler is None or last_window_scaled is None:
        logging.warning("No saved scaler/last_window for this section — creating from train_df")
        scaler = MinMaxScaler()
        scaler.fit(train_df[[target_col]].values.reshape(-1, 1))
        total_scaled = scaler.transform(train_df[[target_col]].values.reshape(-1, 1))
        if len(total_scaled) >= window_size:
            last_window_scaled = total_scaled[-window_size:].reshape(window_size, 1)
        else:
            pad = window_size - len(total_scaled)
            last_window_scaled = np.vstack([np.zeros((pad, 1)), total_scaled.reshape(-1, 1)])
        # persist
        save_state(section_id, model_key, scaler, last_window_scaled)

    # Build combined scaled array = last_window_scaled + newly appended rows from train_df beyond window boundary
    total_scaled = scaler.transform(train_df[[target_col]].values.reshape(-1, 1))
    total_len = len(total_scaled)

    if total_len <= window_size:
        logging.warning("Not enough rows to perform incremental sequences; falling back to full training")
        return full_train(df, df_weather, model_name, task_id, section_id,
                          window_size=window_size, epochs=50, batch_size=32)

    # compute new_scaled_rows as the rows in total_scaled after the first (total_len - window_size) index
    num_new_rows = total_len - window_size
    new_scaled_rows = total_scaled[-num_new_rows:].reshape(num_new_rows, 1) if num_new_rows > 0 else np.empty((0, 1))

    combined = np.vstack([last_window_scaled.reshape(-1, 1), new_scaled_rows]) if new_scaled_rows.size > 0 else last_window_scaled.reshape(-1, 1)

    # create incremental sequences from combined
    X_new, y_new = create_sequences_from_scaled(combined, window_size)

    if X_new.shape[0] == 0:
        logging.warning("No incremental sequences created; skipping incremental update")
        return {"status": "skipped", "reason": "no_new_sequences"}

    # callbacks
    callbacks = [EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)]

    # call model_instance.incremental_train
    history = model_instance.incremental_train(
        X_new, y_new,
        model_file=base_model_file,
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        save_versioned=save_versioned,
        model_dir_for_section=model_dir_for_section
    )

    # update last_window_scaled with tail of total_scaled
    new_last_window = total_scaled[-window_size:].reshape(window_size, 1)
    save_state(section_id, model_key, scaler, new_last_window)

    # Build model bundle to match existing contract
    model_bundle = {
        "model": model_instance.model,
        "train_df": train_df,
        "features": features,
        "trained_by": model_name.upper(),
        "target_col": target_col,
        "le": le,
        f"is_{model_key}": True,
        "window_size": window_size,
        "scaler": scaler,
        "history": history.history if hasattr(history, 'history') else None
    }

    tracker_log_and_progress(task_id, f"✅ Incremental DL training completed for section: {section_id}")
    logging.info(f"✅ Incremental DL training completed for section: {section_id}")

    return model_bundle
