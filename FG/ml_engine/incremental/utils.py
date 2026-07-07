# ml_engine/incremental/utils.py
import os
import joblib
import numpy as np

MODEL_STATE_DIR = os.environ.get("MODEL_STATE_DIR", "model_state")

def save_state(section_id: str, model_name: str, scaler, last_window_scaled: np.ndarray):
    folder = os.path.join(MODEL_STATE_DIR, model_name, f"section_{section_id}")
    os.makedirs(folder, exist_ok=True)
    joblib.dump(scaler, os.path.join(folder, "scaler.pkl"))
    np.save(os.path.join(folder, "last_window.npy"), last_window_scaled)

def load_state(section_id: str, model_name: str):
    folder = os.path.join(MODEL_STATE_DIR, model_name, f"section_{section_id}")
    scaler_path = os.path.join(folder, "scaler.pkl")
    last_window_path = os.path.join(folder, "last_window.npy")
    if os.path.exists(scaler_path) and os.path.exists(last_window_path):
        scaler = joblib.load(scaler_path)
        last_window = np.load(last_window_path)
        return scaler, last_window
    return None, None

def create_sequences_from_scaled(scaled_y: np.ndarray, window_size: int):
    """
    scaled_y shape: (N,1) or (N,)
    returns X (samples, window_size, 1), y (samples,)
    """
    arr = scaled_y.reshape(-1, 1) if scaled_y.ndim == 1 else scaled_y
    X, y = [], []
    total = len(arr)
    for i in range(window_size, total):
        X.append(arr[i-window_size:i, 0])
        y.append(arr[i, 0])
    if not X:
        return np.empty((0, window_size, 1)), np.empty((0,))
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y
