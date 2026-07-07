import numpy as np
from sklearn.preprocessing import MinMaxScaler
from FG.core.utils.log_and_progress import tracker_log_and_progress

def dl_inference(model, train_df, target_col, window_size, scaler, task_id):
    """
    Unified inference for LSTM / GRU / CNN models
    """
    tracker_log_and_progress(task_id, "🧠 Running DL inference...")

    # Extract target
    target_values = train_df[target_col].values.reshape(-1, 1)

    # Use training-time scaler
    scaled_y = scaler.transform(target_values)

    # Build sequence windows
    X_seq = []
    for i in range(window_size, len(scaled_y)):
        X_seq.append(scaled_y[i - window_size:i, 0])

    X_seq = np.array(X_seq).reshape(-1, window_size, 1)

    # True values
    y_true = target_values[window_size:].flatten()

    # Predictions
    y_pred_scaled = model.predict(X_seq).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    return y_true, y_pred
