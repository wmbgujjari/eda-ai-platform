import os
import mlflow

def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment("Daily_Demand_Forecast")

    return mlflow