from typing import Tuple, Any
import os
import logging
import numpy as np
from keras.models import load_model
import mlflow
from typing import Tuple, Any, Optional


class BaseDLModel:
    """Base class for DL models. Subclasses implement build_model().
    Responsibilities:
    - Provide a consistent interface: build_model, train, save, load
    - Keep file I/O and model lifecycle standardized
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None


    def build_model(self, input_shape: Tuple[int, int]):
        """Return a compiled Keras model. Override in subclass."""
        raise NotImplementedError

    def ensure_built(self, input_shape: Tuple[int, int]):
        if self.model is None:
            self.model = self.build_model(input_shape)

    def train(self, X, y, model_file: str, callbacks=None, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """
        Train model (Keras) and save to model_file.
        Returns: (model, history)
        """
        if self.model is None:
            self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))


        self.model.compile(optimizer="adam", loss="mse")

        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            verbose=0
        )


    # Save after training
        if model_file:
            try:
                self.save(model_file)
            except Exception:
                logging.exception("Failed to save model to %s", model_file)
        return self.model, history

    def incremental_train(self, X_inc: np.ndarray, y_inc: np.ndarray,
                          model_file: Optional[str] = None,
                          epochs: int = 1, batch_size: int = 32,
                          callbacks: Optional[list] = None, save_versioned: bool = False,
                          model_dir_for_section: Optional[str] = None):
        """
        Warm-start incremental training on provided X_inc, y_inc (already shaped for the model).
        If model is not built/loaded, raise error — caller should load model via load().
        Optionally save model_file after training, and optionally save a versioned copy.
        Returns history object.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded or built for incremental training.")

        # Ensure compiled
        try:
            self.model.compile(optimizer="adam", loss="mse")
        except Exception:
            pass

        history = self.model.fit(
            X_inc, y_inc,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks or [],
            verbose=0
        )

        # Save model if requested
        if model_file:
            # optional versioned copy
            if save_versioned and model_dir_for_section:
                timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
                ver_path = os.path.join(model_dir_for_section, f"{self.model_name}_incremental_{timestamp}.keras")
                try:
                    self.model.save(ver_path)
                    try:
                        mlflow.log_artifact(ver_path)
                    except Exception:
                        logging.warning("mlflow log artifact (versioned) failed.")
                except Exception:
                    logging.exception("Failed to write versioned model to %s", ver_path)

            # overwrite active file
            try:
                self.model.save(model_file)
                try:
                    mlflow.log_artifact(model_file)
                except Exception:
                    logging.warning("mlflow log artifact (active) failed.")
            except Exception:
                logging.exception("Failed to save model to %s", model_file)

        return history

    def save(self, model_file: str):
        """Save model to disk + try to log to MLflow."""
        if self.model is None:
            raise RuntimeError("No model instance to save")
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        self.model.save(model_file)
        try:
            mlflow.log_artifact(model_file)
        except Exception:
            logging.warning("mlflow log artifact failed for %s", model_file)

    def load(self, model_file: str):
        """Load model from disk and set self.model"""
        if not os.path.exists(model_file):
            raise FileNotFoundError(model_file)
        self.model = load_model(model_file)
        return self.model