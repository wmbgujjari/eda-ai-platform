from FG.models.base_dl_model import BaseDLModel
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from typing import Tuple


class CNNModel(BaseDLModel):
    def __init__(self, model_name: str = "cnn"):
        super().__init__(model_name)


    def build_model(self, input_shape: Tuple[int, int]):
        # input_shape: (seq_len, n_features)
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        return model