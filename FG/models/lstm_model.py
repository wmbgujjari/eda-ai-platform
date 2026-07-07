from FG.models.base_dl_model import BaseDLModel
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from typing import Tuple


class LSTMModel(BaseDLModel):
    def __init__(self, model_name: str = "lstm"):
        super().__init__(model_name)


    def build_model(self, input_shape: Tuple[int, int]):
        # input_shape: (seq_len, n_features)
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64),
            Dense(1)
        ])
        return model