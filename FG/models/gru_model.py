from FG.models.base_dl_model import BaseDLModel
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from typing import Tuple


class GRUModel(BaseDLModel):
    def __init__(self, model_name: str = "gru"):
        super().__init__(model_name)


    def build_model(self, input_shape: Tuple[int, int]):
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(64),
            Dense(1)
        ])
        return model