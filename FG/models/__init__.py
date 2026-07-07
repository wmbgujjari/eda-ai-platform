from .lstm_model import LSTMModel
from .gru_model import GRUModel
from .cnn_model import CNNModel


MODEL_MAP = {
"lstm": LSTMModel,
"gru": GRUModel,
"cnn": CNNModel,
}