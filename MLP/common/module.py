from .np import *
import pickle
from sklearn.metrics import precision_recall_fscore_support
from typing import Tuple


class Module:
    def __init__(self) -> None:
        self.layers = []
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        y = self.predict(x)
        y = np.argmax(y, axis=1)
        accuracy = np.mean(y == t)
        return accuracy

    def compute_macro_micro_avg(
        self, x: np.ndarray, t: np.ndarray
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # print(y.shape, t.shape)
        # print(y, t)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            t, y, average=None
        )
        macro_avg = np.mean(precision), np.mean(recall), np.mean(f1_score)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            t, y, average="micro"
        )
        micro_avg = precision, recall, f1_score
        return macro_avg, micro_avg

    def save_parameters(self, filename: str) -> None:
        if len(self.params) == 0:
            return

        with open(filename, "wb") as f:
            pickle.dump(self.params, f)

    def load_parameters(self, filename: str) -> None:
        try:
            with open(filename, "rb") as f:
                loaded_params = pickle.load(f)
        except FileNotFoundError:
            print(f"Failed to load {filename}")
            return

        param_idx = 0
        for layer in self.layers:
            num_layer_params = len(layer.params)
            layer.params = loaded_params[param_idx : param_idx + num_layer_params]
            param_idx += num_layer_params

    def get_or_create_model_info(self):
        model_info = {}
        for layer in self.layers:
            if hasattr(layer, "get_or_create_model_info"):
                model_info.update(layer.get_or_create_model_info())
        return model_info
