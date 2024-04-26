from .module import Module
from myTorch import np


__all__ = ["Linear"]


class Linear(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: np.ndarray

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.zeros((out_features, in_features))
        if bias:
            self.bias = np.zeros((out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        np.random.seed(0)
        self.weight = np.random.randn(self.out_features, self.in_features)
        if self.bias is not None:
            self.bias = np.random.randn(self.out_features)
