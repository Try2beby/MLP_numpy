from .common.np import *
from .common.layers import Affine, Sigmoid, ReLU, SoftmaxWithLoss
from .common.module import Module

import matplotlib.pyplot as plt


class ThreeLayerNet(Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, std=1e-4):
        super().__init__()

        I, H1, H2, O = input_size, hidden_size1, hidden_size2, output_size

        W1 = std * np.random.randn(I, H1)
        b1 = np.zeros(H1)
        W2 = std * np.random.randn(H1, H2)
        b2 = np.zeros(H2)
        W3 = std * np.random.randn(H2, O)
        b3 = np.zeros(O)
        # W1 = np.random.randn(I, H1) * np.sqrt(2.0 / I)
        # b1 = np.zeros(H1)
        # W2 = np.random.randn(H1, H2) * np.sqrt(2.0 / H1)
        # b2 = np.zeros(H2)
        # W3 = np.random.randn(H2, O) * np.sqrt(2.0 / H2)
        # b3 = np.zeros(O)

        self.layers = [Affine(W1, b1), ReLU(), Affine(W2, b2), ReLU(), Affine(W3, b3)]
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x, t):
        y = self.predict(x)
        loss = self.loss_layer.forward(y, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def plot_weights(self):
        # count the number of Affine layers
        num_affine = sum(isinstance(layer, Affine) for layer in self.layers)
        idx = 0

        plt.figure(figsize=(num_affine * 2, 6))

        # plot weight matrix
        for layer in self.layers:
            if isinstance(layer, Affine):
                plt.subplot(1, num_affine, idx + 1)
                plt.imshow(np.abs(layer.params[0]))
                idx += 1

        # add colorbar to the figure
        plt.colorbar()

        # save image
        plt.savefig("figs/weight_layers.svg")

        plt.show()
