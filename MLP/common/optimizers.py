from .np import *


class Optimizer:
    def __init__(self, lr=0.01, l2_reg=0.0):
        self.lr = lr
        self.l2_reg = l2_reg

    def update(self, params):
        for i in range(len(params)):
            params[i] *= 1.0 - self.lr * self.l2_reg


class SGD(Optimizer):
    """
    随机梯度下降法（Stochastic Gradient Descent）
    """

    def __init__(self, lr=0.01, lr_decay=0.0, l2_reg=0.0):
        super().__init__(lr, l2_reg)
        self.lr_decay = lr_decay

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
        self.lr *= 1.0 - self.lr_decay
        super().update(params)


class Momentum:
    """
    Momentum SGD
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]


class Adam:
    """
    Adam (http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
