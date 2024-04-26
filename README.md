# Numpy 搭建三层神经网络

## 引入必要的库, 加载数据集

```py
import fashionmnist.utils.mnist_reader as mnist_reader
from MLP.main import ThreeLayerNet
from MLP.common.optimizers import SGD, Adam
from MLP.common.np import *
from MLP.common.trainer import Trainer

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats("svg")


cacheDir = "./cache"

X_train_org, y_train_org = mnist_reader.load_mnist(
    "./fashionmnist/data/fashion", kind="train"
)
X_test, y_test = mnist_reader.load_mnist("./fashionmnist/data/fashion", kind="t10k")

# shuffle the training set
np.random.seed(42)
shuffle_index = np.random.permutation(60000)
X_train_org, y_train_org = X_train_org[shuffle_index], y_train_org[shuffle_index]

# get 5000 samples from training set for validation
X_val, y_val = X_train_org[55000:, :], y_train_org[55000:]
X_train, y_train = X_train_org[:55000, :], y_train_org[:55000]
```

## 训练

```py
trainer_settings = {
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_val,
    "y_test": y_val,
}
fit_settings = {
    "verbose": True,
    "max_epoch": 20
}


# hidden_sizes = [64, 128, 256]
# lrs = [0.01, 0.001, 0.0001]
# l2_regs = [0.0001, 0.001, 0.01]

hidden_size1 = 256
hidden_size2 = 128
lrs = [0.01]
l2_regs = [0.001]

best_params = None
highest_accuracy = 0

# 遍历所有参数组合
for lr in lrs:
    for l2_reg in l2_regs:
        # 更新参数设置
        model_settings = {
            "input_size": 28 * 28,
            "hidden_size1": hidden_size1,
            "hidden_size2": hidden_size2,
            "output_size": 10,
        }

        optimizer_settings = {
            "lr": lr,
            "l2_reg": l2_reg,
        }

        # 创建并训练模型
        model = ThreeLayerNet(**model_settings)
        optimizer = SGD(**optimizer_settings)

        trainer = Trainer(model, optimizer, **trainer_settings)
        trainer.fit(**fit_settings)

        # 如果这个模型的准确率更高，就更新最佳参数和最高准确率
        if trainer.accuracy["test"][-1] > highest_accuracy:
            highest_accuracy = trainer.accuracy["test"][-1]
            best_params = {
                "hidden_size1": hidden_size1,
                "hidden_size2": hidden_size2,
                "lr": lr,
                "l2_reg": l2_reg,
            }

print("Best params:", best_params)
```

## 测试

实例化网络

```py
model = ThreeLayerNet(**model_settings)
```

加载权重

```py
model.load_parameters("./cache/best_accuracy.pkl")
```

计算评估指标

```py
model.compute_accuracy(X_test, y_test)
model.compute_macro_micro_avg(X_test, y_test)
```
