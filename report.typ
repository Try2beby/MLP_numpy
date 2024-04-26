== 实现细节

主要实现在文件夹`./MLP/common/`下

- `functions.py` 实现了`softmax`和交叉熵损失函数
- `layers.py` 实现了`Sigmoid`, `ReLU`, `Affine`, `Softmax`, `SoftmaxWithLoss`层的正向和反向传播
- `module.py` 实现了`Module`类, 作为一般多层神经网络的基类, 实现了以下方法
  - `compute_accuracy` 计算模型在给定数据集上的准确率
  - `compute_macro_micro_avg` 计算模型在给定数据集上的宏平均和微平均
  - `save_parameters` 保存模型参数
  - `load_parameters` 加载模型参数
