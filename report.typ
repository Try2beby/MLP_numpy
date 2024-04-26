#show link: underline
= Numpy 搭建三层神经网络

== 数据集介绍
Fashion MNIST 是一个流行的数据集，常用于训练各种机器学习算法识别服装。这个数据集由 Zalando（一家德国的在线时尚零售商）发布，目的是作为传统的 MNIST 数据集的一个替代品，后者包含了手写数字的图片。Fashion MNIST 设计的初衷是提供一个稍微更具挑战性的问题，同时避免 MNIST 数据集的一些问题，如数据集过于简单而不适合现代机器学习算法。

Fashion MNIST 数据集包含 10 个类别的服装图片，如T恤/上衣、裤子、套头衫、裙子、外套、凉鞋、衬衫、运动鞋、包和踝靴。每个类别有 6,000 张训练图像和 1,000 张测试图像，图像的分辨率为 28x28 像素，为灰度图。这使得该数据集在处理和使用上与原始的 MNIST 数据集相似，可以无缝地用于测试不同的机器学习模型和算法。

Fashion MNIST 数据集的使用非常广泛，适合入门级的机器学习项目，是理解图像分类任务的一个很好的起点。同时，由于其相对简单但又比数字识别更接近实际应用，因此也适用于更高级的机器学习和深度学习课程和研究。

== 模型架构

#figure(
  image("./figs/model_MLP.drawio.svg", width: 40%),
  caption: [
    Architecture of the Multi-Layer Perceptron
  ],
)

== 实现细节
repo 地址 #link("https://github.com/Try2beby/MLP_numpy")[here]

主要实现在文件夹`./MLP/common/`下

- `functions.py` 实现了`softmax`和交叉熵损失函数
- `layers.py` 实现了`Sigmoid`, `ReLU`, `Affine`, `Softmax`, `SoftmaxWithLoss`层的正向和反向传播
- `module.py` 实现了`Module`类, 作为一般多层神经网络的基类, 实现了以下方法
  - `compute_accuracy` 计算模型在给定数据集上的准确率
  - `compute_macro_micro_avg` 计算模型在给定数据集上的宏平均和微平均
  - `save_parameters` 保存模型参数
  - `load_parameters` 加载模型参数
- `optimizer.py` 实现了随机梯度下降(SGD), 支持学习率衰减和 $l_2$ -正则化. 此外还实现了 Momentum 和 Adam 优化器
- `trainer.py` 实现`Trainer`类
  - `[load\save]_best_metrics` 加载 \\ 保存最优指标
  - `fit` 训练神经网络. 每个epoch结束时计算一次在训练集和验证集上的指标. 根据当前epoch相应指标与历史最佳指标对比决定是否保存当前权重
  - `plot` 绘制训练集和验证集上的损失函数和分类准确率变化


== 实验结果

尝试了以下参数

```py
hidden_sizes = [64, 128, 256]
lrs = [0.01, 0.001, 0.0001]
l2_regs = [0.0001, 0.001, 0.01]
```
`hidden_sizes1`和`hidden_sizes2`取遍`hidden_sizes`中的值. 得到最优(以准确率为依据)参数组合为

```py
{'hidden_size1':256,'hidden_size2':64,'lr':0.001,12_reg':0.001)
```

其在训练集和验证集上损失函数值和准确率变化如下:

#figure(
  image("./figs/20240426-065913_plot.svg", width: 80%),
  caption: [
    Loss and Accuracy Change on Training and Validation Set
  ],
)

在测试集上的准确率为$86.7%$.

可视化其权重得
#figure(
  image("./figs/weight_layers.svg", width: 60%),
  caption: [
    Visualization of Weights of Each Layer
  ],
)

总体看到，随着网络层数加深，权重分布更加稀疏，偏离均值的权重更多. 具体来说:
- 第一层的可视化显示出许多细小、颜色较深的权重。这可能表明网络在第一层学习了大量的细粒度特征，但是没有单个权重显著地大于其他权重;
- 第二层的权重图示较为分散，显示了更多的变化和结构，但仍然主要由较深的颜色组成。这表明该层的权重中有更多的结构，网络可能开始组合第一层的特征来学习更复杂的表示;
- 第三层中某些权重的绝对值相对较大，这意味着在网络作出最终决策时，这些特征将起到较重要的作用.

再将第一, 第二层每个神经元的权重转为方形矩阵, 得到
// A function to represent a virtual image
#figure(
    grid(
        columns: 1,
        gutter: 2mm,
        image("./figs/weight_layer1.svg", width: 90%),
        image("./figs/weight_layer2.svg", width: 90%),
    ),
    caption: [
        Visualization of Weights of Each Neuron in Layer 1 and Layer 2
    ],
)

可见第一层权重可视化结果非常接近数据集中的服装, 鞋子或包的图像, 说明经过训练, 该网络确实学到了数据集中的细微特征. 第二层的权重则无明显规律可循.