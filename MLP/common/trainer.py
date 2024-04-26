# coding: utf-8
import sys

sys.path.append("..")
import numpy
import time
import matplotlib.pyplot as plt
from .np import *  # import numpy as np
from .util import clip_grads
from .module import Module

import sys


def is_notebook():
    if "ipykernel" in sys.modules:
        return True
    else:
        return False


if is_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import json


class Trainer:
    def __init__(self, model: Module, optimizer, X_train, y_train, X_test, y_test):
        self.model = model
        self.optimizer = optimizer
        self.eval_interval = None
        self.current_epoch = 0
        self.dataset = {"train": (X_train, y_train), "test": (X_test, y_test)}
        self.accuracy = {"train": [], "test": []}
        self.loss = {"train": [], "test": []}
        self.macro_micro_avg = None

    def load_best_metrics(self):
        with open("cache/best_metrics.json", "r") as f:
            best_metrics = json.load(f)
        self.best_accuracy = best_metrics["best_accuracy"]
        self.best_macro_micro_avg = best_metrics["best_macro_micro_avg"]

    def save_best_metrics(self):
        best_metrics = {
            "best_accuracy": self.best_accuracy,
            "best_macro_micro_avg": self.best_macro_micro_avg,
        }
        with open("cache/best_metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=4)

        print("Best metrics saved to cache/best_metrics.json")

    def fit(
        self,
        max_epoch=20,
        batch_size=32,
        max_grad=None,
        # eval_interval=20,
        verbose=False,
    ):
        x = self.dataset["train"][0]
        t = self.dataset["train"][1]
        data_size = len(x)
        max_iters = data_size // batch_size
        # eval_interval = max_iters
        model, optimizer = self.model, self.optimizer

        self.load_best_metrics()
        best_accuracy = self.best_accuracy
        best_macro_micro_avg = self.best_macro_micro_avg

        start_time = time.time()
        for epoch in range(max_epoch):
            # 打乱
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]
            total_loss = 0
            loss_count = 0
            for iters in tqdm(range(max_iters)):
                batch_x = x[iters * batch_size : (iters + 1) * batch_size]
                batch_t = t[iters * batch_size : (iters + 1) * batch_size]

                # 计算梯度，更新参数
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(
                    model.params, model.grads
                )  # 将共享的权重整合为1个
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

            if verbose:
                avg_loss = total_loss / loss_count
                elapsed_time = time.time() - start_time
                print(
                    "| epoch %d | time %d[s] | loss %.2f"
                    % (
                        epoch + 1,
                        elapsed_time,
                        avg_loss,
                    )
                )

                self.loss["train"].append(float(avg_loss))
                total_loss, loss_count = 0, 0

                self.loss["test"].append(
                    model.forward(self.dataset["test"][0], self.dataset["test"][1])
                )

            self.accuracy["train"].append(
                self.model.compute_accuracy(
                    self.dataset["train"][0], self.dataset["train"][1]
                )
            )
            self.accuracy["test"].append(
                self.model.compute_accuracy(
                    self.dataset["test"][0], self.dataset["test"][1]
                )
            )
            self.macro_micro_avg = self.model.compute_macro_micro_avg(
                self.dataset["test"][0], self.dataset["test"][1]
            )

            if self.accuracy["test"][-1] > best_accuracy:
                self.model.save_parameters("cache/" + "best_accuracy.pkl")
                best_accuracy = self.accuracy["test"][-1]
            if all(
                [
                    self.macro_micro_avg[0][i] > best_macro_micro_avg[0][i]
                    for i in range(3)
                ]
            ):
                self.model.save_parameters("cache/" + "best_macro_avg.pkl")
                best_macro_micro_avg[0] = self.macro_micro_avg[0]
            if all(
                [
                    self.macro_micro_avg[1][i] > best_macro_micro_avg[1][i]
                    for i in range(3)
                ]
            ):
                self.model.save_parameters("cache/" + "best_micro_avg.pkl")
                best_macro_micro_avg[1] = self.macro_micro_avg[1]

        self.best_accuracy = best_accuracy
        self.best_macro_micro_avg = best_macro_micro_avg
        self.save_best_metrics()

    def plot(self):
        from matplotlib.ticker import MaxNLocator

        epochs = range(1, len(self.accuracy["train"]) + 1)

        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=color)
        ax1.plot(epochs, self.loss["train"], color=color, label="Train Loss")
        ax1.plot(
            epochs,
            self.loss["test"],
            color=color,
            linestyle="dashed",
            label="Val Loss",
        )
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = "tab:blue"
        ax2.set_ylabel(
            "Accuracy", color=color
        )  # we already handled the x-label with ax1
        ax2.plot(epochs, self.accuracy["train"], color=color, label="Train Accuracy")
        ax2.plot(
            epochs,
            self.accuracy["test"],
            color=color,
            linestyle="dashed",
            label="Val Accuracy",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        # Make x-axis display integer values only
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.legend(loc="upper left")
        plt.show()


def remove_duplicate(params, grads):
    """
    将参数列表中重复的权重整合为1个，
    加上与该权重对应的梯度
    """
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 在共享权重的情况下
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 加上梯度
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 在作为转置矩阵共享权重的情况下（weight tying）
                elif (
                    params[i].ndim == 2
                    and params[j].ndim == 2
                    and params[i].T.shape == params[j].shape
                    and np.all(params[i].T == params[j])
                ):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads
