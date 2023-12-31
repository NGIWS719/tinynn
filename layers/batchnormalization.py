# -*- coding: utf-8 -*-
# @File  : batchnormalization.py
# @Author: Yangjun
# @Date  : 2023/7/23
# @Desc  : 批量归一化
import sys, os

sys.path.append(os.pardir)  # Setting for import the parent directory
import numpy as np
from layers.Layer import Layer


class BatchNormalization(Layer):
    """
    Parameters
    ----------
    gamma: float
    beta: float
    momentum: float; Default value is 0.9
    running_mean: float; At the time of testing, it is the mean
    running_var: float; At the time of testing, it is the var
    name: string; Default value is "Batch Normalization"
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None, name="Batch Normalization"):
        super().__init__(name, op="BatchNorm")
        self.W = gamma
        self.B = beta
        self.momentum = momentum
        # In the case of Conv layer, it is 4 dimensions, and in the case of fully connected layer, it is 2 dimensions
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        # Middle data used for backward
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dw = None
        self.db = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            ub = x.mean(axis=0)
            xc = x - ub
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * ub
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.W * xn + self.B
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        return dx.reshape(*self.input_shape)

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.W * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dub = np.sum(dxc, axis=0)
        dx = dxc - dub / self.batch_size

        self.dw = dgamma
        self.db = dbeta

        return dx


"""
Test
"""

if __name__ == "__main__":
    batch = BatchNormalization(gamma=0.5, beta=0.6)
    imgs = np.random.randint(low=0, high=256, size=(100, 3, 28, 28))
    print("forward:\n")
    print("input_img_shape:", imgs.shape)
    out = batch.forward(imgs)
    print("out_shape:", out.shape)
    print("backward:\n")
    dout = np.random.randint(low=0, high=256, size=out.shape)
    dout = batch.backward(dout)
    print("dout_shape:", dout.shape)
