# -*- coding: utf-8 -*-
# @Time : 2023/10/5 14:48
# @Author : 杨军
# @Email : gpu0@qq.com
# @File : alexnet.py
# @Project : tiny_nn
from layers.convolution import Conv2D
from net.model import Model
from layers.dense import Dense
from layers.maxpool import MaxPooling
from layers.relu import Relu
from layers.softmax_ce import SoftmaxWithLoss


# 简化后的AlexNet，这种情况下16G内存可以跑
class AlexNet(Model):
    def __init__(self, x_train, y_train, epochs=30, weight_decay_lambda=0, sample_batches=True,
                 batch_size=100, optimizer='SGD', optimizer_param={'lr': 0.01}, learning_rate_decay=1, verbose=True):
        super().__init__(x_train, y_train, epochs, weight_decay_lambda, sample_batches, batch_size, optimizer,
                         optimizer_param, learning_rate_decay, verbose)

        # 修改AlexNet的各层以适应32x32的灰度图,输出10个类别
        self.add(Conv2D(kernels_num=16, kernel_h=3, kernel_w=3, stride=1, padding=1, name="Conv1",
                        weights_init_type="he"))
        self.add(Relu(name='Relu1'))
        self.add(MaxPooling(name="MaxPool1", stride=2, pool_h=2, pool_w=2))

        self.add(Conv2D(kernels_num=32, kernel_h=3, kernel_w=3, stride=1, padding=1, name="Conv2",
                        weights_init_type="he"))
        self.add(Relu(name='Relu2'))
        self.add(MaxPooling(name="MaxPool2", stride=2, pool_h=2, pool_w=2))

        self.add(Conv2D(kernels_num=64, kernel_h=3, kernel_w=3, stride=1, padding=1, name="Conv3",
                        weights_init_type="he"))
        self.add(Relu(name='Relu3'))
        self.add(MaxPooling(name="MaxPool3", stride=2, pool_h=2, pool_w=2))

        self.add(Dense(hidden_size=128, name="Dense1", weights_init_type="he"))  # 减小全连接层神经元数量
        self.add(Relu(name='Relu4'))
        self.add(Dense(hidden_size=10, name="Dense2", weights_init_type="he"))

        loss_layer = SoftmaxWithLoss()
        self.add(loss_layer, loss=True)

        # 初始化权重参数
        self.init_weights()

