# -*- coding: utf-8 -*-
# @Time : 2023/10/5 0:02
# @Author : 杨军
# @Email : gpu0@qq.com
# @File : new_lenet5.py
# @Project : tiny_nn
from layers.convolution import Conv2D
from net.model import Model
from layers.dense import Dense
from layers.maxpool import MaxPooling
from layers.relu import Relu
from layers.softmax_ce import SoftmaxWithLoss


class LeNet5(Model):
    def __init__(self, x_train, y_train, epochs=30, weight_decay_lambda=0, sample_batches=True,
                 batch_size=100, optimizer='SGD', optimizer_param={'lr': 0.01}, learning_rate_decay=1, verbose=True):
        super().__init__(x_train, y_train, epochs, weight_decay_lambda, sample_batches, batch_size, optimizer,
                         optimizer_param, learning_rate_decay, verbose)

        # 添加LeNet-5的各层
        self.add(Conv2D(kernels_num=6, kernel_h=5, kernel_w=5, stride=1, padding=0, name="Conv_1",
                        weights_init_type="he"))  # (32+0-5)/1+1=28
        self.add(Relu(name='Relu_1'))
        self.add(MaxPooling(name="MaxPooling_1", stride=2, pool_h=2, pool_w=2))  # (28-2)/2+1=14

        self.add(Conv2D(kernels_num=16, kernel_h=5, kernel_w=5, stride=1, padding=0, name="Conv_2",
                        weights_init_type="he"))  # (14+0-5)/1+1=10
        self.add(Relu(name='Relu_2'))
        self.add(MaxPooling(name="MaxPooling_1", stride=2, pool_h=2, pool_w=2))  # (10-2)/2+1=5

        self.add(Dense(hidden_size=120, name="Dense_1", weights_init_type="he"))  # (,120)
        self.add(Relu(name='Relu_3'))
        self.add(Dense(hidden_size=84, name="Dense_2", weights_init_type="he"))  # (,84)
        self.add(Relu(name='Relu_4'))
        self.add(Dense(hidden_size=10, name="Dense_3", weights_init_type="he"))  # (,84)

        loss_layer = SoftmaxWithLoss()
        self.add(loss_layer, loss=True)

        # 初始化权重参数
        self.init_weights()
