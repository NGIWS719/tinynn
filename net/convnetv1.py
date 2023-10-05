# -*- coding: utf-8 -*-
# @Time : 2023/10/5 18:24
# @Author : 杨军
# @Email : gpu0@qq.com
# @File : convnetv1.py
# @Project : tiny_nn
from layers.batchnormalization import BatchNormalization
from layers.convolution import Conv2D
from layers.dropout import Dropout
from net.model import Model
from layers.dense import Dense
from layers.maxpool import MaxPooling
from layers.relu import Relu
from layers.softmax_ce import SoftmaxWithLoss


class ConvNetV1(Model):
    def __init__(self, x_train, y_train, epochs=30, weight_decay_lambda=0, sample_batches=True,
                 batch_size=100, optimizer='SGD', optimizer_param={'lr': 0.01}, learning_rate_decay=1, verbose=True):
        super().__init__(x_train, y_train, epochs, weight_decay_lambda, sample_batches, batch_size, optimizer,
                         optimizer_param, learning_rate_decay, verbose)

        # 修改AlexNet的各层以适应32x32的灰度图,输出10个类别
        self.add(Conv2D(kernels_num=16, kernel_h=3, kernel_w=3, stride=2, padding=2, name="Conv_1",
                        weights_init_type="he"))  # (28+4-3)/2+1=15
        self.add(BatchNormalization(gamma=1, beta=0))
        self.add(Relu(name='Relu_1'))
        self.add(MaxPooling(name="MaxPooling_1", stride=1, pool_h=3, pool_w=3))  # (15-3)/1+1=13

        self.add(Conv2D(kernels_num=32, kernel_h=3, kernel_w=3, stride=2, padding=1, name="Conv_2",
                        weights_init_type="he"))  # (13+2-3)/2+1=7
        self.add(BatchNormalization(gamma=1, beta=0))
        self.add(Relu(name='Relu_2'))
        self.add(MaxPooling(name="MaxPooling_2", stride=1, pool_h=3, pool_w=3))  # (7-3)/1+1=5

        self.add(Dense(hidden_size=1024, name="Dense_1", weights_init_type="he"))  # (100,1024)
        self.add(Relu(name='Relu_3'))
        self.add(Dropout(drop_ratio=0.5, name="Dropout_1"))

        self.add(Dense(hidden_size=10, name="Dense_2", weights_init_type="he"))  # (100,10)
        self.add(Relu(name='Relu_4'))
        self.add(Dropout(drop_ratio=0.5, name="Dropout_2"))

        loss_layer = SoftmaxWithLoss()
        self.add(loss_layer, loss=True)

        # 初始化权重参数
        self.init_weights()
