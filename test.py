# -*- coding: utf-8 -*-
# @Time : 2023/10/4 18:35
# @Author : 杨军
# @Email : gpu0@qq.com
# @File : test.py
# @Project : tiny_nn
from dataset.mnist import load_mnist
from layers.utils import load_model

# 加载模型权重文件
model = load_model("./results/lenet5.pkl")

# 加载并划分数据集
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
x_test = x_test[:1000]
y_test = y_test[:1000]

# 测试数据集
model.test(x_test, y_test)

# 预测数据集
y = model.predict(x_test)
print(y)
predicted_labels = model.predicted_label(x_test)
print(predicted_labels)
print(y_test)
