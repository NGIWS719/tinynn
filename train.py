# import sys, os
# sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
from net.convnetv1 import ConvNetV1
from net.lenet5 import LeNet5
from net.alexnet import AlexNet
from dataset.mnist import load_mnist
from layers.utils import save_model
import numpy as np
from matplotlib import pyplot as plt


# 读取mnist数据集并划分为4部分
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
x_train = x_train[:10000]
y_train = y_train[:10000]

# 加载模型并设定其参数
model = LeNet5(x_train, y_train, epochs=5, weight_decay_lambda=0.01, sample_batches=True, optimizer='adam',
               learning_rate_decay=0.95, batch_size=100, optimizer_param={'lr': 0.01}, verbose=True)
# model = AlexNet(x_train, y_train, epochs=5, weight_decay_lambda=0.01, sample_batches=True, optimizer='adam',
#                 learning_rate_decay=0.95, batch_size=64, optimizer_param={'lr': 0.01}, verbose=True)
# model = ConvNetV1(x_train, y_train, epochs=5, weight_decay_lambda=0.01, sample_batches=True, optimizer='adam',
#                   learning_rate_decay=0.95, batch_size=100, optimizer_param={'lr': 0.01}, verbose=True)

# 打印网络结构信息
model.desc()
# 模型训练
model.train()

# 保存模型训练权重
# save_model(model, "./results/lenet5.pkl")
# save_model(model, "./results/alexnet.pkl")
# save_model(model, "./results/convnetv1.pkl")

# 绘制图形
loss_history = np.array(model.loss_history)
acc_history = np.array(model.acc_history)

# 第一个关系图
iters = np.arange(int(model.epochs * model.iter_per_epochs) - 1)
plt.subplot(121)
plt.plot(iters, loss_history[1:], label="dropout")
plt.title("iters_loss")
plt.xlabel("iters")
plt.ylabel("loss")

# 第二个关系图
epochs = np.arange(model.epochs)
plt.subplot(122)
plt.plot(epochs, acc_history, label="dropout")
plt.title("epoch_acc")
plt.xlabel("epoch")
plt.ylabel("acc")

# 调整子图间距
plt.subplots_adjust(wspace=0.4)  # 设置水平间距

# 保存图像
plt.savefig("./results/result.png", dpi=300)

# 显示图像
plt.show()
