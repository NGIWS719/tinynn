# Introduction to Tinynn
English | [简体中文](README.zh-cn.md)

Tinynn is an extremely lightweight deep learning framework that minimizes the running components of constructed neural networks, allowing you to easily, freely, and quickly build your network for learning purposes only.

## ~~Install~~

python version >= 3.8

~~`python setup.py install`~~

## Start

```
git clone https://github.com/
cd tiny_nn
pip install -r requirements.txt
python train.py
python test.py
```

```
=================================================================================
layer                         desc                                               
=================================================================================
Conv_1(6@5*5)                 input：(10000, 1, 32, 32)	 output：(10000, 6, 28, 28)
--------------------------------------------------------------------------------------
Relu_1                        input：(10000, 6, 28, 28)	 output：(10000, 6, 28, 28)
--------------------------------------------------------------------------------------
MaxPooling_1(2*2)             input：(10000, 6, 28, 28)	 output：(10000, 6, 14, 14)
--------------------------------------------------------------------------------------
Conv_2(16@5*5)                input：(10000, 6, 14, 14)	 output：(10000, 16, 10, 10)
--------------------------------------------------------------------------------------
Relu_2                        input：(10000, 16, 10, 10)	 output：(10000, 16, 10, 10)
--------------------------------------------------------------------------------------
MaxPooling_1(2*2)             input：(10000, 16, 10, 10)	 output：(10000, 16, 5, 5)
--------------------------------------------------------------------------------------
Dense_1(hiddien_nums:120)     input：(10000, 16, 5, 5)	 output：(10000, 120)
--------------------------------------------------------------------------------------
Relu_3                        input：(10000, 120)	 output：(10000, 120)
--------------------------------------------------------------------------------------
Dense_2(hiddien_nums:84)      input：(10000, 120)	 output：(10000, 84)
--------------------------------------------------------------------------------------
Relu_4                        input：(10000, 84)	 output：(10000, 84)
--------------------------------------------------------------------------------------
Dense_3(hiddien_nums:10)      input：(10000, 84)	 output：(10000, 10)
--------------------------------------------------------------------------------------
Starting training for 5 epochs...
 epoch: 1 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 2.296824, train_acc: 0.268600, lr: 1.000000e-02 
 epoch: 2 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 2.103145, train_acc: 0.332300, lr: 9.500000e-03 
 epoch: 3 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 2.048311, train_acc: 0.348400, lr: 9.025000e-03 
 epoch: 4 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 2.025569, train_acc: 0.363700, lr: 8.573750e-03 
 epoch: 5 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 2.016455, train_acc: 0.382900, lr: 8.145062e-03 
End of training...
Spend time on：0:01:17
=============== Final Test Accuracy ===============
test acc:0.374700
```

## layers

All layers inherit the Layer class

***forward：Layer.forward(x)***

| argument  | explanation          | type  | choose  |
| ---- | -------- | ---- |-----|
| x    | input  | tensor  | need    |

***backward：Layer.backward(dout)***

| argument | explanation                  | type |  choose  |
| ---- |------------------------------| ---- | ---- |
| dout | the data from the next layer | tensor | need |

***Initialization parameters：Layer.init_weights(input_shape)***

| argument        | explanation         | type  | choose |
| ----------- |---------------------|-------|--------|
| input_shape | shape of input data | tuple | need   |

### Fully connected

**Dense(hidden_size,weights_init_type="normal",name="Dense")**

Example：

```python
from layers.dense import Dense
import numpy as np
x = np.random.randn(100,256,6,6)
dense = Dense(hidden_size=4096) 
dense.init_weights(x.shape) 
out = dense.forward(x) 
print("forward:",out.shape)
dout = np.random.randn(*out.shape)
dout = dense.backward(dout) 
print("backward:",dout.shape)
```

```
Output：
forward: (100, 4096)
backward: (100, 256, 6, 6)
```

| argument              | explanation                                                                                                                                                                                                               | type         | choose |
| ----------------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|--------|
| hidden_size       | number of hidden size                                                                                                                                                                                                     | int          | need   |
| weights_init_type | the weight initialization method defaults to "normal" and can be selected from "he" or "xavier". It is recommended to use "xavier" when the activation function is sigmoid, and "he" when the activation function is relu | float/string | optional     |
| name              | nickname, default to 'Dense'                                                                                                                                                                                              | string       | optional     |

### Convolution

**Conv2D(kernels_num ,kernel_h = 5,kernel_w = 5,stride = 1,padding = 0,weights_init_type="normal",name="Conv")**

Example：

```python
from layers.convolution import Conv2D
import numpy as np
con = Conv2D(kernels_num=64,kernel_h=3,kernel_w=3,stride=1,padding=1)
imgs = np.random.randint(low=0,high=256,size=(1000,32,3,3))
con.init_weights(imgs.shape)
print("forward:\n")
print("input_img_shape:",imgs.shape)
out = con.forward(imgs)
print("out_shape:",out.shape)
print("backward:\n")
dout = np.random.randint(low=0,high=256,size=out.shape)
dout = con.backward(dout)
print("dout_shape:",dout.shape)
```

Output：

```
forward:

input_img_shape: (1000, 32, 3, 3)
out_shape: (1000, 64, 3, 3)
backward:

dout_shape: (1000, 32, 3, 3)
```

| argument              | explanation                                                                                                                                                                                                                | type         | choose |
| ----------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|--------|
| kernels_num       | number of convolutional kernels                                                                                                                                                                                            | int          | need   |
| kernel_h          | convolutional kernel height, default to 5                                                                                                                                                                                  | int          | optional     |
| kernel_w          | convolutional kernel width, default to 5                                                                                                                                                                                   | int          | optional     |
| stride            | convolutional kernel sliding step size, default to 1                                                                                                                                                                       | int          | optional     |
| padding           | padding,default to 0                                                                                                                                                                                                       | int          | optional     |
| weights_init_type | The weight initialization method defaults to 'normal' and can be selected from 'he' or 'xavier'. It is recommended to use 'xavier' when the activation function is sigmoid, and 'he' when the activation function is relu. | float/string | optional     |
| name              | nickname, default to 'Conv2D'                                                                                                                                                                                              | string       | optional     |

### MaxPooling

**MaxPooling(pool_h = 2,pool_w = 2,stride = 1,padding = 0,name="MaxPooling")**

Example：

```python
from layers.maxpool import MaxPooling
import numpy as np
maxpool = MaxPooling(stride=2)
imgs = np.random.randint(low=0,high=256,size=(1000,64,3,3))
print("forward:\n")
print("input_img_shape:",imgs.shape)
out = maxpool.forward(imgs)
print("out_shape:",out.shape)
print("backward:\n")
dout = np.random.randint(low=0,high=256,size=out.shape)
dout = maxpool.backward(dout)
print("dout_shape:",dout.shape)  
```

Output：

```
forward:

input_img_shape: (1000, 64, 3, 3)
out_shape: (1000, 64, 1, 1)
backward:

dout_shape: (1000, 64, 3, 3)
```

| argument    | explanation                       | type | choose |
| ------- |-----------------------------------|------|--------|
| pool_h  | Pooling height, default to 2      | int  | optional     |
| pool_w  | Pooling width, default to 2       | int  | optional     |
| stride  | sliding step size, default to 1   | int  | optional     |
| padding | padding,default to 0              | int  | optional     |
| name    | nickname, default to 'MaxPooling' | int  | optional     |



### Activation function

**Relu:**

```python
from layers.relu import Relu
import numpy as np
relu = Relu()
x = np.random.randint(low=-10, high=10,size=(10,))
print("x:",x)
out = relu.forward(x)
print("forward:",out)
dout = np.random.randint(low=-10, high=10,size=(10,))
print("dout:",dout)
dout = relu.backward(out)
print("backward:",dout)
```

Output：

```
x: [ 7  5  0  0  6  5  9 -3 -2  2]
forward: [7 5 0 0 6 5 9 0 0 2]
dout: [  2  -9  -4 -10   5   3   9  -6  -1  -3]
backward: [7 5 0 0 6 5 9 0 0 2]
```

**Sigmoid：**

```python
from layers.sigmoid import Sigmoid
import numpy as np
sigmoid = Sigmoid()
x = np.random.randint(low=-10, high=10,size=(5,))
print("x:",x)
out = sigmoid.forward(x)
print("forward:",out)
dout = np.random.randint(low=-10, high=10,size=(5,))
print("dout:",dout)
dx = sigmoid.backward(dout)
print("backward:",dx)
```

Output：

```
x: [-6  2 -7  8 -9]
forward: [2.47262316e-03 8.80797078e-01 9.11051194e-04 9.99664650e-01
 1.23394576e-04]
dout: [  9   9   3  -8 -10]
backward: [ 0.02219858  0.94494227  0.00273066 -0.0026819  -0.00123379]
```



### Batch Normalization

**BatchNormalization(gamma, beta, momentum=0.9, running_mean=None, running_var=None,name="Batch Normalization")**

Example：

```python
from layers.batchnormalization import BatchNormalization
import numpy as np
batch = BatchNormalization(gamma=0.5,beta=0.6)
imgs = np.random.randint(low=0,high=256,size=(100,3,28,28))
print("forward:\n")
print("input_img_shape:",imgs.shape)
out = batch.forward(imgs)
print("out_shape:",out.shape)
print("backward:\n")
dout = np.random.randint(low=0,high=256,size=out.shape)
dout = batch.backward(dout)
print("dout_shape:",dout.shape)
```

Output：

```
forward:

input_img_shape: (100, 3, 28, 28)
out_shape: (100, 3, 28, 28)
backward:

dout_shape: (100, 3, 28, 28)
```



| argument         | explanation                      | type   | choose   |
| ------------ |----------------------------------|--------|----------|
| gamma        |                                  | float  | need     |
| beta         |                                  | float  | need     |
| momentum     | default to 0.9                   | float  | optional       |
| running_mean | mean under test conditions       | float  | optional |
| running_var  | variance under test conditions   | float  | optional       |
| name         | default to “Batch Normalization” | string | optional       |



### Dropout

**Dropout(drop_ratio = 0.5,name="Doupout")**

Example：

```python
from layers.dropout import Dropout
dropout = Dropout(drop_ratio = 0.5,name="Doupout_1")
```



## net

### Model

**Model(x_train, t_train, epochs=30, weight_decay_lambda=0, sample_batches=True,**

​         **batch_size=100, optimizer='SGD', optimizer_param={'lr': 0.01}, learning_rate_decay=1, verbose=True)**

Attribute：

| attribute         | explanation                              |
| ------------ |------------------------------------------|
| loss_history | record the loss value for each iteration |
| acc_history  | record the accuracy of each epoch        |

Initialization parameters：

| argument                | explanation                                                                                              | type    | choose   |
| ------------------- |----------------------------------------------------------------------------------------------------------|---------|----------|
| x_train             | train data                                                                                               | tensor  | need     |
| t_train             | train dada label                                                                                         | tensor  | need     |
| epochs              | number of training epochs, default to 30                                                                 | int     | optional |
| weight_decay_lambda | weight attenuation coefficient, default to 0                                                             | float   | optional |
| sample_batches      | whether to train according to mini batch, default to True. If False, all data will be calculated at once | boolean | optional |
| batch_size          | mini-batch size, default to 100                                                                          | int     | optional |
| optimizer           | gradient descent optimizer, default to sgd, optional options include: momentum, adagrad, adam            | string  | optional |
| optimizer_param     | optimizer parameters, such as learning rate, etc; Default to {'lr ': 0.01}                               | dict    | optional |
| learning_rate_decay | learning rate attenuation coefficient, default to 1                                                      | float   | optional |
| verbose             | print training progress, default to True                                                                 | boolean | optional |

Method：

**Model.add(layer,loss=False)**

Add a layer to the model. The layer belongs to the Layer class, and loss=False indicates that the added layer is not the layer of the loss function. For example:

```python
from net.model import Model
model = Model(x_train,y_train)
model.add(Dense(hidden_size=100))
model.add(Relu())
```

**Model.init_weights()**

Initialize the parameters of each layer in the model. Note that this method must be called first when using forward propagation

**Model.desc()**

Output detailed information for each layer. Note that this method requires calling init_ Use after the weight method

**Model.predict(x,train_flg=False)**

Predict the input x, train_flg=False indicates that this prediction represents training. If during the training process, train_flg needs to be set to True to return the predicted value

**Model.accuracy(x,t)**

Calculate accuracy, where x is the input data, t is the correct unlabeled value, and returns a floating-point number

**Model.test(x_test,t_test)**


Calculate the accuracy of the test set and return a floating point number

**Model.train()**

Start training

### AlexNet

Tinynn has implemented AlexNet, which can be directly used

**AlexNet(x_train, y_train, epochs=5, weight_decay_lambda=0, sample_batches=True,** 

​        **batch_size=64, optimizer="SGD", optimizer_param={"lr": 0.01}, learning_rate_decay=1, verbose=True)**

Consistency between parameters and model

Example：

```python
from net.alexnet import AlexNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
x_train = x_train[:10000]
t_train = t_train[:10000]

model = AlexNet(x_train,t_train,epochs=5,optimizer='adam',weight_decay_lambda=0.01,learning_rate_decay=0.95)
model.train()
model.desc()
```

**LeNet5.train()**

Start training

**LeNet5.getModel()**

Returns a Model object

**LeNet5.predict(x)**

Predicting x

**LeNet5.desc()**

Print network details

## optimizers

SGD:stochastic gradient descent

```python
from optimizers.sgd import SGD
op = SGD(lr=0.001)
op.update(model) 
```

Momentum:

```python
from optimizers.momentum import Momentum
op = Momentum(lr=0.001,momentum=0.9)
op.update(model) 
```

AdaGrad：

```python
from optimizers.adagrad import AdaGrad
op = AdaGrad(lr=0.001)
op.update(model) 
```

Adam:

```python
from optimizers.adam import Adam
op = Adam(lr=0.001, beta1=0.9, beta2=0.999)
op.update(model) 
```

## Model Saving and Loading

### Model Save

```python
from layers.utils import save_model
save_model(model,"model.pkl")
```

### Model Load

```python
from layers.utils import load_model
model = load_model("model.pkl")
```

## Reference
- [https://www.cnblogs.com/yj179101536/p/16523220.html](https://www.cnblogs.com/yj179101536/p/16523220.html)
- [https://www.cnblogs.com/yj179101536/p/16424497.html](https://www.cnblogs.com/yj179101536/p/16424497.html)
- [https://zhuanlan.zhihu.com/p/78713744](https://zhuanlan.zhihu.com/p/78713744)
