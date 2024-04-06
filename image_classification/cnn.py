from keras import Input, layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
from numpy import float32, uint8
from numpy.typing import NDArray

# * 双分支网络（待完善）
# * 多头网络（待完善）
# * Inception 模块（待完善）


# 本例子使用线性堆叠网络
model = models.Sequential()
model.add(Input(shape=(28, 28, 1)))

# * 1.1 先创建一个卷积层（Conv2D）
# * 1.1.1 过滤器
# ? 第一个参数为什么称为 filters？过滤器？
# ? 过滤器的个数由什么决定？
# * 1.1.2 卷积核，一般大小为 3 * 3 或者 5 * 5
# ? 卷积核是什么？
# ? 卷积核大小为什么是 3 * 3 ？
# * 1.1.3 激活函数，relu
# ? 激活函数是什么？
# ? 为什么需要激活函数？
# * 1.1.4 输入尺寸，即图像的长宽，最后一个维度为RGB
layer_conv2d_1 = layers.Conv2D(32, (3, 3), activation="relu")
model.add(layer_conv2d_1)

# * 1.2 增加一个池化层（MaxPool2D）
# * 1.1.1 池化大小
# ? 第一个参数为什么称为 filters？过滤器？
layer_pool2d_1 = layers.MaxPool2D(pool_size=(2, 2))
model.add(layer_pool2d_1)

# * 1.3 再增加一个卷积层（Conv2D）
# ? 为什么过滤器个数从 32 变成了 64 ？
layer_conv2d_2 = layers.Conv2D(64, (3, 3), activation="relu")
model.add(layer_conv2d_2)

# * 1.4 再增加一个池化层（MaxPool2D）
layer_pool2d_2 = layers.MaxPool2D(pool_size=(2, 2))
model.add(layer_pool2d_2)

# * 1.5 再增加一个卷积层（Conv2D）
layer_conv2d_3 = layers.Conv2D(64, (3, 3), activation="relu")
model.add(layer_conv2d_3)

model.summary()

# * 1.6 再增加一个展平层（Flatten）
layer_flatten = layers.Flatten()
model.add(layer_flatten)

# * 1.7 展平后将网络输入到一个密集连接分类器网络层（Dense）
layer_dense_1 = layers.Dense(64, activation="relu")
model.add(layer_dense_1)

# * 1.8 最后输入到一个密集连接分类器网络层（Dense）
layer_dense_2 = layers.Dense(10, activation="softmax")
model.add(layer_dense_2)

model.summary()

# * 2 编译模型
# * 2.1 优化器用来调整每轮的参数（权重）
# * 2.2 损失函数用于反馈模型得到的数据和预测值的距离
model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


# * 3 准备输入数据集
# * 3.1 数据集切分为训练集和测试集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images: NDArray[uint8]
train_images: NDArray[float32] = (
    train_images.reshape((60000, 28, 28, 1)).astype(float32) / 255
)

test_images: NDArray[uint8]
test_images: NDArray[float32] = (
    test_images.reshape((10000, 28, 28, 1)).astype(float32) / 255
)

# * 3.2 将输入的分类标签转换为二分类的标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# * 4 拟合模型
# * 4.1 训练批次即训练的轮次，可以根据结果调整该参数
# * 4.2 批次大小即每个批次更改权重的次数
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# * 5 评估模型
test_loss, test_accu = model.evaluate(test_images, test_labels)
print(test_loss, test_accu)
