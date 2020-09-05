
#在很多时候，我们只需要建立一个结构相对简单和典型的神经网络（比如上文中的 MLP 和 CNN），并使用常规的手段进行训练。
# 这时，Keras 也给我们提供了另一套更为简单高效的内置方法来建立、训练和评估模型。


import tensorflow as tf
import numpy as np


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]

# 三种定义网络结构的方式
#  1. 继承 keras.Model, call() 定义任意复杂的网络结构. 之前的章节均采用此模式
#  2. keras.Model 构造函数, 传入 inputs和outputs, 定义任意复杂的网络结构
#  3. keras.model.Sequential(), 只能定义简单的1层传1层的网络结构


# 第2种方式: Functional API 模式建立模型, 更灵活
#
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# 第3种方式: Keras Sequential 方式定义模型
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(100, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10),
#     tf.keras.layers.Softmax()
# ])


# 模型编译
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

num_epochs = 5
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()

# 模型训练
model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)

# 评估
print(model.evaluate(data_loader.test_data, data_loader.test_label))
