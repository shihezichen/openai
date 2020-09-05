# 自定义层、损失函数和评估指标
# 可以继承 tf.keras.Model 编写自己的模型类，也可以继承 tf.keras.layers.Layer 编写自己的层。

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


# 使用自定义层替换到Dense层
#  Dense特征: 100个神经元, 激活为relu
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.w = None
        self.b = None

    # 首次调用时调用
    # 输入inputs形如:
    '''
        [
            [x1,x2,x3,x4],  -> 为每个神经元都需要准备 [w1,w2,w3,w4] -> y=w1x1+w2x2+w3x3+w4x4
            [], 
            ...
            []        
        ]
    '''
    # 则 inputs_shape 形如:  [50, 4] ,其中50是 batch_size,代表欧数据为50行, 4为第二维的个数,  [60000, 28, 28, 1]
    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_variable(name='w',
                                   shape=[input_shape[-1], self.units], initializer=tf.random_normal_initializer())
        self.b = self.add_variable(name='b',
                                   shape=[self.units], initializer=tf.random_normal_initializer())

    def call(self, inputs):
        # 线性变换
        y_pred = tf.matmul(inputs, self.w) + self.b
        # 激活函数
        #y_pred = tf.nn.relu(y_pred)
        return y_pred

# 自定义损失函数
# 继承 keras.losses.Loss类, 重写call方法
# y_true 形如下, 它每个元素是个标量,         # 对y_true降维, 只取最内层的.  降维前: [ [1],[2],[5], ...,[7] ] , 降维后: [ 1, 2, 5, ... ,7 ]需要转为one-hot编码:
'''
   y_true = [
           1, --> [0,1,0,0,0,0,0]
           2,
           5,
           ...
   ]
   y_pred = [
       [0,1,0,0,0,0,0],
       ...    
   ]
'''
class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # 对y_true降维, 只取最内层的.  降维前:  y_true=[ [1],[2],[5], ...,[7] ] , 降维后: [ 1, 2, 5, ... ,7 ]
        # 此处y_true会多一维度是TF针对自定义损失/自定义评估时才会出现, 传入自定义函数的的会多一个维度
        y_true = tf.squeeze(y_true, axis=-1)
        # 对 y_true 做one_hot编码
        # 由于one_hot每个元素是整数, 而y_true每个元素是浮点数,因此还需要做类型转化
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=10)
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - y_true)))


# 自定义评估函数
class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    # 每次训练了一批样本后, 传入 update_state 函数
    def update_state(self, y_true, y_pred, sample_weight=None):
        # 对y_true降维, 只取最内层的.  降维前: [ [1],[2],[5], ...,[7] ] , 降维后: [ 1, 2, 5, ... ,7 ]
        #  这样就可以和y_pred进行比较了(维度相同)
        y_true = tf.squeeze(y_true, axis=-1)
        # tf.argmax ()找到最大的那个, 然后和 y_true 比较, 返回 true, false, 再转化为1,0
        #  备注: y_true, y_pred都是浮点数, tf.cast() 转成整数
        values = tf.cast(tf.equal(tf.cast(y_true, tf.int32), tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])   # batch_size
        self.count.assign_add(tf.reduce_sum(values))  # 答对的次数(使用求和可以把为1的数量统计出来)

    def result(self):
        return self.count / self.total



# Keras Sequential 方式定义模型
#
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # 使用自定义的Linearlayer替换tf的Dense层
    #tf.keras.layers.Dense(100, activation=tf.nn.relu),
    LinearLayer(100),
    #tf.keras.layers.Dense(10),
    LinearLayer(10),
    tf.keras.layers.Softmax()
])


# 模型编译
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #loss=tf.keras.losses.sparse_categorical_crossentropy,
    loss=MeanSquaredError(),
    #metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    metrics=[SparseCategoricalAccuracy()]

)

num_epochs = 5
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()

# 模型训练
model.fit(data_loader.train_data, data_loader.train_label, epochs=num_epochs, batch_size=batch_size)

# 评估
print(model.evaluate(data_loader.test_data, data_loader.test_label))
