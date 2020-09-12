import tensorflow as tf
import matplotlib.pyplot as plt
# 数据增强
from tensorflow.keras.preprocessing.image import ImageDataGenerator

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test  /255.0

# 图像增强
image_gen_train = ImageDataGenerator(
    rescale=1. /1.,  # 如果位图像, 分母为255时, 可桂枝0~1
    rotation_range=45,   # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,    # 高度偏移
    horizontal_flip=True,   # 水平翻转
    zoom_range=.5         # 图像随机缩放50%
)

print("Before shape: ", x_train[0].shape)
# 给数据增加一个维度, 从(60000, 28, 28 ) -> (60000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
print("After shape: ", x_train[0].shape)
image_gen_train.fit(x_train)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=1024, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()

