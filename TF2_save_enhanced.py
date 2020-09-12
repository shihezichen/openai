# 图像增强
# 模型保存

import tensorflow as tf
import matplotlib.pyplot as plt
# 数据增强
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
# 图片处理
from PIL import Image

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

# 图像增强
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

# model装载weights
checkpoint_save_path = './checkpoint/mnist.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-'*20, 'load the model', '-'*20)
    model.load_weights(checkpoint_save_path)

# 保存weight
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
# 训练
# history = model.fit(x_train, y_train, batch_size=6000, epochs=5,
#                     validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])
#model.summary()


# 打印/提取可训练参数
#  threshold表示超过多少行省略显示
# np.set_printoptions(threshold=np.inf)  # np.inf表示无限大
# np.set_printoptions(threshold=6)
# print('-'*20, 'save trainable variable to file', '-'*20)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()
#
# # 显示训练集和验证集的acc和loss曲线
# acc = history.history['sparse_categorical_accuracy']
# val_acc = history.history['val_sparse_categorical_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(loss, label="Training Losss")
# plt.plot(val_loss, label="Validation Loss")
# plt.title("Trainging and Validation Loss")
# plt.legend()
# plt.show()

# 预测
#img_path = input("Path of test picture:")
BASE_DIR='/home/arthur/Study/AI_Pic/MyHandwritePics/'
img = Image.open(BASE_DIR + "test.png")
img = img.resize((28, 28), Image.ANTIALIAS)
img_arr = np.array(img.convert('L'))
# 方法1:  取反, 白底黑字 变为 黑底白字, 仍然是灰度
img_arr = 255 - img_arr
# 方法2: 变为只有黑白色的高对比图片
# for i in range(28):
#     for j in range(28):
#         if img_arr[i][j] < 20-:
#             img_arr[i][j] = 255
#         else:
#             img_arr[i][j] = 0
# 归一化
img_arr = img_arr / 255.0
# 训练时都是batch输入的, 因此要把 (28, 28) 前添加一个维度,变为 (1, 28, 28)
x_predict = img_arr[tf.newaxis, ...]
result = model.predict(x_predict)
pred = tf.argmax(result, axis=1)
print('\n')
tf.print(pred)
