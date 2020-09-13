import tensorflow as tf

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test  /255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()

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
