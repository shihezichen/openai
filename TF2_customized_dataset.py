from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

BASE_DIR = '/home/arthur/Downloads/MOOC_TF2.1/class4/class4/FASHION_FC/fashion_image_label/'
TRAIN_PATH = BASE_DIR + 'fashion_train_jpg_60000'
TRAIN_LABEL = BASE_DIR + 'fashion_train_jpg_60000.txt'
TEST_PATH = BASE_DIR + 'fashion_test_jpg_10000'
TEST_LABEL = BASE_DIR + 'fashion_test_jpg_10000.txt'
X_TRAIN_SAVE_PATH = BASE_DIR + 'x_train_save.npy'
Y_TRAIN_SAVE_PATH = BASE_DIR + 'y_train_save.npy'
X_TEST_SAVE_PATH = BASE_DIR + 'x_test_save.npy'
Y_TEST_SAVE_PATH = BASE_DIR + 'y_test_save.npy'


# 从原始文件加载生成数据集
def generate_dataset(path, lable_file):
    f = open(lable_file, 'r')
    contents = f.readlines()
    f.close()
    x, y_ = [], []
    for content in contents:
        img_name, label_value = content.split()
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path)
        # 变为8位宽灰度值的np.array格式
        img = np.array(img.convert('L'))
        # 数据归一化
        img = img / 255.0
        x.append(img)
        y_.append(label_value)

    x = np.array(x)
    # 变为np.array 并变为64位整数
    y_ = np.array(y_).astype(np.int64)

    return x, y_

# 转载数据, 如果有保存, 从保存文件加载, 否则从原始文件生成数据集
def load_data():
    global TRAIN_PATH, TRAIN_LABEL, TEST_PATH, TEST_LABEL
    global X_TRAIN_SAVE_PATH, Y_TRAIN_SAVE_PATH, X_TEST_SAVE_PATH, Y_TEST_SAVE_PATH
    is_saved = os.path.exists(X_TRAIN_SAVE_PATH) and os.path.exists(X_TRAIN_SAVE_PATH) \
              and os.path.exists(Y_TEST_SAVE_PATH) and os.path.exists(Y_TEST_SAVE_PATH)
    if is_saved:
        print('-' * 20, 'Load Datasets', '-' * 20)
        x_train = np.load(X_TRAIN_SAVE_PATH)
        x_test = np.load(X_TEST_SAVE_PATH)
        y_train = np.load(Y_TRAIN_SAVE_PATH)
        y_test = np.load(Y_TEST_SAVE_PATH)
    else:
        print('-'*20, 'Generate Datasets', '-'*20)
        x_train, y_train = generate_dataset(TRAIN_PATH, TRAIN_LABEL)
        x_test, y_test = generate_dataset(TEST_PATH, TEST_LABEL)

        print('-'*20, 'Save Datasets', '-'*20)
        x_train = np.reshape(x_train, (len(x_train), -1))
        x_test = np.reshape(x_test, (len(x_test), -1))
        np.save(X_TRAIN_SAVE_PATH, x_train)
        np.save(X_TEST_SAVE_PATH, x_test)
        np.save(Y_TRAIN_SAVE_PATH, y_train)
        np.save(Y_TEST_SAVE_PATH, y_test)

    return (x_train, y_train), (x_test, y_test)



if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    model.fit(x_train, y_train, batch_size=6000, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()