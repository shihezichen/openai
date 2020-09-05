# Recurrent Neural Network, RNN

# 循环神经网络（Recurrent Neural Network, RNN）是一种适宜于处理序列数据的神经网络，被广泛用于语言模型、文本生成、机器翻译等。

import tensorflow as tf
import numpy as np

class DataLoader():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt',
            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        # 按字母a-z排序的列表
        self.chars = sorted(list(set(self.raw_text)))
        # （每个字母的序号，字母）
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        # （字母，每个字母的序号）
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        # 把原文本中的字母转成对应的序号
        self.text = [self.char_indices[c] for c in self.raw_text]
          

    # seq_length : 片段的长度，  batch_size: 多少组片段
    # 返回的信息中第一个数组：
    '''
        [
           [41,44,46],  <--- 第 1 组片段，每组片段有 seq_length个元素， 每个元素都是字符的序号indice
           [...],
           ...
           [...]   <--- 第 batch_size 组片段
        ]
    '''
    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            # 每次从文章中截取 seq_length 长度的片段， 起始位置随机
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index+seq_length])
            # 这个片段对应的下一个字符也保存起来
            next_char.append(self.text[index+seq_length])
        # 返回片段和片段下一个字符两个数组
        return np.array(seq), np.array(next_char)       # [batch_size, seq_length], [num_batch]


#
# data_loader = DataLoader()
# batch, y = data_loader.get_batch(3, 10)
# print(batch)
# inputs = tf.one_hot(batch, depth=len(data_loader.chars))  # [batch_size, seq_length, num_chars]
# print("inputs:", inputs[:, 1, :])
# import sys
# sys.exit(0)


'''
one-hot 编码
  [45, 7, 1]  -> 46  变为one-hot编码为：
  [ [0,0,0,...,1,0,0,...],  [0,0,...,1,0,0,...], [...] ]  -> [ 0,0,...,0,1,0,...]   
'''

class RNN(tf.keras.Model):
    # num_chars：  决定了 onehot编码长度
    # batch_size:  样本个数
    # seq_length:  每个样本字符个数
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cell = tf.keras.layers.LSTMCell(units=256)
        # 最后一层的输出由于也是one-hot编码，因此神经元个数也是 num_chars 个
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs, from_logits=False):
        # 对传入的样本做 one-hot 编码, 每个字母序号都变为一个 num_chars 位的one-hot编码
        # 输入: [batch_size, seq_length]
        # 输出: [batch_size, seq_length, num_chars]
        inputs = tf.one_hot(inputs, depth=self.num_chars)       # [batch_size, seq_length, num_chars]
        # 初始状态
        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)   # 获得 RNN 的初始状态
        # 把所有样本的第一个字符, 第二个字符, 第三个字符它们的onehot编码依次录入到 RNN的cell中, 不断刷新state和output
        for t in range(self.seq_length):
            # 输入: inputs[:, t, :] 表示取所有第一个维度(即所有batch_size个样本)的取第2个维度(即onehot编码那个维度)的第t个元素的所有元素(第三个维度)
            # 这样形成了一个数组, 有batch_size个元素, 每个元素是一个数组,数组为num_char个元素,即one-hot编码
            # 输出: 但第三个字符录入后, output即为预测出来的此片段 预测出来的下一个字符(即第四个字符)的编码(它不一定是one-hot,每个位上都有值)
            output, state = self.cell(inputs[:, t, :], state)   # 通过当前输入和前一时刻的状态，得到输出和当前时刻的状态
        logits = self.dense(output)
        if from_logits:                     # from_logits 参数控制输出是否通过 softmax 函数进行归一化
            # 原样输出
            return logits
        else:
            # 把输出处理为概率后输出
            # 即一个向量, 有num_chars个元素,每个元素是概率值, 代表是某个字符的可能性
            #   [0.1, 0.2, 0.01, ..., 0.6, ...]
            return tf.nn.softmax(logits)


    # 自定义的预测函数, 让整个预测不再那么死板, 体现一些灵活性
    # temperature越大, 则logtis向量元素除以它之后, 之间的差异性就越小, 则概率值相差越小, 则选择下一个字符的随机性就越大
    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        # 调用call方法进行预测
        logits = self(inputs, from_logits=True)                         # 调用训练好的RNN模型，预测下一个字符的概率分布
        # 由于call方法没有做概率, 因此手动求概率归一化
        # softmax()会把数值向量变为总和为1的概率向量,体现数值之间的差异化, 最后得到的prob是一个概率值向量[0.01,0,0,...,0.5,000]
        prob = tf.nn.softmax(logits / temperature).numpy()              # 使用带 temperature 参数的 softmax 函数获得归一化的概率分布值
        # 对于每一行样本, 得到其概率,  然后random.choice随机挑选一个字符. 这样每个样本都得到了一个下一个字符的序号编号
        return np.array([np.random.choice(self.num_chars, p=prob[i, :])  # 使用 np.random.choice 函数，
                         for i in range(batch_size.numpy())])

# 迭代1000次训练
num_batches = 1000
# 连续40个字符的短语
seq_length = 40
# 依次取50个短语进行批量训练
batch_size = 50
learning_rate = 1e-3

data_loader = DataLoader()
model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
# 优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for batch_index in range(num_batches):
    # x: 每次取50个样本, 每个样本40个字符
    # y: 50个单字, 每个单字是一个数字, 代表序号
    X, y = data_loader.get_batch(seq_length, batch_size)
    with tf.GradientTape() as tape:
        # 通过RNN进行预测得到y_pred
        y_pred = model(X)
        # 上面输出的y_pred是一个概率值的向量, 使用sparse_xxx 会把y标量变为one-hot向量,以便于和有y_pred比较
        # 交叉熵可以自动求两个向量之间的损失值pychar
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        # 求batch_size(50个)样本损失的均值
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))



# 预测: 提供40个字, 循环预测后400个字
X_, _ = data_loader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:      # 丰富度（即temperature）分别设置为从小到大的 4 个值
    X = X_
    print("diversity %f:" % diversity)
    # 在当前的40个字基础上, 预测出400个字
    for t in range(400):
        y_pred = model.predict(X, diversity)    # 预测下一个字符的编号
        print(data_loader.indices_char[y_pred[0]], end='', flush=True)  # 输出预测的字符
        # 取后39个字, 在加上新预测出来的一个字, 作为新的X
        X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)     # 将预测的字符接在输入 X 的末尾，并截断 X 的第一个字符，以保证 X 的长度不变
    print("\n")