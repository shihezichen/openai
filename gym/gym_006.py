import numpy as np
import math
import random
import matplotlib.pyplot as plt


# 随机产生点x,y, 取值均在范围 [0, size]
def generatePts(size):
    x = random.random() * size
    y = random.random() * size
    return x, y


# 是否在圆内
def isInCircle(pt, r):
    # 到圆心的距离小于半径
    return math.sqrt(pt[0] ** 2 + pt[1] ** 2) <= r

# 画出正方形和1/4圆弧
def draw():
    plt.axes().set_aspect('equal')
    arc = np.linspace(0, np.pi / 2, 100)
    plt.plot(1 * np.cos(arc), 1 * np.sin(arc))

# 随机数方式计算pi
def calcPi():
    # 边长
    a = 1
    # 落入园弧的点的个数
    ptsInCircle = 0
    # 落入正方形的店的个数
    ptsInSquare = 0
    # 样本数量
    samples = 1000

    draw()
    for i in range(samples):
        pt = generatePts(a)
        plt.plot(pt[0], pt[1], 'c.')
        ptsInSquare += 1
        if isInCircle(pt, a):
            ptsInCircle += 1

    plt.show()
    # 计算Pi
    return 4 * ptsInCircle / ptsInSquare


if __name__ == '__main__':
    pi = calcPi()
    print('PI is: {} '.format(pi))
