import tensorflow as tf
import cv2
import sys
import random
import numpy as np
from collections import deque

#sys.path('game/')

GAME = 'bird'
ACTIONS = 2
OBSERVE = 10**4
EXPLORE = 2*(10**5)
FINAL_EPSILON = 10**(-4)
BATCH = 32

print(FINAL_EPSILON)
print(cv2.__version__)

def createNetwork():
    pass

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fcl = createNetwork()

def main():
    playGame()

if __name__ == 'main':
    main()



