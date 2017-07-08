from numpy.random import randint
from numpy import argmax
from keras.utils.np_utils import to_categorical
k = 8
n = 20
x = randint(0, k, (n,))
t = to_categorical(x, k)
print x
print(t)
print(argmax(to_categorical(x, k)))