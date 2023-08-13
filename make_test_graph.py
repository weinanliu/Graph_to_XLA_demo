#!/bin/python

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def muladd(x, y, basis):
    print("trace!")
    a1 = math_ops.matmul(x, y, name = "x_mul_y")
    return math_ops.add(a1, basis, name = "out")

x = tf.constant([[1,2,3],[4,5,6]])
y = tf.constant([
    [7,8,9,10],
    [11,12,13,14],
    [15,16,17,18]])
basis = tf.constant([
    [19,20,21,22],
    [23,24,25,26]])


print("no JIT!!!")
out = muladd(x,y,basis)
print(out)


jit_muladd = tf.function(muladd, jit_compile = True)
print("JIT first!!!")
out = jit_muladd(x,y,basis)
print(out)
print("JIT second!!!")
out = jit_muladd(x,y,basis)
print(out)


g = tf.Graph()
with g.as_default():
    x_shape = tf.TensorShape([2, 3])
    y_shape = tf.TensorShape([3, 4])
    basis_shape = tf.TensorShape([2, 4])
    x = array_ops.placeholder(dtypes.float32, shape = x_shape, name = "x_hold")
    y = array_ops.placeholder(dtypes.float32, shape = y_shape, name = "y_hold")
    basis = array_ops.placeholder(dtypes.float32, shape = basis_shape, name = "basis")
    print("Trace GRAPH!!!")
    muladd(x, y, basis)
    with open("test_graph.pbtxt", 'w') as f:
        f.write(g.as_graph_def().__str__())
    print("Trace OK!!!")




