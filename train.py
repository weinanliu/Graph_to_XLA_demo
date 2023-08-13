#!/bin/python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.python.platform import tf_logging 

def generate_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10)
        ])


def compile_model(model):
    model.compile(
            optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),
            jit_compile = True
            )
    return model

print("1")

tf_logging.set_verbosity(1)

print("COMPILING MODEL!!!!")
model = compile_model(generate_model())

print("LOADING DATA!!!!")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
two_images = x_train[0:2]
two_images_label = y_train[0:2]

print("TRAIN TWO!!!!")
model.fit(x = two_images, y = two_images_label)

print("PREDICTING ONE!!!!")
out = model.predict(two_images)
print("input:")
print(two_images)
print("output:")
print(out)
