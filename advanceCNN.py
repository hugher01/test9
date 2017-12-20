git clone https://github.com/tensorflow/models.git
cd models/tutorials/image/cifar10

# import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps=3000
batch_size=128
data_dir='/tmp/cifar10_data/cifar-10-batches-bin'

def variable_with_weight_loss(shape,stddev,wl):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))





