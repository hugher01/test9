# code:utf-8
# 《Tensorflow实战》
# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import pandas as pd
import numpy as np

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data = pd.read_csv('.\\MNIST.train.csv', header=0, dtype=np.int)

sess = tf.InteractiveSession()

in_units = 784
h1_units = 300
w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)

y_ = tf.placeholder(tf.float32, [None,10])
cross_enropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_enropy)

tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = data.values[i:i+100, 1:],data['label'].values[i:i+100]
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

data_test = pd.read_csv('.\\MNIST.test.csv', header=0, dtype=np.int)
batch_test_xs, batch_test_ys = data_test.values[:, 1:],data_test['label'].values
print(accuracy.eval({x: batch_test_xs, y_: batch_test_ys, keep_prob: 1.0}))
# print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
