# code:utf-8
'''
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf

learn = tf.contrib.learn


def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)
    logits, loss = learn.models.logistic_regression(features, target)
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1)
    return tf.arg_max(logits, 1), loss, train_op


iris = datasets.load_iris()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0)
classifier = learn.Estimator(model_fn=my_model)
classifier.fit(x_train, y_train, steps=100)

y_predicted = classifier.predict(x_test)

score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy:%.2f%%' % (score * 100))
'''

import tensorflow as tf
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
learn =tf.contrib.learn

hidden_size=30
num_layers=2
timesteps=10
training_steps=10000
batch_size=32

training_examples=10000
testing_examples=1000
sample_gap=0.01

def generate_data(seq):
    x=[]
    y=[]

    for i in range(len(seq)-timesteps-1):
        x.append([seq[i:i+timesteps]])
        y.append([seq[i + timesteps]])

    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(x,y):
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layers)
    x_=tf.unpack(x,axis=1)
    output,_=tf.nn.rnn(cell,x_,dtype=tf.float32)
    output=output[-1]





