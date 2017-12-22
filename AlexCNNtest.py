# 有问题
from datetime import datetime
import math
import time
import tensorflow as tf

max_steps = 3000
batch_size = 128
# image_size=24
image_size = 224
data_dir = 'D:\\tmp\\cifar10_data\\cifar-10-batches-bin'


# 取得训练及测试数据
# import cifar10, cifar10_input
# images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
# images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

def print_activtions(t):
    print(t.op.name, '', t.get_shape().as_list())


def loss_cal(fc3, labels):
    labels = tf.cast(labels, tf.float32)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(fc3), reduction_indices=[1]))
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3, labels=labels,
    #                                                                name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def inference(images, labels):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activtions(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activtions(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activtions(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activtions(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activtions(pool5)

    with tf.name_scope('fc1') as scope:
        kernel = tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc = tf.matmul(pool5_flat, kernel)
        bias = tf.nn.bias_add(fc, biases)
        fc1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(fc1)
        keep_prob = tf.placeholder(tf.float32)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        kernel = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        fc = tf.matmul(fc1_drop, kernel)
        bias = tf.nn.bias_add(fc, biases)
        fc2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(fc2)

        keep_prob = tf.placeholder(tf.float32)
        fc2_drop = tf.nn.dropout(fc2, keep_prob)

    with tf.name_scope('fc3') as scope:
        kernel = tf.Variable(tf.truncated_normal([4096, batch_size], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[batch_size], dtype=tf.float32), trainable=True, name='biases')
        fc = tf.matmul(fc2_drop, kernel)
        bias = tf.nn.bias_add(fc, biases)
        fc3 = tf.nn.softmax(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(fc3)

    loss = loss_cal(fc3, labels)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(fc3, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # top_k_op = tf.nn.in_top_k(fc3, labels, 1)

    return loss, train_op, accuracy, parameters


def run_benchmark():
    import numpy as np
    with tf.Graph().as_default():
        true_count = 0
        step = 0
        total_sample_count = max_steps * batch_size

        images = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
        labels = tf.placeholder(tf.int32, [batch_size])

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        while step < max_steps:
            # 取数据
            images = tf.Variable(tf.random_normal([batch_size,
                                                   image_size,
                                                   image_size, 3],
                                                  dtype=tf.float32,
                                                  stddev=1e-1))

            labels = tf.Variable(tf.ceil(tf.random_normal([batch_size,
                                                           1,
                                                           1, 1],
                                                          dtype=tf.float32)))

            # images, labels = sess.run([images_train, labels_train])
            start_time = time.time()
            loss, train_op, accuracy, parameters = inference(images, labels)
            sess.run([loss, train_op, accuracy, parameters], feed_dict={images: images, labels: labels})
            duration = time.time() - start_time
            true_count += np.sum(accuracy)
            step += 1
            precision = true_count / total_sample_count
            print('precision @ 1=%.3f' % precision)

            if step % 10 == 0:
                example_per_sec = batch_size / duration
                sec_per_batch = float(duration)

                format_str = ('step %d,loss= %.2f (%.1f example/sec; %.3f sec/batch)')
                print(format_str % (step, loss, example_per_sec, sec_per_batch))

    num_examples = 1000
    import math
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0

    while step < num_iter:
        image_batch = tf.Variable(tf.random_normal([batch_size,
                                                    image_size,
                                                    image_size, 3],
                                                   dtype=tf.float32,
                                                   stddev=1e-1))

        label_batch = tf.Variable(tf.ceil(tf.random_normal([batch_size,
                                                            1,
                                                            1, 1],
                                                           dtype=tf.int32)))
        # image_batch,label_batch=sess.run([images_test,labels_test])
        loss, train_op, accuracy, parameters = inference(images, labels)
        loss, train_op, accuracy, parameters = sess.run([loss, train_op, accuracy, parameters],
                                                        feed_dict={images: image_batch, labels: label_batch})
        true_count += np.sum(accuracy)
        step += 1

    precision = true_count / total_sample_count
    print('precision @ 1=%.3f' % precision)


if __name__ == '__main__':
    run_benchmark()
