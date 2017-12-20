from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activtions(t):
    print(t.op.name, '', t.get_shape().as_list())

def inference(images):
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
        kernel = tf.Variable(tf.truncated_normal([13*13*256,4096], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, 13*13*256])
        fc = tf.matmul(pool5_flat, kernel)
        bias = tf.nn.bias_add(fc, biases)
        fc1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(fc1)

    with tf.name_scope('fc2') as scope:
        kernel = tf.Variable(tf.truncated_normal([4096,4096], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
        fc = tf.matmul(fc1, kernel)
        bias = tf.nn.bias_add(fc, biases)
        fc2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(fc2)

    with tf.name_scope('fc3') as scope:
        kernel = tf.Variable(tf.truncated_normal([4096,1000], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32), trainable=True, name='biases')
        fc = tf.matmul(fc1, kernel)
        bias = tf.nn.bias_add(fc, biases)
        fc3 = tf.nn.softmax(bias, name=scope)
        parameters += [kernel, biases]
        print_activtions(fc3)

    return fc3, parameters

def loss(logits,labels):
    labels=tf.cast(labels,tf.int64)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

loss=loss(logits,label_holder)
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op=tf.nn.in_top_k(logits,label_holder,1)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time=time.time()
    image_batch,label_batch=sess.run([images_train,labels_train])
    _,loss_value=sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
    duration=time.time()-start_time

    if step%10==0:
        example_per_sec=batch_size/duration
        sec_per_batch=float(duration)

        format_str=('step %d,loss= %.2f (%.1f example/sec; %.3f sec/batch)')
        print(format_str % (step,loss_value,example_per_sec,sec_per_batch))

num_examples=1000
import math
num_iter=int(math.ceil(num_examples/batch_size))
true_count=0
total_sample_count=num_iter*batch_size
step=0
while step<num_iter:
    image_batch,label_batch=sess.run([images_test,labels_test])
    predictions=sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
    true_count+=np.sum(predictions)
    step+=1

precision=true_count/total_sample_count
print('precision @ 1=%.3f'%precision)
# 0.695

def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s:step %d,duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s:%s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string, num_batches, mn, sd))

def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        pool5, parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, pool5, "Forward")

        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward-backward")

run_benchmark()
