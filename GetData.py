import tensorflow as tf
import numpy as np


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 0:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机截取图像
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    distorted_image = tf.image.resize_images(distorted_image, height, width, method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


def inference(image_batch):
    # 预测网络
    logit = None

    return logit


def calc_loss(logit, label_batch):
    # 计算损失函数
    loss = None

    return loss


files = tf.train.match_filenames_once("/path/to/file_pattern-*")  # 搜索相似名称文件路径
filename_queue = tf.train.string_input_producer(files, shuffle=False)  # 文件列表输入队列

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenSequenceFeature([], tf.string),
        'label': tf.FixedLenSequenceFeature([], tf.int64),
        'height': tf.FixedLenSequenceFeature([], tf.int64),
        'width': tf.FixedLenSequenceFeature([], tf.int64),
        'channels': tf.FixedLenSequenceFeature([], tf.int64), })
image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])
image_size = 299
distored_image = preprocess_for_train(decoded_image, image_size, image_size, None)

min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch([distored_image, label], batch_size=batch_size, capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)

learning_rate = 0.01

logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)



