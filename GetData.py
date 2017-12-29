import tensorflow as tf

def preprocess_for_train(image,height,width,bbox):
    if bbox is None:
        bbox=tf.constant([0.0,0.0,1.0,1.0],dtype=tf.float32,shape=[1,1,4])

    if image.dtype!=tf.float32:
        image=tf.image.convert_image_dtype(image,dtype=tf.float32)

    bbox_begin,bbox_size,_=



files = tf.train.match_filenames_once("/path/to/file_pattern-*")  # 搜索相似名称文件路径
filename_queue = tf.train.string_input_producer(files, shuffle=False)  # 文件列表输入队列

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features=tf.parse_single_example(
    serialized_example,
    features={
        'image':tf.FixedLenSequenceFeature([],tf.string),
        'label':tf.FixedLenSequenceFeature([],tf.int64),
        'height':tf.FixedLenSequenceFeature([],tf.int64),
        'width':tf.FixedLenSequenceFeature([],tf.int64),
        'channels':tf.FixedLenSequenceFeature([],tf.int64),})
image,label=features['image'],features['label']
height,width=features['height'],features['width']
channels=features['channels']

decoded_image=tf.decode_raw(image,tf.uint8)
decoded_image.set_shape([height,width,channels])
image_size=299
# distored_image=prep







