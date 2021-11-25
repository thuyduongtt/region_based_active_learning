
if __name__ == '__main__':
    import tensorflow as tf
    print('Start session ...')
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print('Session is started!')

