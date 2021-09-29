import tensorflow.compat.v1 as tf

if __name__ == '__main__':
    var_train = tf.trainable_variables()
    print(var_train)
