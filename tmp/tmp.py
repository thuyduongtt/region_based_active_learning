import tensorflow as tf

if __name__ == '__main__':
    var_train = tf.compat.v1.trainable_variables()
    print(var_train)
