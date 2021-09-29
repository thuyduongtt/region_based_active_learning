import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    arr = np.arange(10).reshape(2, 5)
    print(arr)
    s = tf.math.reduce_sum(arr)
    print(s)
