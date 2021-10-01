import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    arr = np.arange(75).reshape([3, 5, 5])
    arr2 = arr.reshape([-1])
    print(arr)
    print(arr2)
