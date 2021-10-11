import time

import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import jaccard_score


def dice_vs_jaccard(y_pred, y_true):
    y_true_f = y_true.reshape([1, 20 * 20])
    y_pred_f = y_pred.reshape([1, 20 * 20])
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))

    jaccard = jaccard_score(y_true_f[0], y_pred_f[0])

    print(dice)
    print(jaccard)


def test_img():
    arr = np.random.rand(256, 256, 1)
    arr2 = (arr * 255).astype(np.uint8).squeeze()
    print(arr2.shape)
    img = Image.fromarray(arr2)
    img.save('test.png')


def test_file_time():
    n = 10000

    start_time = time.time()
    f = open('log_1.txt', 'a')
    for i in range(n):
        print(i, file=f)
    f.close()
    end_time = time.time()
    time_1 = end_time - start_time

    start_time = time.time()
    for i in range(n):
        with open('log_2.txt', 'a') as f:
            print(i, file=f)
    end_time = time.time()
    time_2 = end_time - start_time

    print(time_1)
    print(time_2)
    print(f'{time_2 / time_1:.0f}')


if __name__ == '__main__':
    test_file_time()
