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


if __name__ == '__main__':
    import time

    t = time.time()
    print(t)
    print(int(t))
