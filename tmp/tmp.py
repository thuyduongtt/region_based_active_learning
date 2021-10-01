import numpy as np
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


if __name__ == '__main__':
    y_pred = np.random.rand(20, 20, 1)
    y_true = np.random.rand(20, 20, 1)

    y_pred[y_pred >= 0.5] = 1.
    y_pred[y_pred < 0.5] = 0.

    y_true[y_true >= 0.5] = 1.
    y_true[y_true < 0.5] = 0.

    dice_vs_jaccard(y_pred, y_true)
