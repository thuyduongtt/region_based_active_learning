# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:56:26 2018

@author: s161488
"""
############################################################
#  Loss Functions
############################################################
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, precision_score, recall_score, jaccard_score
from PIL import Image
from pathlib import Path
# import time

# save_dir = 'images'


def calc_score(pred, label, func):
    """
    pred: [batch_size, im_h, im_w, 1]
    label: [batch_size, im_h, im_w, 1]
    """

    # pred_to_save = pred[0]
    # label_to_save = label[0]

    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    score = func(y_true=label, y_pred=pred)

    ## EXPORT IMAGES FOR DEBUGGING
    # print(func.__name__)
    # unique_pred, count_pred = np.unique(pred, return_counts=True)
    # unique_label, count_label = np.unique(label, return_counts=True)
    # print(f'pred : {pred.min()}, {pred.max()}, {dict(zip(unique_pred, count_pred))}')
    # print(f'label: {label.min()}, {label.max()}, {dict(zip(unique_label, count_label))}')
    #
    # # export to images
    # if not Path(save_dir).exists():
    #     Path(save_dir).mkdir()
    #
    # timestamp = time.time()

    # img_pred = Image.fromarray(pred_to_save * 255.)
    # img_pred.save(f'{save_dir}/{timestamp}_{func.__name__}_pred.png')
    # img_label = Image.fromarray(label_to_save * 255.)
    # img_label.save(f'{save_dir}/{timestamp}_{func.__name__}_label.png')

    return score.astype('float32')


def tf_f1_score(pred, label):
    def _f1_score(pd, lb):
        return calc_score(pd, lb, f1_score)

    f1score_tensor = tf.py_func(_f1_score, [pred, label], tf.float32)
    return f1score_tensor


def tf_accuracy_score(pred, label):
    def _accuracy_score(pd, lb):
        return calc_score(pd, lb, accuracy_score)

    f1score_tensor = tf.py_func(_accuracy_score, [pred, label], tf.float32)
    return f1score_tensor


def tf_precision_score(pred, label):
    def _precision_score(pd, lb):
        return calc_score(pd, lb, precision_score)

    f1score_tensor = tf.py_func(_precision_score, [pred, label], tf.float32)
    return f1score_tensor


def tf_recall_score(pred, label):
    def _recall_score(pd, lb):
        return calc_score(pd, lb, recall_score)

    f1score_tensor = tf.py_func(_recall_score, [pred, label], tf.float32)
    return f1score_tensor


def tf_jaccard_score(pred, label):
    def _jaccard_score(pd, lb):
        return calc_score(pd, lb, jaccard_score)

    f1score_tensor = tf.py_func(_jaccard_score, [pred, label], tf.float32)
    return f1score_tensor


def calc_auc_score(pred, label):
    """
    In this function, the pred is not just predicted label, instead it's the predicted probability
    pred: [batch_size, im_h, im_w, 1]
    label: [batch_size, im_h, im_w, 1]
    """
    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    fpr, tpr, thresholds = roc_curve(y_true=label, y_score=pred,
                                     pos_label=1)
    auc_value = auc(fpr, tpr)
    return auc_value.astype('float32')


def tf_auc_score(pred, label):
    auc_score_tensor = tf.py_func(calc_auc_score, [pred, label], tf.float32)
    return auc_score_tensor


def Loss(logits, labels, binary_mask, auxi_weight, loss_name):
    """Calculate the cross entropy loss for the edge detection.
    Args:
    logits: The logits output from the edge detection channel. Shape [6, Num_Batch, Image_Height, Image_Width, 1] (6 is because 5 
    side-output, and 1 fuse-output). 
    labels: The edge label. Shape [Num_Batch, Image_height, Image_Width, 1].
    DCAN: A boolean variable. If DCAN is True. The last channel we use softmax_cross_entropy. If it's False, then last feature map
    we use sigmoid_loss. 
    
    Returns:
    Edge_loss: the total loss for the edge detection channel.
    
    Operations:
    Because there is a class imbalancing situation, the pixels belong to background must be much larger than the pixels belong to
    edge, so that we add a penalty beta. beta = Y_/Y. 
    """
    y = tf.reshape(labels, [-1])
    y = tf.cast(tf.not_equal(y, 0), tf.int32)
    Num_Map = np.shape(logits)[0]
    cost = 0
    binary_mask = tf.reshape(binary_mask, [-1])  # [batch_size, imh, imw, 1]
    binary_mask = tf.cast(tf.not_equal(binary_mask, 0), tf.int32)
    for i in range(Num_Map):
        cross_entropy_sep = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(logits[i], [-1, 2]),
                                                                           labels=y)
        cross_entropy_sep = tf.boolean_mask(cross_entropy_sep, tf.equal(binary_mask, 1))
        cross_entropy_sep = tf.reduce_mean(cross_entropy_sep, name='auxiliary' + loss_name + '%d' % i)
        tf.add_to_collection('loss', cross_entropy_sep)
        cost += cross_entropy_sep
    fuse_map = tf.add_n(logits)
    fuse_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=tf.reshape(fuse_map, [-1, 2]))
    fuse_cost = tf.reduce_mean(tf.boolean_mask(fuse_cost, tf.equal(binary_mask, 1)), name=loss_name + 'fuse_cost')
    tf.add_to_collection('loss', fuse_cost)
    cost = auxi_weight * cost + fuse_cost

    pred = tf.reshape(tf.argmax(fuse_map, axis=-1, output_type=tf.int32), [-1])

    # binarize
    pred_bi = tf.boolean_mask(pred, tf.equal(binary_mask, 1))
    y_bi = tf.boolean_mask(y, tf.equal(binary_mask, 1))

    pred_bi_cond_f1 = tf.equal(tf.reduce_sum(pred_bi), 0)
    y_bi_cond_f1 = tf.equal(tf.reduce_sum(y_bi), 0)

    accuracy = tf.cond(tf.logical_and(pred_bi_cond_f1, y_bi_cond_f1),
                       lambda: tf.constant(1.0),
                       lambda: tf_f1_score(pred=pred_bi, label=y_bi))

    auc_pred_bi = tf.boolean_mask(tf.reshape(fuse_map[:, :, :, 1], [-1]), tf.equal(binary_mask, 1))
    auc_score = tf.cond(tf.equal(tf.reduce_mean(y_bi), 1),
                        lambda: tf.constant(0.0),
                        lambda: tf_auc_score(pred=auc_pred_bi, label=y_bi))

    # other metrics: accuracy_score, precision_score, recall_score, jaccard_score
    accuracy_score_val = tf.cond(tf.logical_and(pred_bi_cond_f1, y_bi_cond_f1),
                                 lambda: tf.constant(1.0),
                                 lambda: tf_accuracy_score(pred=pred_bi, label=y_bi))
    precision_score_val = tf.cond(tf.logical_and(pred_bi_cond_f1, y_bi_cond_f1),
                                  lambda: tf.constant(1.0),
                                  lambda: tf_precision_score(pred=pred_bi, label=y_bi))
    recall_score_val = tf.cond(tf.logical_and(pred_bi_cond_f1, y_bi_cond_f1),
                               lambda: tf.constant(1.0),
                               lambda: tf_recall_score(pred=pred_bi, label=y_bi))
    jaccard_score_val = tf.cond(tf.logical_and(pred_bi_cond_f1, y_bi_cond_f1),
                                lambda: tf.constant(1.0),
                                lambda: tf_jaccard_score(pred=pred_bi, label=y_bi))

    metrics = {
        'accuracy_score': accuracy_score_val,
        'precision_score': precision_score_val,
        'recall_score': recall_score_val,
        'jaccard_score': jaccard_score_val
    }

    return cost, accuracy, auc_score, metrics


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('loss')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    return loss_averages_op


def train_op_batchnorm(total_loss, global_step, initial_learning_rate, lr_decay_rate, decay_steps, epsilon_opt, var_opt,
                       MOVING_AVERAGE_DECAY):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    lr_decay_rate,
                                    staircase=False)
    # if staircase is True, then the division between global_step/decay_steps is a interger,
    # otherwise it's not an interger.
    tf.summary.scalar('learning_rate', lr)

    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # variables_averages_op = variables_averages.apply(var_opt)
    with tf.control_dependencies(update_ops):
        loss_averages_op = _add_loss_summaries(total_loss)
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr, epsilon=epsilon_opt)
            grads = opt.compute_gradients(total_loss, var_list=var_opt)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = variables_averages.apply(var_opt)

    return train_op
