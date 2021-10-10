# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:27:21 2018
This script is for preparing the input for the active learning, we need to have training data, pool data,
validation data.
@author: s161488
"""
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import image as contrib_image
import cv2

from CONSTS import IM_PAD_WIDTH, IM_PAD_HEIGHT, IM_CHANNEL, val_data_path

path_mom = "DATA/"  # NOTE, NEED TO BE MANUALLY DEFINED


def prepare_train_data(path, select_benign_train, select_mali_train):
    """
    Args:
        path: the path where the data is saved
        select_benign_train: a list of selected benign images
        select_mali_train: a list of selected malignant images
    Ops:
        First, the images, labels. edges, im_index, cls_index can be extracted from the np.load
        images: shape [85, im_h, im_w, 3]
        labels: shape [85, im_h, im_w, 1]
        im_index: shape [85]
        cls_index: shape [85]
        Start_Point will determine how many images are initialized as training image
    Output:
    
    X_train, Y_train
    X_pool, Y_pool
    X_val, Y_val    
    """
    data_set = np.load(path, allow_pickle=True).item()
    images = data_set['image']
    labels = data_set['label']
    edges = data_set['edge']
    imageindex = data_set['ImageIndex']
    classindex = data_set['ClassIndex']
    benign_index = np.where(np.array(classindex) == 1)[0]
    mali_index = np.where(np.array(classindex) == 2)[0]

    # print('benign_index')
    # print(benign_index)  # even indices
    # print('mali_index')
    # print(mali_index)  # odd indices

    choose_index_tr = np.concatenate([benign_index[select_benign_train], mali_index[select_mali_train]], axis=0)
    data_train = extract_diff_data(images, labels, edges, imageindex, classindex, choose_index_tr)

    n_benign = len(benign_index)
    n_malignant = len(mali_index)

    benign_index_left = np.delete(range(n_benign), select_benign_train)
    mali_index_left = np.delete(range(n_malignant), select_mali_train)

    print(f'n_benign = {n_benign}, n_malignant = {n_malignant}')

    n = 10
    remain_benign = n_benign - n
    remain_maglinant = n_malignant - n
    choose_index_pl = np.concatenate([benign_index[benign_index_left[:remain_benign]], mali_index[mali_index_left[:remain_maglinant]]], axis=0)
    data_pl = extract_diff_data(images, labels, edges, imageindex, classindex, choose_index_pl)

    if val_data_path is None:
        choose_index_val = np.concatenate([benign_index[benign_index_left[-5:]], mali_index[mali_index_left[-5:]]], axis=0)
        data_val = extract_diff_data(images, labels, edges, imageindex, classindex, choose_index_val)
    else:
        data_set = np.load(val_data_path, allow_pickle=True).item()
        images = data_set['image']
        labels = data_set['label']
        edges = data_set['edge']
        imageindex = data_set['ImageIndex']
        classindex = data_set['ClassIndex']
        choose_index_val = np.arange(len(images))

        data_val = extract_diff_data(images, labels, edges, imageindex, classindex, choose_index_val)

    return data_train, data_pl, data_val


def prepare_pool_data(path, aug=False):
    data_set = np.load(path, allow_pickle=True).item()
    images = data_set['image']
    labels = data_set['label']
    edges = data_set['edge']
    imageindex = data_set['ImageIndex']
    classindex = data_set['ClassIndex']
    select_benign_train = [0, 1, 2, 3, 4]
    select_mali_train = [2, 4, 5, 6, 7]

    benign_index = np.where(np.array(classindex) == 1)[0]
    mali_index = np.where(np.array(classindex) == 2)[0]

    n_benign = len(benign_index)
    n_malignant = len(mali_index)

    benign_index_left = np.delete(range(n_benign), select_benign_train)
    mali_index_left = np.delete(range(n_malignant), select_mali_train)

    print(f'n_benign = {n_benign}, n_malignant = {n_malignant}')

    n = 10
    remain_benign = n_benign - n
    remain_maglinant = n_malignant - n
    choose_index_pl = np.concatenate(
        [benign_index[benign_index_left[:remain_benign]], mali_index[mali_index_left[:remain_maglinant]]], axis=0)
    data_pl = extract_diff_data(images, labels, edges, imageindex, classindex, choose_index_pl)
    if aug is True:
        targ_height_npy = IM_PAD_HEIGHT  # this is for padding images
        targ_width_npy = IM_PAD_WIDTH  # this is for padding images
        x_image_val, y_label_val, y_edge_val = padding_training_data(data_pl[0], data_pl[1],
                                                                     data_pl[2], targ_height_npy,
                                                                     targ_width_npy)
        data_pl = [x_image_val, y_label_val, y_edge_val]

    return data_pl


def prepare_test_data(path):
    if "_test" not in path:
        print("-------I am loading the data from pool set------")
        return prepare_pool_data(path)
    data_set = np.load(path, allow_pickle=True).item()
    images = data_set['image']
    labels = data_set['label']
    edges = data_set['edge']
    imageindex = data_set['ImageIndex']
    classindex = data_set['ClassIndex']
    return images, labels, edges, imageindex, classindex


def generate_batch(x_image_tr, y_label_tr, y_edge_tr, y_binary_mask_tr, batch_index, batch_size):
    im_group = [x_image_tr, y_label_tr, y_edge_tr, y_binary_mask_tr]
    im_batch = []
    for single_im in im_group:
        _im_batch = single_im[batch_index:(batch_size + batch_index)]
        im_batch.append(_im_batch)
    batch_index = batch_index + batch_size
    return im_batch[0], im_batch[1], im_batch[2], im_batch[3], batch_index


def padding_training_data(x_image, y_label, y_edge, target_height, target_width):
    """Each image has different size, so I need to pad it with zeros to make sure each image has the same size.
       Then I can perform random crop, rotation and other augmentation on per batch instead of per image
    """
    x_im_pad, y_la_pad, y_ed_pad = [], [], []
    num_image = np.shape(x_image)[0]
    for i in range(num_image):
        image_pad, label_pad, edge_pad = padding_zeros(x_image[i], y_label[i], y_edge[i], target_height, target_width)
        x_im_pad.append(image_pad)
        y_la_pad.append(label_pad)
        y_ed_pad.append(edge_pad)
    x_im_pad = np.reshape(x_im_pad, [num_image, target_height, target_width, IM_CHANNEL])
    y_la_pad = np.reshape(y_la_pad, [num_image, target_height, target_width, 1])
    y_ed_pad = np.reshape(y_ed_pad, [num_image, target_height, target_width, 1])
    return x_im_pad, y_la_pad, y_ed_pad


def padding_zeros(image, label, edge, target_height, target_width):
    im_h, im_w, _ = np.shape(image)
    delta_w = target_width - im_w
    delta_h = target_height - im_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    image_pad = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode='constant')
    label_pad = np.pad(label, ((top, bottom), (left, right)), mode='constant')
    edge_pad = np.pad(edge, ((top, bottom), (left, right)), mode='constant')
    return image_pad, label_pad, edge_pad


def extract_diff_data(image, label, edge, im_index, cls_index, choose_index):
    new_data = [[] for _ in range(5)]
    old_data = [image, label, edge, im_index, cls_index]
    for i in choose_index:
        for single_new, single_old in zip(new_data, old_data):
            single_new.append(single_old[i])
    return new_data[0], new_data[1], new_data[2], new_data[3], new_data[4]


def aug_train_data(image, label, edge, binary_mask, batch_size, aug, imshape):
    """This function is used for performing data augmentation. 
    image: placeholder. shape: [Batch_Size, im_h, im_w, 3], tf.float32
    label: placeholder. shape: [Batch_Size, im_h, im_w, 1], tf.int64
    edge: placeholder. shape: [Batch_Size, im_h, im_w, 1], tf.int64
    binary_mask: placeholder. shape: [Batch_Size, im_h, im_w, 1], tf.int64
    aug: bool
    imshape: [targ_h, targ_w, ch]
    Outputs:
    image: [Batch_Size, targ_h, targ_w, 3]
    label: [Batch_Size, targ_h, targ_w, 1]
    edge: [Batch_Size, targ_h, targ_w, 1]
    binary_mask: [Batch_Size, targ_h, targ_w, 1]
    
    """
    image = tf.cast(image, tf.int64)
    bigmatrix = tf.concat([image, label, edge, binary_mask], axis=3)
    target_height = imshape[0].astype('int32')
    target_width = imshape[1].astype('int32')
    if aug is True:
        bigmatrix_crop = tf.random_crop(bigmatrix, size=[batch_size, target_height, target_width, IM_CHANNEL + 3])
        bigmatrix_crop = tf.cond(tf.less_equal(tf.reduce_sum(bigmatrix_crop[:, :, :, IM_CHANNEL + 2]), 10),
                                 lambda: tf.image.resize_image_with_crop_or_pad(bigmatrix, target_height, target_width),
                                 lambda: bigmatrix_crop)
        # instead of judging by label, should do it by the binary mask!
        k = tf.random_uniform(shape=[batch_size], minval=0, maxval=6.5, dtype=tf.float32)
        bigmatrix_rot = contrib_image.rotate(bigmatrix_crop, angles=k)
        image_aug = tf.cast(bigmatrix_rot[:, :, :, 0:IM_CHANNEL], tf.float32)
        label_aug = bigmatrix_rot[:, :, :, IM_CHANNEL]
        edge_aug = bigmatrix_rot[:, :, :, IM_CHANNEL + 1]
        binary_mask_aug = bigmatrix_rot[:, :, :, IM_CHANNEL + 2]
    else:
        bigmatrix_rot = tf.image.resize_image_with_crop_or_pad(bigmatrix, target_height, target_width)
        image_aug = tf.cast(tf.cast(bigmatrix_rot[:, :, :, 0:IM_CHANNEL], tf.uint8), tf.float32)
        label_aug = tf.cast(bigmatrix_rot[:, :, :, IM_CHANNEL], tf.int64)
        edge_aug = tf.cast(bigmatrix_rot[:, :, :, IM_CHANNEL + 1], tf.int64)
        binary_mask_aug = tf.cast(bigmatrix_rot[:, :, :, IM_CHANNEL + 2], tf.int64)
    return image_aug, tf.expand_dims(label_aug, -1), tf.expand_dims(edge_aug, -1), tf.expand_dims(binary_mask_aug, -1)


def collect_test_data(resize=True):
    test_a_path = path_mom + "/Data/QB_test_benign.npy"
    test_b_path = path_mom + "/Data/QB_test_mali.npy"
    image_tot, label_tot = [], []
    target_height, target_width = IM_PAD_HEIGHT, IM_PAD_WIDTH
    for single_path in [test_a_path, test_b_path]:
        data_set = np.load(single_path, allow_pickle=True).item()
        images = data_set['image']
        y_label_pl = data_set['label']
        y_edge_pl = data_set['edge']
        x_image_val = []
        y_label_val = []
        if resize is True:
            for single_im, single_label in zip(images, y_label_pl):
                for _im_, _path_ in zip([single_im, single_label], [x_image_val, y_label_val]):
                    _im_ = cv2.resize(_im_, dsize=(IM_PAD_WIDTH, IM_PAD_HEIGHT), interpolation=cv2.INTER_CUBIC)
                    _path_.append(_im_)
        else:
            x_image_val, y_label_val, y_edge_val = padding_training_data(images, y_label_pl, y_edge_pl, target_height,
                                                                         target_width)
        image_tot.append(x_image_val)
        label_tot.append(y_label_val)
    image_tot = np.concatenate([image_tot[0], image_tot[1]], axis=0)
    label_tot = np.concatenate([label_tot[0], label_tot[1]], axis=0)
    print("The shape of the test images", np.shape(image_tot))
    return image_tot, label_tot
