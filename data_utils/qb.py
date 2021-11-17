# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:43:17 2018
This script is utilized for creating the data 
    data['image'] = image [1, Number_of_Image]
    data['label'] = label [1, Number_of_Image]
    data['edge'] = Edge [1, Number_of_Image]
    data['boundingbox'] = Bounding_Box [1, Number_of_Image, Maximum_Num_of_Instances(30), 4](x_min,x_max,y_min,y_max)
@author: s161488

"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import dilation, disk

from CONSTS import DS_NAME, INVERTED

path_mom = "DATA/"  # the directory for saving data
# path_mom = "/Users/bo/Documents/Exp_Data/"
path_use = path_mom + 'Data'
if not os.path.exists(path_use):
    os.makedirs(path_use)


def get_file_list(split='train'):
    path = Path(path_mom, DS_NAME, split)
    input_dir_1 = 'input1'
    input_dir_2 = 'input2'
    label_dir = 'label'

    train_image_filename = []
    train_label_filename = []

    for regionOrPatch in Path(path, input_dir_1).iterdir():
        if regionOrPatch.name.startswith('.'):
            continue
        if regionOrPatch.is_dir():
            for patch in regionOrPatch.iterdir():
                train_image_filename.append((
                    str(patch),
                    str(Path(path, input_dir_2, regionOrPatch.stem, patch.name)),
                ))
                train_label_filename.append(str(Path(path, label_dir, regionOrPatch.stem, patch.name)))
        else:
            train_image_filename.append((
                str(regionOrPatch),
                str(Path(path, input_dir_2, regionOrPatch.name))
            ))
            train_label_filename.append(str(Path(path, label_dir, regionOrPatch.name)))

    print(f"Total number of {split} images:", len(train_image_filename))

    return train_image_filename, train_label_filename


def read_data(im_list, la_list, split_name='train'):
    """This function is utilized to read the image and label
    """
    images = []
    labels = []
    image_indices = []
    index = 0

    for im_filename, la_filename in zip(im_list, la_list):
        im_file_1, im_file_2 = im_filename
        im1 = Image.open(im_file_1)
        im2 = Image.open(im_file_2)
        im = np.concatenate((im1, im2), axis=-1)

        label_im = Image.open(la_filename)
        if INVERTED:
            la = 255 - np.array(label_im)  # invert image: white pixels indicate changes
        else:
            la = np.array(label_im)

        images.append(im)
        labels.append(la)
        image_indices.append(index)
        index = index + 1

    print(f'{index} {DS_NAME} {split_name} images are loaded')
    return images, labels, image_indices


def extract_edge(label):
    """This function is utilized to extract the edge from the ground truth
    Args:
        label: The ground truth of all images 
        shape [Number_of_image, Image_Height, Image_Width,1]
    Returns:
        The edge feature map. If the pixel belongs to edge, then the label is set to be one. 
        If the pixel doesn't belong to edge, then the label is set to be zero.
        shape: [Number_of_image, Image_height, Image_Width,1]
    
    The requirement for this function is scipy!
    """
    selem = disk(3)
    edge_feat = []
    for la_sep in label:
        sx = ndimage.sobel(la_sep, axis=0, mode='constant')
        sy = ndimage.sobel(la_sep, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        row = (np.reshape(sob, -1) > 0) * 1
        edge_sep = np.reshape(row, np.shape(sob))
        edge_sep = dilation(edge_sep, selem)
        edge_feat.append(edge_sep.astype('int64'))
    return edge_feat


def extract_benign_malignant(im_indices):
    class_indices = []
    for i in im_indices:
        class_indices.append(i % 2 + 1)
    return class_indices


def transfer_data_to_dict(split='train'):
    """This function is utilized to save the original image in a dictionary
    Return:
        data['image'] = image [1, Number_of_Image*4] 85*4
        data['label'] = label [1, Number_of_Image*4]
        data['edge'] = Edge [1, Number_of_Image*4]
    Requirements:
        from collections import defaultdict
    """
    from collections import defaultdict
    tr_im, tr_la = get_file_list(split)
    image, label, image_indices = read_data(tr_im, tr_la)
    cla_ind_fin = extract_benign_malignant(image_indices)
    edge = extract_edge(label)

    data = defaultdict(list)
    data['image'] = image
    data['label'] = label
    data['edge'] = edge
    data['ImageIndex'] = image_indices
    data['ClassIndex'] = cla_ind_fin
    filename = path_mom + f"/Data/{DS_NAME}_{split}.npy"
    if os.path.isfile(filename):
        print("Remove the existing data file", os.remove(filename))
        print("Saving the data in path:", filename.split(".")[0])
    else:
        print("Oh, this is the first time of creating this file")
        print(f"Creating the training data npy for {DS_NAME}")

    np.save(filename.split(".")[0], data)


# The standardeivation for the boudnign box ([ 158.73026619,  241.80872181,  159.79253117,  240.67909876])
def transfer_data_to_dict_test():
    """This function is utilized to save the test image in a dictionary
    Return:
        data['image'] = image [1, Number_of_Image*4] 85*4
        data['label'] = label [1, Number_of_Image*4]
        data['edge'] = Edge [1, Number_of_Image*4]
    Requirements:
        from collections import defaultdict
    """
    from collections import defaultdict
    test_im, test_la = get_file_list(split='test')
    image, label, image_indices = read_data(test_im, test_la, "test")
    cla_ind_fin = extract_benign_malignant(image_indices)

    image_benign = []
    label_benign = []
    image_index_benign = []
    image_mali = []
    label_mali = []
    image_index_mali = []
    for i in image_indices:
        if cla_ind_fin[i] == 1:
            image_benign.append(image[i])
            label_benign.append(label[i])
            image_index_benign.append(image_indices[i])
        else:
            image_mali.append(image[i])
            label_mali.append(label[i])
            image_index_mali.append(image_indices[i])

    edge_benign = extract_edge(label_benign)
    edge_mali = extract_edge(label_mali)

    cla_ind_benign = [1] * len(image_index_benign)
    cla_ind_mali = [2] * len(image_index_mali)

    data = defaultdict(list)
    data['image'] = image_benign
    data['label'] = label_benign
    data['edge'] = edge_benign
    data['ImageIndex'] = image_index_benign
    data['ClassIndex'] = cla_ind_benign
    filename = path_mom + f"/Data/{DS_NAME}_test_benign.npy"
    if os.path.isfile(filename):
        print("Remove the existing data file", os.remove(filename))
        print("Saving the data in path:", filename.split(".")[0])
    else:
        print("Creating QB test data (benign)")

    print("Saving the data in path:", filename.split(".")[0])
    np.save(filename.split(".")[0], data)
    data1 = defaultdict(list)
    data1['image'] = image_mali
    data1['label'] = label_mali
    data1['edge'] = edge_mali
    data1['ImageIndex'] = image_index_mali
    data1['ClassIndex'] = cla_ind_mali
    filename = path_mom + f"/Data/{DS_NAME}_test_mali.npy"
    if os.path.isfile(filename):
        print("Remove the existing data file", os.remove(filename))
        print("Saving the data in path:", filename.split(".")[0])
    else:
        print("Creating QB test data (malignant)")
    print("Saving the data in path:", filename.split(".")[0])
    np.save(filename.split(".")[0], data1)


if __name__ == '__main__':
    transfer_data_to_dict('train')
    transfer_data_to_dict('test')
    transfer_data_to_dict_test()
