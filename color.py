"""
@ File:     color.py
@ Author:   wzl
@ Datetime: 2019-12-19 14:42
"""
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from labelme import utils

CANNY_LOWER_THRESHOLD = 20  # default 50
CANNY_HIGHER_THRESHOLD = 80  # default 150
GAUSSIAN_KERNEL_SIZE = 5  # default 3
DILATE_EDGE_KERNEL_SIZE = 11  # default 3

FILL = 1


def colored_mask(filename):
    """
    :param filename: image label json file name
    :return: colored mask and origin point
    """
    data = json.load(open(filename))

    img = utils.img_b64_to_arr(data['imageData'])
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

    if 'edge' not in lbl_names.keys() or 'origin' not in lbl_names.keys():
        print("not labeled")
        exit()
    for h in range(0, lbl.shape[0]):
        for w in range(0, lbl.shape[1]):
            if lbl[h, w] == lbl_names['origin']:
                return (lbl == lbl_names['edge']).astype(np.uint8), (h, w)

# colored_mask('query/back/green.json')
