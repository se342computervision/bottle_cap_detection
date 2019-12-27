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


def colored_mask(filename, side_flipped=False):
    """
    :param filename: image label json file name
    :param side_flipped: flip SIDE mask
    :return: colored mask and origin point
    """
    data = json.load(open(filename))

    img = utils.img_b64_to_arr(data['imageData'])
    # lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
    lbl_names = dict()
    lbl_names['origin'] = 1
    lbl_names['edge'] = 2
    lbl = utils.shapes_to_label(img.shape, data['shapes'], lbl_names)
    data_origin = []
    for item in data['shapes']:
        if item['label'] == 'origin':
            data_origin.append(item)
    # lbl0, lbl_names0 = utils.labelme_shapes_to_label(img.shape, data_origin)
    lbl_names0 = dict()
    lbl_names0['origin'] = 1
    lbl0 = utils.shapes_to_label(img.shape, data_origin, lbl_names0)

    if 'edge' not in lbl_names.keys() or 'origin' not in lbl_names.keys():
        print("%s not labeled", filename)
        exit()
    for h in range(0, lbl.shape[0]):
        for w in range(0, lbl.shape[1]):
            if lbl[h, w] == lbl_names['origin'] or lbl0[h, w] == lbl_names0['origin']:
                mask = (lbl == lbl_names['edge']).astype(np.uint8) + (lbl == lbl_names['origin']).astype(np.uint8)
                origin_point = (h, w)
                if side_flipped:
                    mask = np.rot90(mask, 2)
                    origin_point = (img.shape[0] - h, img.shape[1] - w)
                return mask, origin_point
    print("%s not labeled", filename)
    exit()


if __name__ == "__main__":
    colored_mask('query/back/test.json', False)
