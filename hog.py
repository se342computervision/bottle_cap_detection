"""
@ File:     hog.py
@ Author:   wzl
@ Datetime: 2019-12-17 15:23
"""
import os
import numpy as np
import cv2
import pickle

from skimage.feature import hog

HOG_WEIGHT = 0.7  # HOG is good
HOG_WEIGHT_LESS = 0.3  # SIFT is good
HEURIS_BACK_SOFTMAX = 50  # heuristic choose BACK

FRONT = 0
BACK = 1
SIDE = 2
NONE = 3

direct_lower_str = ['front', 'back', 'side']


def hog_des(img_query):
    """
    :param img_query: query image lists
    :return: HOG descriptor lists
    """
    # evaluate HOG descriptor
    fd = [[], [], []]
    for direct in range(FRONT, NONE):
        for query in img_query[direct]:
            if isinstance(query, list):
                # rotated image for augmentation
                fd_aug = []
                for query_rot in query:
                    query_temp = cv2.resize(query_rot, (512, 512))
                    fd_query, hog_query = hog(query_temp, orientations=8, pixels_per_cell=(4, 4),
                                              cells_per_block=(1, 1), visualize=True, multichannel=True)
                    fd_aug.append(fd_query)
                fd[direct].append(fd_aug)
            else:
                query_temp = cv2.resize(query, (512, 512))
                fd_query, hog_query = hog(query_temp, orientations=8, pixels_per_cell=(4, 4),
                                          cells_per_block=(1, 1), visualize=True, multichannel=True)
                fd[direct].append(fd_query)
    save_fd(fd)
    return fd


def save_fd(fd):
    """
    :param fd: HOG descriptor lists
    """
    with open('./hog/fd.dat', 'wb') as f:
        pickle.dump(fd, f)


def load_fd():
    """
    :return: HOG descriptor lists
    """
    target = './hog/fd.dat'
    if os.path.getsize(target) > 0:
        with open(target, "rb") as f:
            unpickler = pickle.Unpickler(f)
            fd = unpickler.load()
    return fd


def hog_match(fd, img_query, img_query_name, img_train, softmax_sift=None, good_sift=False):
    """
    :param fd: HOG descriptor lists
    :param img_query: query image lists
    :param img_query_name: query image name lists
    :param img_train: target image
    :param softmax_sift: ensemble matching result in SIFT and HOG
    :return: image type, image matched, matched image json
    """
    train_temp = cv2.resize(img_train, (512, 512))
    fd_train, hog_train = hog(train_temp, orientations=8, pixels_per_cell=(4, 4),
                              cells_per_block=(1, 1), visualize=True, multichannel=True)
    softmax = [[], [], []]
    img_selected = [None, None, None]
    op_min_dis = [[], [], []]
    for direct in range(FRONT, NONE):
        op_min = None
        for fd_query, query, filename in zip(fd[direct], img_query[direct], img_query_name[direct]):
            if isinstance(fd_query, list):
                softmax_tempmax = None
                for fd_query_temp, query_temp in zip(fd_query, query):
                    op = np.linalg.norm(fd_query_temp - fd_train)
                    if op_min is None or op < op_min:
                        op_min = op
                        img_selected[direct] = (query_temp, filename)
                    softmax_temp = np.exp(-op)
                    if softmax_tempmax is None or softmax_temp > softmax_tempmax:
                        softmax_tempmax = softmax_temp
                softmax[direct].append(softmax_tempmax)
            else:
                op = np.linalg.norm(fd_query - fd_train)
                if op_min is None or op < op_min:
                    op_min = op
                    img_selected[direct] = (query, filename)
                softmax[direct].append(np.exp(-op))
        op_min_dis[direct] = op_min
    softsum = np.sum(softmax[FRONT]) + np.sum(softmax[BACK]) + np.sum(softmax[SIDE])
    ensemble_softmax = [[], [], []]
    for direct in range(FRONT, NONE):
        softmax[direct] = softmax[direct] / softsum
        ensemble_softmax[direct] = softmax[direct]
        # if softmax_sift is None:
        #     ensemble_softmax[direct] = softmax[direct]
        # else:
        #     if good_sift is True:
        #         ensemble_softmax[direct] = softmax[direct] * HOG_WEIGHT_LESS + softmax_sift[direct] * (1 - HOG_WEIGHT_LESS)
        #     else:
        #         ensemble_softmax[direct] = softmax[direct] * HOG_WEIGHT + softmax_sift[direct] * (1 - HOG_WEIGHT)
    img_type = NONE
    max_softmax = 0
    img_final_selected = None
    for direct in range(FRONT, NONE):
        if len(ensemble_softmax[direct]) == 0:
            continue
        if max(ensemble_softmax[direct]) > max_softmax:
            max_softmax = max(ensemble_softmax[direct])
            img_type = direct
            img_final_selected = img_selected[direct]

    # heuristic: back SIFT softmax is much larger then front, choose back
    a = HEURIS_BACK_SOFTMAX * max(softmax_sift[FRONT])
    b = max(softmax_sift[BACK])
    if op_min_dis[FRONT] - 3 < op_min_dis[BACK] < op_min_dis[FRONT] + 3 and \
            (min(op_min_dis) > 102 or op_min_dis[FRONT] - 0.5 < op_min_dis[BACK] < op_min_dis[FRONT] + 0.5) \
            and HEURIS_BACK_SOFTMAX * max(softmax[BACK]) > max(softmax[FRONT]) and \
            max(softmax_sift[BACK]) > HEURIS_BACK_SOFTMAX * max(softmax_sift[FRONT]) and \
            (img_type == FRONT or img_type == BACK):
        img_type = BACK
        for query, filename, softmax_temp in zip(img_query[BACK], img_query_name[BACK], softmax_sift[BACK]):
            if softmax_temp == max(softmax_sift[BACK]):
                if isinstance(query, list):
                    img_selected[BACK] = (query[0], filename)
                else:
                    img_selected[BACK] = (query, filename)
        img_final_selected = img_selected[BACK]

    return img_type, img_final_selected[0], img_final_selected[1]
