"""
@ File:     hog.py
@ Author:   wzl
@ Datetime: 2019-12-17 15:23
"""
import os
import numpy as np
import cv2

from skimage.feature import hog

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
    for direct in range(FRONT, NONE):
        i = 0
        for fd_array in fd[direct]:
            if isinstance(fd_array, list):
                j = 0
                for fd_array_temp in fd_array:
                    np.savetxt('./hog/' + direct_lower_str[direct] + str(i) + '-' + str(j)+ '.txt', fd_array_temp, delimiter=',')
                    j += 1
            else:
                np.savetxt('./hog/'+direct_lower_str[direct]+str(i)+'.txt', fd_array, delimiter = ',')
            i += 1

def load_fd():
    """
    :return: HOG descriptor lists
    """
    fd = [[], [], []]
    for direct in range(FRONT, NONE):
        i = 0
        filename = './hog/'+direct_lower_str[direct]+str(i)+'.txt'
        filename_aug = './hog/'+direct_lower_str[direct]+str(i)+'-0.txt'
        while os.path.exists(filename) or os.path.exists(filename_aug):
            if os.path.exists(filename_aug):
                assert not os.path.exists(filename)
                assert not os.path.exists('./hog/'+direct_lower_str[direct]+str(i)+'-12.txt')
                fd_array = []
                for rotidx in range(0,12):
                    filename_temp = './hog/'+direct_lower_str[direct]+str(i)+'-'+str(rotidx)+'.txt'
                    fd_array_temp = np.loadtxt(filename_temp,dtype=np.float64)
                    fd_array.append(fd_array_temp)
                fd[direct].append(fd_array)
            else:
                fd_array = np.loadtxt(filename,dtype=np.float64)
                fd[direct].append(fd_array)
            i += 1
            filename = './hog/' + direct_lower_str[direct] + str(i) + '.txt'
            filename_aug = './hog/' + direct_lower_str[direct] + str(i) + '-0.txt'
    return fd

def hog_match(fd, img_query, img_query_name, img_train):
    """
    :param fd: HOG descriptor lists
    :param img_query: query image lists
    :param img_query_name: query image name lists
    :param img_train: target image
    :return: image type, image matched, matched image json, softmax list
    """
    train_temp = cv2.resize(img_train, (512, 512))
    fd_train, hog_train = hog(train_temp, orientations=8, pixels_per_cell=(4, 4),
                              cells_per_block=(1, 1), visualize=True, multichannel=True)
    softmax = [[], [], []]
    img_selected = [None, None, None]
    for direct in range(FRONT, NONE):
        op_min = None
        for fd_query, query, filename in zip(fd[direct], img_query[direct], img_query_name[direct]):
            if isinstance(fd_query, list):
                for fd_query_temp in fd_query:
                    op = np.linalg.norm(fd_query_temp - fd_train)
                    if op_min is None or op < op_min:
                        op_min = op
                        img_selected[direct] = (query, filename)
                    softmax[direct].append(np.exp(op))
            else:
                op = np.linalg.norm(fd_query-fd_train)
                if op_min is None or op < op_min:
                    op_min = op
                    img_selected[direct] = (query, filename)
                softmax[direct].append(np.exp(op))
    softsum = np.sum(softmax[FRONT])+np.sum(softmax[BACK])+np.sum(softmax[SIDE])
    for direct in range(FRONT, NONE):
        softmax[direct] = softmax[direct] / softsum
    img_type = NONE
    min_softmax = 1
    img_final_selected = None
    for direct in range(FRONT, NONE):
        if min(softmax[direct]) < min_softmax:
            min_softmax =min(softmax[direct])
            img_type = direct
            img_final_selected = img_selected[direct]
    return img_type, img_final_selected[0], img_final_selected[1], softmax

