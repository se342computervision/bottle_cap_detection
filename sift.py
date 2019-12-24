"""
@ File:     sift.py
@ Author:   wzl
@ Datetime: 2019-12-15 22:30
"""
import os
import numpy as np
import cv2
import hog
import color
import rotation

# match threshold
MIN_MATCH_COUNT = 10  # default 10
RATIO_TEST_DISTANCE = 0.7  # default 0.7
FLANN_INDEX_KDTREE = 0  # do not change

FRONT = 0
BACK = 1
SIDE = 2
NONE = 3
direct_str = ['FRONT', 'BACK', 'SIDE']
direct_lower_str = ['front', 'back', 'side']


def sift_init():
    """
    init SIFT descriptors and keypoints for all images in /query
    :return: just pass them to sift_match
    """
    query_img_name = [[], [], []]
    query_img = [[], [], []]
    query_img_hog = [[], [], []]
    kp = [[], [], []]
    des = [[], [], []]
    for direct in range(FRONT, NONE):
        for base_path, folder_list, file_list in os.walk('query/' + direct_lower_str[direct]):
            for file_name in file_list:
                filename = os.path.join(base_path, file_name)
                if filename[-4:] != '.png' and filename[-4:] != '.jpg':
                    continue
                query_img_name[direct].append(filename)
                query_img[direct].append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
                query_img_hog[direct].append(cv2.imread(filename))
            # rotate query img in rotate directory for augmentation
            for folder_name in folder_list:
                if folder_name != 'rotate':
                    continue
                for base_path_rot, folder_list_rot, file_list_rot in os.walk(
                        'query/' + direct_lower_str[direct] + '/rotate'):
                    for file_name_rot in file_list_rot:
                        filename_rot = os.path.join(base_path_rot, file_name_rot)
                        if filename_rot[-4:] != '.png' and filename_rot[-4:] != '.jpg':
                            continue
                        query_img_name[direct].append(filename_rot)
                        query_img[direct].append(cv2.imread(filename_rot, cv2.IMREAD_GRAYSCALE))
                        img_hog_temp = cv2.imread(filename_rot)
                        img_hog_aug = rotation.rotate(img_hog_temp)
                        query_img_hog[direct].append(img_hog_aug)
            break  # only traverse top level

    # Initiate HOG fd
    # fd = hog.hog_des(img_hog)
    fd = hog.load_fd()

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    for direct in range(FRONT, NONE):
        for img_temp, img_name in zip(query_img[direct], query_img_name[direct]):
            kp_temp, des_temp = orb.detectAndCompute(img_temp, None)
            if des_temp is None:
                print(img_name + ": SIFT cannot detect keypoints and descriptor")
                kp[direct].append(None)
                des[direct].append(None)
                continue
            kp[direct].append(kp_temp)
            des[direct].append(des_temp)
    return query_img, query_img_hog, query_img_name, kp, des, fd


# load query images
def sift_match(input_image, query_img, query_img_hog, query_img_name, kp, des, fd):
    """
    :param input_image
    :param query_img
    :param query_img_hog
    :param query_img_name
    :param kp
    :param des
    :param fd
    :return: bottle cap position(FRONT=0, BACK=1, SIDE=2), mask for coloring, origin point on mask
    """
    # Initiate SIFT detector
    orb = cv2.ORB_create()

    img_train = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    img_train_hog = input_image.copy()

    # SIFT feature detect
    kp_train, des_train = orb.detectAndCompute(img_train, None)
    if des_train is None:
        print("SIFT cannot detect keypoints and descriptor")
        # fallback to HOG matching
        selected, img_selected, img_selected_name = hog.hog_match(fd, query_img_hog, query_img_name, img_train_hog)
        print("%s (HOG)\n\n" % direct_str[selected])
        img_mask, origin_point = color.colored_mask(str(img_selected_name.split('.')[0]) + '.json')
        return selected, img_mask, origin_point

    # use FLANN matcher
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = [[], [], []]
    for direct in range(FRONT, NONE):
        for des_temp in des[direct]:
            if des_temp is None:
                matches[direct].append(None)
                continue
            matched = flann.knnMatch(np.asarray(des_temp, np.float32), np.asarray(des_train, np.float32), k=2)
            if len(matched) == 1:
                print("error")
                exit()
            matches[direct].append(matched)

    # store all the good matches as per Lowe's ratio test.
    selected = NONE
    max_match = 0
    matchesMask = None
    img_selected = None
    kp_selected = None
    good_selected = None
    mask_selected = None
    img_selected_name = None
    softmax_sift = [[], [], []]
    for direct in range(FRONT, NONE):
        for img_temp, img_temp_name, kp_temp, des_temp, matches_temp in zip(query_img[direct], query_img_name[direct],
                                                                            kp[direct], des[direct], matches[direct]):
            # SIFT cannot detect this query img, skip it
            if kp_temp is None or des_temp is None or matches_temp is None:
                softmax_sift[direct].append(np.exp(0))
                continue

            good = []
            for m, n in matches_temp:
                if m.distance < RATIO_TEST_DISTANCE * n.distance:
                    good.append(m)

            # softmax evaluation for SIFT match, used in HOG match
            softmax_sift[direct].append(np.exp(len(good)))

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_train[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # no matches is inlier, skip this train image
                if M is None:
                    continue

                matchesMask = mask.ravel().tolist()
                match_sum = np.sum(matchesMask)
                if match_sum > max_match:
                    max_match = match_sum
                    selected = direct
                    # img_selected = img_temp
                    img_selected_name = img_temp_name

    if selected == NONE:
        print("too less matches found by SIFT")
        # fallback to HOG matching
        softmax_sum = sum(softmax_sift[FRONT]) + sum(softmax_sift[BACK]) + sum(softmax_sift[SIDE])
        for direct_temp in range(FRONT, NONE):
            softmax_sift[direct_temp] = softmax_sift[direct_temp] / softmax_sum
        selected, img_selected, img_selected_name = hog.hog_match(fd, query_img_hog, query_img_name,
                                                                           img_train_hog, softmax_sift)
        print("%s (HOG)\n\n" % direct_str[selected])
        img_mask, origin_point = color.colored_mask(str(img_selected_name.split('.')[0]) + '.json')
        return selected, img_mask, origin_point
    else:
        print("%s (SIFT)\n\n" % direct_str[selected])
        img_mask, origin_point = color.colored_mask(str(img_selected_name.split('.')[0]) + '.json')
        return selected, img_mask, origin_point


if __name__ == "__main__":
    input_image0 = cv2.imread("train/DSC02776-7-2868-2668.jpg")
    query_img0, query_img_hog0, query_img_name0, kp0, des0, fd0 = sift_init()
    selected0, img_mask0, origin_point0 = sift_match(input_image0, query_img0, query_img_hog0, query_img_name0, kp0, des0, fd0)
    a = 0
