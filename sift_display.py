"""
@ File:     sift_display.py
@ Author:   wzl
@ Datetime: 2019-12-19 12:32
"""
import os
import numpy as np
import cv2
import hog
# import color
import rotation
from matplotlib import pyplot as plt

# use HOG match?
HOG_APP = 1
# rebuild HOG descriptor?
REBUILD_HOG = 0

# match threshold
MIN_MATCH_COUNT = 10  # default 10
RATIO_TEST_DISTANCE = 0.7  # default 0.7

FRONT = 0
BACK = 1
SIDE = 2
NONE = 3
direct_str = ['FRONT', 'BACK', 'SIDE']
direct_lower_str = ['front', 'back', 'side']

# load query images
imgname = [[], [], []]
img = [[], [], []]
img_hog = [[], [], []]
for direct in range(FRONT, NONE):
    for base_path, folder_list, file_list in os.walk('query/' + direct_lower_str[direct]):
        for file_name in file_list:
            filename = os.path.join(base_path, file_name)
            if filename[-4:] != '.png' and filename[-4:] != '.jpg':
                continue
            imgname[direct].append(filename)
            img[direct].append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
            img_hog[direct].append(cv2.imread(filename))
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
                    imgname[direct].append(filename_rot)
                    img[direct].append(cv2.imread(filename_rot, cv2.IMREAD_GRAYSCALE))
                    img_hog_temp = cv2.imread(filename_rot)
                    img_hog_aug = rotation.rotate(img_hog_temp)
                    img_hog[direct].append(img_hog_aug)
        break  # only traverse top level

# get HOG descriptor
if HOG_APP == 1:
    if REBUILD_HOG == 1:
        fd = hog.hog_des(img_hog)
    else:
        fd = hog.load_fd()
else:
    fd = None

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp = [[], [], []]
des = [[], [], []]
for direct in range(FRONT, NONE):
    for img_temp, img_name in zip(img[direct], imgname[direct]):
        kp_temp, des_temp = orb.detectAndCompute(img_temp, None)
        if des_temp is None:
            print(img_name + ": SIFT cannot detect keypoints and descriptor")
            kp[direct].append(None)
            des[direct].append(None)
            continue
        kp[direct].append(kp_temp)
        des[direct].append(des_temp)

# use FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

for base_path, folder_list, file_list in os.walk('train'):
    for file_name in file_list:
        filename = os.path.join(base_path, file_name)
        if filename[-4:] != '.png' and filename[-4:] != '.jpg':
            continue
        img_train = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img_train_hog = cv2.imread(filename)

        img_train_matched = None
        kp_train, des_train = orb.detectAndCompute(img_train, None)
        if des_train is None or len(des_train) < 2:
            print(filename + ": SIFT cannot detect keypoints and descriptor")
            if HOG_APP == 1:
                # fallback to HOG matching
                selected, img_selected, img_selected_name = hog.hog_match(fd, img_hog, imgname, img_train_hog)
                print("%s is %s (HOG)" % (filename, direct_str[selected]))
                # img_mask, origin_point = color.colored_mask(str(img_selected_name.split('.')[0])+'.json')
                img_output = cv2.drawMatches(img_selected, None, img_train, None, None, None, None)
                plt.imshow(img_output, 'gray'), plt.show()
            continue
        matches = [[], [], []]
        for direct in range(FRONT, NONE):
            for des_temp in des[direct]:
                if des_temp is None:
                    matches[direct].append(None)
                    continue
                matched = flann.knnMatch(np.asarray(des_temp, np.float32), np.asarray(des_train, np.float32), k=2)
                # if len(matched) == 1:
                #     print("FLANN match error")
                #     matches[direct].append(None)
                #     continue
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
        good_sift = False
        assert len(img) == len(img_hog) == len(imgname) == len(kp) == len(des) == len(matches)
        if HOG_APP:
            assert len(img) == len(fd)
        for direct in range(FRONT, NONE):
            for img_temp, img_temp_name, kp_temp, des_temp, matches_temp in zip(img[direct], imgname[direct],
                                                                                kp[direct], des[direct],
                                                                                matches[direct]):
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

                        h, w = img_temp.shape
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        mask_selected = matchesMask
                        img_selected = img_temp
                        img_selected_name = img_temp_name
                        kp_selected = kp_temp
                        good_selected = good
                        img_train_matched = cv2.polylines(img_train, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        if selected == NONE:
            print("%s: Not enough matches are found by SIFT" % filename)
            mask_selected = None

            if HOG_APP == 1:
                softmax_sum = sum(softmax_sift[FRONT]) + sum(softmax_sift[BACK]) + sum(softmax_sift[SIDE])
                for direct_temp in range(FRONT, NONE):
                    softmax_sift[direct_temp] = softmax_sift[direct_temp] / softmax_sum
                # fallback to HOG matching
                selected, img_selected, img_selected_name = hog.hog_match(fd, img_hog, imgname, img_train_hog,
                                                                          softmax_sift)
                print("%s is %s (HOG)" % (filename, direct_str[selected]))
                # img_mask, origin_point = color.colored_mask(str(img_selected_name.split('.')[0]) + '.json')
                img_output = cv2.drawMatches(img_selected, None, img_train, None, None, None, None)
                plt.imshow(img_output, 'gray'), plt.show()
        else:
            print("%s is %s (SIFT)" % (filename, direct_str[selected]))
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=mask_selected,  # draw only inliers
                               flags=2)

            # get 3 keypoints, do linear transformation
            # kps_src = []
            # kps_dst = []
            # for mask, good_match in zip(mask_selected, good_selected):
            #     if mask == 1:
            #         kps_src.append(kp_selected[good_match.queryIdx])
            #         kps_dst.append(kp_train[good_match.trainIdx])
            #         if len(kps_src) >= 3:
            #             break
            #
            # pts_src = np.float32([kps_src[0].pt, kps_src[1].pt, kps_src[2].pt])
            # pts_dst = np.float32([kps_dst[0].pt, kps_dst[1].pt, kps_dst[2].pt])
            # M = cv2.getAffineTransform(pts_src, pts_dst)
            # dst = cv2.warpAffine(img_selected, M, (img_selected.shape[0], img_selected.shape[1]))
            # cv2.imshow('image', dst)
            # plt.imshow(dst, 'warp'), plt.show()

            # img_mask, origin_point = color.colored_mask(str(img_selected_name.split('.')[0]) + '.json')
            img_output = cv2.drawMatches(img_selected, kp_selected, img_train_matched, kp_train, good_selected, None,
                                         **draw_params)
            plt.imshow(img_output, 'gray'), plt.show()
