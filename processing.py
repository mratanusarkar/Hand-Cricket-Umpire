# after both the players make their moves,
# main.py passes every roi from the video frame here to get the estimate / prediction
# TODO
#  contains:
#  * get_player_move()    input: image from roi (m x n x 3) dimensions
#                        output: player move: any of the integers (0, 1, 2, ..., 6, -1)

import numpy as np
import math
import cv2
import bin
import hsv


def get_thresholds(sample_image_list):
    thresholds = []
    for image in sample_image_list:
        bin_thresh = bin.get_bin_thresh(image)
        hsv_thresh_low, hsv_thresh_up = hsv.get_hsv_thresh(image)
        entry = (bin_thresh, hsv_thresh_low, hsv_thresh_up)
        thresholds.append(entry)
    return thresholds


def get_player_move(img, bin_thresh, hsv_thresh):
    img_bin_thresh = bin.get_bin_thresh_image(img, bin_thresh)
    cv2.imshow("test", img_bin_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
