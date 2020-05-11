# contains all pre-processing related to binary
# TODO
#  contains:
#  * get_bin_thresh()    input: small square image of (k x k x 3) dimensions
#                        output: bin_thresh (float)
#  * get_bin_image()     input: n image 3D in a list, threshold value
#                        output: n 2D binary image in a list

import cv2
import numpy as np


def get_bin_thresh(img, maxval=255):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    desired = maxval * np.zeros_like(img_gray, np.uint8)
    threshold = 0

    while True:
        _, img_bin = cv2.threshold(img_gray, threshold, maxval, cv2.THRESH_BINARY_INV)
        if threshold == maxval:
            break
        if np.array_equal(img_bin, desired):
            break
        threshold += 1

    return threshold


def get_bin_thresh_image(img, threshold, maxval=255):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, threshold, maxval, cv2.THRESH_BINARY_INV)
    return img_bin
