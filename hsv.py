# contains all pre-processing related to hsv
# TODO
#  contains:
#  * get_hsv_thresh()    input: small square image of (k x k x 3) dimensions
#                        output: lower_bound: [l_h, l_s, l_v], upper_bound: [u_h, u_s, u_v]
#  * get_hsv_image()     input: n image 3D in a list, lower_bound, upper_bound
#                        output: n 2D binary image in a list

import cv2
import numpy as np


def get_hsv_thresh(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    l_h = np.amin(h)
    l_s = np.amin(s)
    l_v = np.amin(v)

    u_h = np.amax(h)
    u_s = np.amax(s)
    u_v = np.amax(v)

    lower_bound = [l_h, l_s, l_v]
    upper_bound = [u_h, u_s, u_v]

    return lower_bound, upper_bound


def get_hsv_thresh_image(img, lower_bound, upper_bound):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_thresh = np.array(lower_bound, dtype=np.uint8)
    upper_thresh = np.array(upper_bound, dtype=np.uint8)

    img_thresh = cv2.inRange(img_hsv, lower_thresh, upper_thresh)

    return img_thresh


def nothing(x):
    pass


def get_hsv_thresh_manual(cap):
    # window to find out the lower and upper threshold values
    cv2.namedWindow("Tracking Console")
    cv2.createTrackbar("Lower Hue", "Tracking Console", 0, 255, nothing)
    cv2.createTrackbar("Lower Sat", "Tracking Console", 0, 255, nothing)
    cv2.createTrackbar("Lower Val", "Tracking Console", 0, 255, nothing)
    cv2.createTrackbar("Upper Hue", "Tracking Console", 255, 255, nothing)
    cv2.createTrackbar("Upper Sat", "Tracking Console", 255, 255, nothing)
    cv2.createTrackbar("Upper Val", "Tracking Console", 255, 255, nothing)

    while True:
        # read image frame
        _, frame = cap.read()

        # convert the color image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # get the threshold values from user
        l_h = cv2.getTrackbarPos("Lower Hue", "Tracking Console")
        l_s = cv2.getTrackbarPos("Lower Sat", "Tracking Console")
        l_v = cv2.getTrackbarPos("Lower Val", "Tracking Console")
        u_h = cv2.getTrackbarPos("Upper Hue", "Tracking Console")
        u_s = cv2.getTrackbarPos("Upper Sat", "Tracking Console")
        u_v = cv2.getTrackbarPos("Upper Val", "Tracking Console")

        # track objects in the video using HSV color thresholds
        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])

        # create a mask with the above thresholds
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # apply the mask to extract the result
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # show the result
        cv2.imshow("mask", mask)
        cv2.imshow("result", result)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            return lower_bound, upper_bound
            break
    cv2.destroyAllWindows()
