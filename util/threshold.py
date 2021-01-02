import cv2
import numpy as np


def get_bin_thresh(img, maxval=255):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    desired = maxval * np.zeros_like(img_gray, np.uint8)
    threshold = 0

    while True:
        _, img_bin = cv2.threshold(img_gray, threshold, maxval, cv2.THRESH_BINARY)
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
    cv2.destroyAllWindows()


def get_bin_images_from_samples(sample_image_list):
    bin_thresh_images = []
    for image in sample_image_list:
        bin_thresh = get_bin_thresh(image)
        bin_img = get_bin_thresh_image(image, bin_thresh)
        bin_thresh_images.append(bin_img)
    return bin_thresh_images


def get_hsv_images_from_samples(sample_image_list):
    hsv_thresh_images = []
    for image in sample_image_list:
        lb, ub = get_hsv_thresh(image)
        img_hsv = get_hsv_thresh_image(image, lb, ub)
        hsv_thresh_images.append(img_hsv)
    return hsv_thresh_images


def get_bin_images_from_thresh_list(image, bin_thresh_list):
    """
    :param image: every frame of the video
    :param bin_thresh_list: the binary thresh list obtained at the beginning
    :return:
    """
    bin_thresh_images = []
    for thresh in bin_thresh_list:
        bin_img = get_bin_thresh_image(image, thresh)
        bin_thresh_images.append(bin_img)
    return bin_thresh_images


def get_hsv_images_from_thresh_list(image, hsv_thresh_list):
    """
    :param image: every frame of the video
    :param hsv_thresh_list: the hsv thresh list obtained at the beginning
    :return:
    """
    hsv_thresh_images = []
    for thresh in hsv_thresh_list:
        lb, ub = thresh
        img_hsv = get_hsv_thresh_image(image, lb, ub)
        hsv_thresh_images.append(img_hsv)
    return hsv_thresh_images


def get_bin_thresh_list(sample_image_list):
    bin_thresh_list = []
    for image in sample_image_list:
        bin_thresh = get_bin_thresh(image)
        bin_thresh_list.append(bin_thresh)
    return bin_thresh_list


def get_hsv_thresh_list(sample_image_list):
    hsv_thresh_list = []
    for image in sample_image_list:
        lb, ub = get_hsv_thresh(image)
        hsv_thresh_list.append((lb, ub))
    return hsv_thresh_list


# TODO
def statistical_thresholding(image, number_of_blocks, threshold_type, statistical_measure):
    """
    :param image: expected to be a square image (m x m x 3) or (m x m x 1), where m is even
    :param number_of_blocks: if number_of_blocks = n, the image will be split into n^2 small sub-images
    :param threshold_type: "bin" or "hsv" ?
    :param statistical_measure: "mean", "median" or "mode"?
    :return: compute the threshold for each sub-image of entered type and return it's "mean", "median" or "mode"
    as entered in statistical_measure, which is a much more accurate threshold parameter.
    """
    pass