import cv2
import numpy as np


def nothing(x):
    pass


def get_hsv(cap):
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
        # cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("result", result)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            return lower_bound, upper_bound
            break
    cap.read()
    cv2.destroyAllWindows()
