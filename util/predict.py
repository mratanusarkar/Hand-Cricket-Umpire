import cv2
import numpy as np
import math
from .threshold import get_bin_images_from_thresh_list, get_hsv_images_from_thresh_list


def get_player_move(img, bin_thresh, hsv_thresh, debug=False, player=1):

    # define modes and tunings
    mode = "hsv"    # options: "bin", "hsv", "all"

    # return variable
    move_estimate = 0

    # get the threshold images
    bin_images = get_bin_images_from_thresh_list(img, bin_thresh)
    hsv_images = get_hsv_images_from_thresh_list(img, hsv_thresh)

    # combine them to get the ultimate 2D bin image of the hand
    img_bin_combined = np.zeros_like(bin_images[0])
    img_hsv_combined = np.zeros_like(hsv_images[0])

    for i in range(len(bin_images)):
        img_bin_combined = img_bin_combined + bin_images[i]
        img_hsv_combined = img_hsv_combined + hsv_images[i]

    if mode == "bin":
        img_th_combined = img_bin_combined
    if mode == "hsv":
        img_th_combined = img_hsv_combined
    if mode == "all":
        img_th_combined = img_bin_combined + img_hsv_combined

    if debug:
        cv2.imshow("img_bin_combined " + str(player), img_bin_combined)
        cv2.imshow("img_hsv_combined " + str(player), img_hsv_combined)
        cv2.imshow("img_th_combined " + str(player), img_th_combined)

    # dilate and blur to remove the noises
    kernel = np.ones((3, 3), np.uint8)
    img_hand = cv2.dilate(img_th_combined, kernel, iterations=2)
    img_hand = cv2.GaussianBlur(img_hand, (5, 5), 100)

    # find contours
    contours, hierarchy = cv2.findContours(img_hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # take the contour with the biggest area
    contour_hand = max(contours, key=lambda x: cv2.contourArea(x))  # hand

    # approximate the contour
    epsilon = 0.0005 * cv2.arcLength(contour_hand, True)
    approx = cv2.approxPolyDP(contour_hand, epsilon, True)  # hand

    # convex hull around the hand
    hull = cv2.convexHull(contour_hand)

    # calculate the area of the hand and the hull
    area_contour = cv2.contourArea(contour_hand)
    area_hull = cv2.contourArea(hull)

    # calculate area ratio := contour area / hull area
    area_ratio = ((area_hull - area_contour) / area_hull) * 100

    # calibration
    # print("area_hand / area_hull", (area_contour/area_hull))
    # print(area_ratio)

    # find defects in the convex hull w.r.t hand
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    # count the number of defects in the convex hull
    defect_count = 0
    try:
        for defect in defects:
            p1, p2, p3, d = defect[0]
            first_point = tuple(approx[p1][0])
            last_point = tuple(approx[p2][0])
            in_point = tuple(approx[p3][0])

            # find the length of all sides of the triangles
            a = math.sqrt((last_point[0] - first_point[0]) ** 2 + (last_point[1] - first_point[1]) ** 2)
            b = math.sqrt((in_point[0] - first_point[0]) ** 2 + (in_point[1] - first_point[1]) ** 2)
            c = math.sqrt((last_point[0] - in_point[0]) ** 2 + (last_point[1] - in_point[1]) ** 2)

            # using Heron's Formula to find the area
            s = (a + b + c) / 2
            triangle_area = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between the point and the convex hull
            distance = (2 * triangle_area) / a

            # finding out the angle using cosine rule := c^2 = a^2 + b^2 − 2ab cos(C)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # in radian
            angle = angle * 180 / math.pi  # in degree

            if angle <= 90 and distance > 30:
                defect_count += 1
                cv2.circle(img, in_point, 3, [255, 0, 0], -1)

            # draw lines connecting the fingers
            cv2.line(img, first_point, last_point, [0, 255, 0], 2)

            # number of defects = no of fingers-1 in this method.

            # calculate the number of fingers and display the output
            font = cv2.FONT_HERSHEY_SIMPLEX
            line = cv2.LINE_AA
            if defect_count == 0:
                if area_contour < 2000:
                    # cv2.putText(img, "No Hand Detected", (0, 50), font, 2, (0, 0, 255), 3, line)
                    print("No Hand Detected")
                    move_estimate = -1
                else:
                    # zero
                    if area_ratio < 12:
                        # cv2.putText(img, "0", (0, 50), font, 2, (0, 0, 255), 3, line)
                        move_estimate = 0
                    elif area_ratio < 17.5:
                        # cv2.putText(img, "6", (0, 50), font, 2, (0, 0, 255), 3, line)
                        move_estimate = 6
                    else:
                        # cv2.putText(img, "1", (0, 50), font, 2, (0, 0, 255), 3, line)
                        move_estimate = 1
            elif defect_count == 1:
                # cv2.putText(img, "2", (0, 50), font, 2, (0, 0, 255), 3, line)
                move_estimate = 2
            elif defect_count == 2:
                # cv2.putText(img, "3", (0, 50), font, 2, (0, 0, 255), 3, line)
                move_estimate = 3
            elif defect_count == 3:
                # cv2.putText(img, "4", (0, 50), font, 2, (0, 0, 255), 3, line)
                move_estimate = 4
            elif defect_count == 4:
                # cv2.putText(img, "5", (0, 50), font, 2, (0, 0, 255), 3, line)
                move_estimate = 5
            else:
                # cv2.putText(img, "Reposition Your Hand", (0, 50), font, 2, (0, 0, 255), 3, line)
                print("Reposition Your Hand")
                move_estimate = -2

            if debug:
                cv2.imshow("player " + str(player), img)
    except:
        print("No hand and defects fount! error in processing the image!")
        move_estimate = -3

    # return the move, the modified frame, the binary image of the hand
    return move_estimate, img, img_th_combined
