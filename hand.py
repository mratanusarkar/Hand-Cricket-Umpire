import numpy as np
import math
import cv2

# video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():

    # read frame
    ret, frame = cap.read()

    # flip image frame
    # img = frame[:, ::-1, :]
    img = cv2.flip(frame, 1)

    # BGR to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of hsv skin color
    lower_skin = np.array([0, 0, 40], dtype=np.uint8)
    upper_skin = np.array([150, 80, 160], dtype=np.uint8)

    # extract image portion that has skin color (hand)
    img_skin = cv2.inRange(img_hsv, lower_skin, upper_skin)

    # dilate and blur to remove the noises
    kernel = np.ones((3, 3), np.uint8)
    img_hand = cv2.dilate(img_skin, kernel, iterations=4)
    img_hand = cv2.GaussianBlur(img_hand, (5, 5), 100)

    # find contours
    contours, hierarchy = cv2.findContours(img_hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # take the contour with the biggest area
    contour_hand = max(contours, key=lambda x: cv2.contourArea(x))  # hand

    # approximate the contour
    epsilon = 0.0005 * cv2.arcLength(contour_hand, True)
    approx = cv2.approxPolyDP(contour_hand, epsilon, True)          # hand

    # convex hull around the hand
    hull = cv2.convexHull(contour_hand)

    # calculate the area of the hand and the hull
    area_contour = cv2.contourArea(contour_hand)
    area_hull = cv2.contourArea(hull)

    # calculate area ratio := contour area / hull area
    area_ratio = ((area_hull - area_contour) / area_hull) * 100
    # print(area_ratio)

    # find defects in the convex hull w.r.t hand
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    # count the number of defects in the convex hull
    defect_count = 0
    for defect in defects:
        p1, p2, p3, d = defect[0]
        first_point = tuple(approx[p1][0])
        last_point = tuple(approx[p2][0])
        in_point = tuple(approx[p3][0])

        # find the length of all sides of the triangles
        a = math.sqrt((last_point[0] - first_point[0])**2 + (last_point[1] - first_point[1])**2)
        b = math.sqrt((in_point[0] - first_point[0]) ** 2 + (in_point[1] - first_point[1]) ** 2)
        c = math.sqrt((last_point[0] - in_point[0]) ** 2 + (last_point[1] - in_point[1]) ** 2)

        # using Heron's Formula to find the area
        s = (a + b + c) / 2
        triangle_area = math.sqrt(s*(s-a)*(s-b)*(s-c))

        # distance between the point and the convex hull
        distance = (2 * triangle_area) / a

        # finding out the angle using cosine rule := c^2 = a^2 + b^2 − 2ab cos(C)
        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))   # in radian
        angle = angle * 180 / math.pi                           # in degree

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
                cv2.putText(img, "No Hand Detected", (0, 50), font, 2, (0, 0, 255), 3, line)
            else:
                # zero
                if area_ratio < 12:
                    cv2.putText(img, "0", (0, 50), font, 2, (0, 0, 255), 3, line)
                elif area_ratio < 17.5:
                    cv2.putText(img, "6", (0, 50), font, 2, (0, 0, 255), 3, line)
                else:
                    cv2.putText(img, "1", (0, 50), font, 2, (0, 0, 255), 3, line)
        elif defect_count == 1:
            cv2.putText(img, "2", (0, 50), font, 2, (0, 0, 255), 3, line)
        elif defect_count == 2:
            cv2.putText(img, "3", (0, 50), font, 2, (0, 0, 255), 3, line)
        elif defect_count == 3:
            cv2.putText(img, "4", (0, 50), font, 2, (0, 0, 255), 3, line)
        elif defect_count == 4:
            cv2.putText(img, "5", (0, 50), font, 2, (0, 0, 255), 3, line)
        else:
            cv2.putText(img, "Reposition Your Hand", (0, 50), font, 2, (0, 0, 255), 3, line)

    # display the frame
    cv2.imshow('frame', img)

    # quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()
