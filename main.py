########################################################################################################################
########################################################################################################################
#
# TODO:
#  ===================================================== BEGINNING =====================================================
#  ********* phase 1: (get thresholds) *********
#  1. show Start Screen
#  2. split the screen into 2 regions for player-1 & player-2
#  3. create a square (roi) in both the regions for player to place their hands
#  4. ask them to place their hands into the square to take n-samples (small squares)
#  5. pass these n-samples (small square image of (k x k x 3) dimensions) to bin_thresh() & hsv_thresh()
#     to get the min & max / lower & upper threshold values for bin & hsv
#  6. when ready, display commentary of "game begins"
#  ...
#  ====================================================== GAME ON ======================================================
#  ********* phase 2: (get estimate) *********
#  1. in every move
#  2. every turn display "3 2 1 go..." or something similar for the players to play at the same time
#  3. take the snapshot of the hands (2) and send it to an evaluator / predictor function
#  4. evaluator / predictor function sends the images to bin_hand() and hsv_hand() to get b number of bin-thresh images
#     and h number of hsv-thresh images (all 2D binary images)
#  5. combine these (b+h) number of images in a smart way (add / subtract / and / or /... etc) to get perfect hand image
#  6. remove noise from hand_img
#  7. apply morphological transformations
#  8. get contours
#  9. get convex hull & defects to get features from the hand (eg: area of hull, no of finger, joints, angles, etc etc)
#  10. use all the features to come out with an algorithm to detect hand-cricket number played (0, 1, 2, ..., 6)
#  11. if error, or unable to detect, return -1
#  12. return the evaluated / predicted number to main.py (main.py will receive 2 numbers for 2 player hand images)
#  13. you may want to get predictions of more than one image frame and take the mode of the predictions
#  14. if and of the value obtained is -1, there is an error somewhere, throw an alert to the players
#  15. you can extract who has the problem out of player-1 / player-2
#  16. if error, go back to phase 2! else, continue...
#  ...
#  ********* phase 3: (get the game going) *********
#  1. calculate the runs and wickets of each player after every ball.
#  2. increment ball count and overs elapsed
#  3. display the current scores, wickets, overs
#  4. check for innings end (all out / over up)
#  5. if innings end, swap the batsman and the bowler
#  6. set target and display target if it's the second innings
#  7. watch for game over
#  8. after every ball, go back to "phase 2" and repeat the process till here, until game over
#  9. if game over, move to GAME OVER
#  ...
#  ===================================================== GAME OVER =====================================================
#  ********* phase 4: (end game) *********
#  1. display the winner
#  2. display all the game stats
#  3. replay? end game?
#  4. if replay, start over
#  5. if end game, quit
#
########################################################################################################################
########################################################################################################################

import cv2
import numpy as np
import processing
import bin
import hsv

# video capture
cap = cv2.VideoCapture(0)

# input video frame properties
_, frame = cap.read()
HEIGHT = frame.shape[0]
WIDTH = frame.shape[1]

# defining player areas and it's coordinates
gap = 20
p1_x1, p1_y1 = (gap, int(HEIGHT / 3))
p1_x2, p1_y2 = (int(WIDTH / 2 - gap), HEIGHT - gap)
p2_x1, p2_y1 = (int(WIDTH / 2 + gap), int(HEIGHT / 3))
p2_x2, p2_y2 = (int(WIDTH - gap), HEIGHT - gap)

# sample positions
sampling_permission = False
timer = 0
sample_sq_len = 10
p1_sample_images = []
p2_sample_images = []
sample_coordinates = [
    (100, 262),
    (138, 225),
    (164, 225),
    (197, 241),
    (104, 328),
    (143, 302),
    (168, 292),
    (193, 299),
    (217, 323),
    (108, 389),
    (154, 356),
    (207, 365),
    (129, 414),
    (178, 404)
]

# player variables
p1_bin_thresholds = None
p1_hsv_thresholds = None
p2_bin_thresholds = None
p2_hsv_thresholds = None

# game variables
game_state = 0  # 0: beginning, 1: game on, 2:game over


def click_event(event, x, y, flags, params):
    # left click: image pixel coordinate
    if event == cv2.EVENT_LBUTTONDOWN:
        cord = '(' + str(x) + ', ' + str(y) + ')'
        print('(X,Y) =', cord)


while cap.isOpened():

    # read frame
    ret, frame = cap.read()

    # flip image frame
    frame = cv2.flip(frame, 1)

    # cutout two sub images representing player areas
    player_1_img = frame[p1_y1:p1_y2, p1_x1:p1_x2, :].copy()
    player_2_img = frame[p2_y1:p2_y2, p2_x1:p2_x2, :].copy()

    # draw demarcation areas on the frame
    cv2.line(frame, (int(WIDTH / 2), 0), (int(WIDTH / 2), HEIGHT), (0, 255, 0), 2)
    cv2.rectangle(frame, (p1_x1, p1_y1), (p1_x2, p1_y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (p2_x1, p2_y1), (p2_x2, p2_y2), (0, 255, 0), 2)

    # BEGINNING:
    # init game and get the thresholds
    if game_state == 0:

        # use the line of code below to extract coordinates
        # cv2.setMouseCallback('frame', click_event)

        if sampling_permission:
            for coordinate in sample_coordinates:
                # cut the images and store it in a list
                p1_sample_images.append(frame[y:(y + sample_sq_len), x:(x + sample_sq_len), :].copy())
                p2_sample_images.append(
                    frame[y:(y + sample_sq_len), (x + int(WIDTH / 2)):(x + int(WIDTH / 2) + sample_sq_len), :].copy())

            # end of beginning
            game_state += 1

            # get the thresholds
            p1_bin_thresholds = processing.get_bin_thresh_list(p1_sample_images)
            p2_bin_thresholds = processing.get_bin_thresh_list(p2_sample_images)
            p1_hsv_thresholds = processing.get_hsv_thresh_list(p1_sample_images)
            p2_hsv_thresholds = processing.get_hsv_thresh_list(p2_sample_images)

        # draw small squares to show the sampling areas in the images
        for coordinate in sample_coordinates:
            # unpack the x and y
            x, y = coordinate

            # show the sampling squares
            cv2.rectangle(frame, (x, y), (x + sample_sq_len, y + sample_sq_len), (255, 0, 0), 1)
            cv2.rectangle(frame, (x + int(WIDTH / 2), y), (x + int(WIDTH / 2) + sample_sq_len, y + sample_sq_len),
                          (255, 0, 0), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        timer += 1.0
        print(timer)

        if timer == 300.0:
            timer = 0
            sampling_permission = True

        # do not proceed further until successful sampling is done and correct thresholds are obtained
        continue

    # get the threshold images
    p1_bin_images = processing.get_bin_images_from_thresh_list(player_1_img, p1_bin_thresholds)
    p1_hsv_images = processing.get_hsv_images_from_thresh_list(player_1_img, p1_hsv_thresholds)
    p2_bin_images = processing.get_bin_images_from_thresh_list(player_2_img, p2_bin_thresholds)
    p2_hsv_images = processing.get_hsv_images_from_thresh_list(player_2_img, p2_hsv_thresholds)

    p1_frame, p1_move = processing.get_player_move(player_1_img, p1_bin_thresholds, p1_hsv_thresholds)
    p2_frame, p2_move = processing.get_player_move(player_2_img, p2_bin_thresholds, p2_hsv_thresholds)

    print("P1: " + str(p1_move) + "-----" + "P2: " + str(p2_move))

    # display the frame
    cv2.imshow('frame', frame)
    cv2.imshow('p1_frame', p1_frame)
    cv2.imshow('p2_frame', p2_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()
