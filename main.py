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

import numpy as np
import pygame
import cv2
import processing
import bin
import hsv

# opencv video capture
cap = cv2.VideoCapture(0)

# input video frame properties
_, frame = cap.read()
HEIGHT = frame.shape[0]
WIDTH = frame.shape[1]

# initialize pygame
pygame.init()

# global variables
running = True

# Input key states (keyboard)
Q_KEY_PRESSED = False
ESC_KEY_PRESSED = False

# create display window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Cricket")

# main game loop begins
while running:

    # clear last frame
    window.fill((0, 0, 0))

    # register events
    for event in pygame.event.get():
        # Quit Event
        if event.type == pygame.QUIT:
            running = False
            cap.release()
            cv2.destroyAllWindows()

    # read frame
    ret, frame = cap.read()

    # flip image frame
    frame = cv2.flip(frame, 1)

    bg_img = pygame.surfarray.make_surface(frame)

    # render the background
    window.blit(bg_img, (0, 0))

    # render the display
    pygame.display.update()

pygame.quit()
