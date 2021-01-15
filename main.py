import numpy as np
import pygame
import cv2
from util.threshold_generator import get_thresholds
from util.predict import get_player_move


# user input variables
DEBUGGING_MODE = True           # make this flag True to enable debugging mode
CAMERA_NUMBER = 1               # default is 0
FLIP_CAMERA = True              # to flip camera horizontally


# get the thresholds from a opencv ui for processing
p1_bin_thresholds, p2_bin_thresholds, p1_hsv_thresholds, p2_hsv_thresholds = get_thresholds(CAMERA_NUMBER)


# initialize pygame
pygame.init()

# opencv video capture
cap = cv2.VideoCapture(CAMERA_NUMBER, cv2.CAP_DSHOW)

# get video frame properties
_, frame = cap.read()
HEIGHT = frame.shape[0]
WIDTH = frame.shape[1]

# defining player areas and it's coordinates
gap = 20
p1_x1, p1_y1 = (gap, int(HEIGHT / 3))
p1_x2, p1_y2 = (int(WIDTH / 2 - gap), HEIGHT - gap)
p2_x1, p2_y1 = (int(WIDTH / 2 + gap), int(HEIGHT / 3))
p2_x2, p2_y2 = (int(WIDTH - gap), HEIGHT - gap)


# global variables
running = cap.isOpened()
pause_state = 0
p1_move = -1
p2_move = -1

# Input key states (keyboard)
SPACE_BAR_PRESSED = 0
ENTER_KEY_PRESSED = 0
ESC_KEY_PRESSED = 0
Q_KEY_PRESSED = 0


# game objects
trigger_capture = False
innings = 1
p1_runs = 0
p2_runs = 0
p1_wickets = 0
p2_wickets = 0
over = 0
ball = 0


# create pygame display window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Cricket Umpire")
window_icon = pygame.image.load("resources/images/cricket-logo.png")
pygame.display.set_icon(window_icon)

# game sounds
pause_sound = None
game_over_sound = None


# main game loop begins
while running:

    # clear last frame
    window.fill((0, 0, 0))

    # register events
    for event in pygame.event.get():
        # Quit Event
        if event.type == pygame.QUIT:
            running = False
        # Keypress Down Event
        if event.type == pygame.KEYDOWN:
            # Space Bar down
            if event.key == pygame.K_SPACE:
                print("LOG: Space Bar Pressed Down")
                SPACE_BAR_PRESSED = 1
                trigger_capture = True
            # Enter Key down ("Carriage RETURN key" from old typewriter lingo)
            if event.key == pygame.K_RETURN:
                print("LOG: Enter Key Pressed Down")
                ENTER_KEY_PRESSED = 1
            # Esc Key down
            if event.key == pygame.K_ESCAPE:
                print("LOG: Escape Key Pressed Down")
                ESC_KEY_PRESSED = 1
            # Q Key down
            if event.key == pygame.K_q:
                print("LOG: Q Key Pressed Down")
                Q_KEY_PRESSED = 1
        # Keypress Up Event
        if event.type == pygame.KEYUP:
            # Space Bar up
            if event.key == pygame.K_SPACE:
                print("LOG: Space Bar Released")
                SPACE_BAR_PRESSED = 0
            # Enter Key up ("Carriage RETURN key" from old typewriter lingo)
            if event.key == pygame.K_RETURN:
                print("LOG: Enter Key Released")
                ENTER_KEY_PRESSED = 0
            # Esc Key up
            if event.key == pygame.K_ESCAPE:
                print("LOG: Escape Key Released")
                ESC_KEY_PRESSED = 0
            # Q Key down
            if event.key == pygame.K_q:
                print("LOG: Q Key Released")
                Q_KEY_PRESSED = 0

    # check for manual quit game
    if ESC_KEY_PRESSED or Q_KEY_PRESSED:
        running = False

    # read frame
    ret, frame = cap.read()

    # flip image frame
    if FLIP_CAMERA:
        frame = cv2.flip(frame, 1)

    # OPENCV Processing to get predictions#

    # cutout two sub images representing player areas
    player_1_img = frame[p1_y1:p1_y2, p1_x1:p1_x2, :].copy()
    player_2_img = frame[p2_y1:p2_y2, p2_x1:p2_x2, :].copy()

    # draw demarcation areas on the frame
    cv2.line(frame, (int(WIDTH / 2), 0), (int(WIDTH / 2), HEIGHT), (0, 255, 0), 2)
    cv2.rectangle(frame, (p1_x1, p1_y1), (p1_x2, p1_y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (p2_x1, p2_y1), (p2_x2, p2_y2), (0, 255, 0), 2)

    # get both the player moves
    if SPACE_BAR_PRESSED == 0 and trigger_capture:
        p1_move = get_player_move(player_1_img, p1_bin_thresholds, p1_hsv_thresholds, DEBUGGING_MODE, 1)
        p2_move = get_player_move(player_2_img, p2_bin_thresholds, p2_hsv_thresholds, DEBUGGING_MODE, 2)

        trigger_capture = False
        print("P1: " + str(p1_move) + "\t" + "P2: " + str(p2_move))

        # update game data
        if p1_move >= 0 and p2_move >= 0:           # valid move
            ball += 1
            if p1_move == p2_move:                  # out!
                if innings == 1:
                    p1_wickets += 1
                if innings == 2:
                    p2_wickets += 1
            else:                                   # scored runs
                if innings == 1:
                    p1_runs += p1_move
                if innings == 2:
                    p2_runs += p2_move
        else:                                       # invalid move
            print("invalid move! try again!")

    # game logic
    if ball == 6:
        over += 1
        ball = 0

    if innings == 1 and (over == 10 or p1_wickets == 10):
        innings = 2
        ball = 0
        over = 0

    if innings == 2 and (over == 10 or p2_wickets == 10 or p2_runs > p1_runs):
        # game over
        print("Game Over!!")
        if p1_runs > p2_runs:
            print("Player 1 won!!")
        if p1_runs < p2_runs:
            print("Player 2 won!!")
        if p1_runs == p2_runs:
            print("Match Draw!!")
        break

    if DEBUGGING_MODE:
        print("Innings  : " + str(innings))
        print("Overs    : " + str(over) + " / " + str(ball))
        print("Player 1 : " + str(p1_runs) + " / " + str(p1_wickets))
        print("Player 2 : " + str(p2_runs) + " / " + str(p2_wickets))

    # view opencv perspective if debugging mode is on
    if DEBUGGING_MODE:
        cv2.imshow("opencv", frame)

    # convert to pygame surface
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)

    # blit the frame
    window.blit(frame, (0, 0))

    # render the display
    pygame.display.update()


# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()
pygame.quit()
