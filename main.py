import numpy as np
import pygame
import cv2

# user input variables
DEBUGGING_MODE = True           # make this flag True to enable debugging mode
CAMERA_NUMBER = 1               # default is 0
FLIP_CAMERA = True              # to flip camera horizontally


# initialize pygame
pygame.init()

# opencv video capture
cap = cv2.VideoCapture(CAMERA_NUMBER, cv2.CAP_DSHOW)


# get video frame properties
_, frame = cap.read()
HEIGHT = frame.shape[0]
WIDTH = frame.shape[1]


# global variables
running = cap.isOpened()
pause_state = 0


# Input key states (keyboard)
SPACE_BAR_PRESSED = 0
ENTER_KEY_PRESSED = 0
ESC_KEY_PRESSED = 0
Q_KEY_PRESSED = 0


# game objects


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



    ##############################

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
