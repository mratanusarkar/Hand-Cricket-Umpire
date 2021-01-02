from .threshold import *


def get_thresholds(camera):

    # video capture
    cap = cv2.VideoCapture(camera)

    # input video frame properties
    _, frame = cap.read()
    height = frame.shape[0]
    width = frame.shape[1]

    # defining player areas and it's coordinates
    gap = 20
    p1_x1, p1_y1 = (gap, int(height / 3))
    p1_x2, p1_y2 = (int(width / 2 - gap), height - gap)
    p2_x1, p2_y1 = (int(width / 2 + gap), int(height / 3))
    p2_x2, p2_y2 = (int(width - gap), height - gap)

    # sample positions
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

    while cap.isOpened():
        # read frame
        ret, frame = cap.read()

        # flip image frame
        frame = cv2.flip(frame, 1)

        # # cutout two sub images representing player areas
        # player_1_img = frame[p1_y1:p1_y2, p1_x1:p1_x2, :].copy()
        # player_2_img = frame[p2_y1:p2_y2, p2_x1:p2_x2, :].copy()

        # draw demarcation areas on the frame
        cv2.line(frame, (int(width / 2), 0), (int(width / 2), height), (0, 255, 0), 2)
        cv2.rectangle(frame, (p1_x1, p1_y1), (p1_x2, p1_y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (p2_x1, p2_y1), (p2_x2, p2_y2), (0, 255, 0), 2)

        # draw small squares to show the sampling areas in the images
        for coordinate in sample_coordinates:
            # unpack the x and y
            x, y = coordinate

            # show the sampling squares
            cv2.rectangle(frame, (x, y), (x + sample_sq_len, y + sample_sq_len), (255, 0, 0), 1)
            cv2.rectangle(frame, (x + int(width / 2), y), (x + int(width / 2) + sample_sq_len, y + sample_sq_len),
                          (255, 0, 0), 1)

        cv2.imshow('Threshold Sampler', frame)

        # if space bar is pressed, generate the threshold
        if cv2.waitKey(25) & 0xFF == 32:

            for coordinate in sample_coordinates:
                # unpack the x and y
                x, y = coordinate

                # cut the images and store it in a list
                p1_sample_images.append(
                    frame[y:(y + sample_sq_len), x:(x + sample_sq_len), :].copy()
                )
                p2_sample_images.append(
                    frame[y:(y + sample_sq_len), (x + int(width / 2)):(x + int(width / 2) + sample_sq_len), :].copy()
                )

            # get the thresholds
            p1_bin_thresholds = get_bin_thresh_list(p1_sample_images)
            p2_bin_thresholds = get_bin_thresh_list(p2_sample_images)
            p1_hsv_thresholds = get_hsv_thresh_list(p1_sample_images)
            p2_hsv_thresholds = get_hsv_thresh_list(p2_sample_images)

            break

    # when everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # return the thresholds
    return p1_bin_thresholds, p2_bin_thresholds, p1_hsv_thresholds, p2_hsv_thresholds
