from bouysorwater import BuoyDetector
import cv2
import numpy as np
import time


def find_distances(bd, scale):
    """Calculates distances from each contour and creates list of obstacle distances from camera.
    Return:
        A list where each one represents an obstacle distance.
    """
    focal_length = 40  # for webcam? Need to change for sailboat's camera when we find that
    obstacle_size = 1016  # size of a buoy in mm
    mm_per_pixel = 1/600  # also based on camera, need to figure this out

    distances = []

    for contour in bd.filter_contours_output:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        distances.append(obstacle_size * focal_length /
                         (max(width, height) * mm_per_pixel))

    return distances


vid = cv2.VideoCapture(0)

print("Press q to quit.")

while (True):
    _, frame = vid.read()  # get a frame from the webcam

    # scale image
    max_dimension = max(frame.shape)
    scale = 700 / max_dimension
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    bd = BuoyDetector()
    bd.process(frame)

    contours = bd.filter_contours_output
    found = contours != None

    print(find_distances(bd, scale))

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow('buoy detection', frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
