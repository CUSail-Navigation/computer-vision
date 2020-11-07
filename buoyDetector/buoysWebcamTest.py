from bouysorwater import BuoyDetector
import cv2
import numpy as np
import time
from picamera.array import PiRGBArray
from picamera import PiCamera


def find_distances(bd, scale):
    """Calculates distances from each contour and creates list of obstacle distances from camera.
    Return:
        A list where each one represents an obstacle distance.
    """
    focal_length = 3.60  # focal length of raspberry pi camera 2
    obstacle_size = 1016  # size of a buoy in mm
    mm_per_pixel = 1/600  # also based on camera, need to figure this out

    distances = []

    for contour in bd.filter_contours_output:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        distances.append(obstacle_size * focal_length /
                         (max(width, height) * mm_per_pixel))

    return distances


camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
vid = cv2.VideoCapture(0)

time.sleep(0.1)

print("Press q to quit.")

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    rawCapture.truncate()
    frame = frame.array
    # scale image
    max_dimension = max(frame.shape)
    scale = 700 / max_dimension
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    bd = BuoyDetector()
    bd.process(frame)

    contours = bd.filter_contours_output
    found = contours != None

    #print(find_distances(bd, scale))

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow('buoy detection', frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
