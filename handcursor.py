import cv2
import numpy as np
import pyautogui
from utils import detector_utils as detector_utils

# Load the hand detection model
detection_graph, sess = detector_utils.load_inference_graph()

# Detection parameters
score_thresh = 0.4
num_hands_detect = 1

# Initialize webcam
cap = cv2.VideoCapture(0)
im_width, im_height = (cap.get(3), cap.get(4))

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Cursor smoothing factor
smoothening = 5
prev_x, prev_y = 0, 0

while True:
    # Capture frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Detect hand
    boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

    # Process detection for cursor control
    if scores[0] > score_thresh:
        # Calculate hand position
        (left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width,
                                      boxes[0][0] * im_height, boxes[0][2] * im_height)

        # Calculate centroid of the hand
        center_x = int((left + right) / 2)
        center_y = int((top + bottom) / 2)

        # Map the hand position to the screen size
        screen_x = np.interp(center_x, [0, im_width], [0, screen_width])
        screen_y = np.interp(center_y, [0, im_height], [0, screen_height])

        # Smooth cursor movement
        cursor_x = prev_x + (screen_x - prev_x) / smoothening
        cursor_y = prev_y + (screen_y - prev_y) / smoothening

        # Move the cursor
        pyautogui.moveTo(cursor_x, cursor_y)  # No axis inversion for natural movement

        # Update previous cursor position
        prev_x, prev_y = cursor_x, cursor_y

    # Exit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cap.release()
