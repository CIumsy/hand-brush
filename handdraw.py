from utils import detector_utils as detector_utils
import cv2
import numpy as np
from collections import deque

# Initialize drawing parameters
drawing_points = deque(maxlen=512)
draw_color = (0, 0, 255)  # Default color: Red
brush_thickness = 2
is_drawing = False
straight_line_mode = False  # Flag for straight line mode
eraser_mode = False  # Flag for eraser mode

# Load the hand detection model
detection_graph, sess = detector_utils.load_inference_graph()

# Detection parameters
score_thresh = 0.4
num_hands_detect = 1

# Initialize webcam
cap = cv2.VideoCapture(0)
im_width, im_height = (cap.get(3), cap.get(4))

# Create a white canvas
canvas = np.ones((int(im_height), int(im_width), 3), dtype=np.uint8) * 255

# Store last drawn position for straight line mode
last_point = None

# Create a window
cv2.namedWindow('Hand Drawing', cv2.WINDOW_NORMAL)

# Brush color options
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0)]  # Red, Green, Blue, Black
color_index = 0

while True:
    # Capture frame
    ret, image_np = cap.read()
    image_np = cv2.flip(image_np, 1)
    try:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")

    # Detect hand
    boxes, scores = detector_utils.detect_objects(
        image_np, detection_graph, sess)

    # Process hand detection for drawing
    if scores[0] > score_thresh:
        # Calculate hand position
        (left, right, top, bottom) = (boxes[0][1] * im_width, boxes[0][3] * im_width,
                                      boxes[0][0] * im_height, boxes[0][2] * im_height)

        # Calculate centroid of the hand
        center_x = int((left + right) / 2)
        center_y = int((top + bottom) / 2)

        # Calculate hand aspect ratio to detect if it's a fist or open hand
        hand_ratio = (bottom - top) / (right - left)

        # If hand is in drawing position (not a fist, hand_ratio > 1.2)
        if hand_ratio > 1.2:
            if straight_line_mode and last_point:
                # Draw a straight line from the last point to the current point
                cv2.line(canvas, last_point, (center_x, center_y), draw_color, brush_thickness)
            elif not eraser_mode:
                # Draw freehand
                drawing_points.appendleft((center_x, center_y))
                is_drawing = True
            last_point = (center_x, center_y)
        else:
            drawing_points.appendleft(None)
            last_point = None
            is_drawing = False

    # Draw lines between points for freehand drawing
    for i in range(1, len(drawing_points)):
        if drawing_points[i - 1] is None or drawing_points[i] is None:
            continue
        cv2.line(canvas, drawing_points[i - 1], drawing_points[i], draw_color, brush_thickness)

    # Add UI elements
    ui_text = [
        "Press 'c' to clear canvas",
        "Press 'q' to quit",
        "Press 's' to save canvas",
        "Press 'z' to undo last stroke",
        "Press 'e' to toggle eraser mode",
        "Press 'l' to toggle straight line mode",
        "Press 'b' to change brush color",
        "Press '+' to increase brush thickness",
        "Press '-' to decrease brush thickness"
    ]
    for i, text in enumerate(ui_text):
        cv2.putText(canvas, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Show the result
    cv2.imshow('Hand Drawing', canvas)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.ones((int(im_height), int(im_width), 3), dtype=np.uint8) * 255
        drawing_points.clear()
    elif key == ord('s'):
        cv2.imwrite('drawing.png', canvas)
    elif key == ord('z') and drawing_points:
        drawing_points.pop()  # Undo last point
    elif key == ord('e'):
        eraser_mode = not eraser_mode
        draw_color = (255, 255, 255) if eraser_mode else colors[color_index]
    elif key == ord('l'):
        straight_line_mode = not straight_line_mode
    elif key == ord('b'):
        color_index = (color_index + 1) % len(colors)
        draw_color = colors[color_index]
    elif key == ord('+'):
        brush_thickness += 1
    elif key == ord('-') and brush_thickness > 1:
        brush_thickness -= 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
