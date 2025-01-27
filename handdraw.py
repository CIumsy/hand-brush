import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize drawing parameters
drawing_points = deque(maxlen=512)
previous_strokes = []  # To store previous strokes with their respective colors
draw_color = (0, 0, 255)  # Default color: Red
brush_thickness = 5
is_drawing = False
straight_line_mode = False  # Flag for straight line mode
eraser_mode = False  # Flag for eraser mode
show_tutorial = False  # Flag for showing tutorial

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Real-time hand tracking
    max_num_hands=1,         # Track up to 1 hand
    min_detection_confidence=0.3,  # Adjust this value for better detection
    min_tracking_confidence=0.3
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set resolution to 1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

im_width, im_height = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a white canvas
canvas = np.ones((int(im_height), int(im_width), 3), dtype=np.uint8) * 255

# Store last drawn position for straight line mode
last_point = None

# Brush color options
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0),(255, 255, 0)]  # Red, Green, Blue, Black, Yellow
color_index = 0

# Stroke sizes (brush thickness)
brush_sizes = [5, 10, 15, 20]
brush_size_index = 0

# Define the tutorial text
tutorial_text = """
    Controls:
    - 'q' to Quit
    - 'c' to Clear Canvas
    - 'l' to Toggle Straight Line Mode
    - 'r' to Change Color to Red
    - 'g' to Change Color to Green
    - 'b' to Change Color to Blue
    - 'k' to Change Color to Black
    - 'y' to Change Color to Yellow
    - '+' to Increase Brush Size
    - '-' to Decrease Brush Size
    - 'e' to Toggle Eraser Mode
    - 't' to Show/Hide This Tutorial
"""

# MediaPipe Hand Tracking and Drawing Loop
while True:
    # Capture frame
    ret, image_np = cap.read()
    image_np = cv2.flip(image_np, 1)
    
    # Convert frame to RGB for MediaPipe
    # image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(image_np)

    # Process hand detection for drawing
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of index fingertip (landmark 8)
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Convert normalized coordinates to screen coordinates
            center_x = int(x * im_width)
            center_y = int(y * im_height)

            # Calculate hand aspect ratio to detect if it's a fist or open hand
            hand_ratio = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)

            # If hand is in drawing position (open hand)
            if hand_ratio > 0.1:  # This can be adjusted for better accuracy
                if straight_line_mode and last_point:
                    # Draw a straight line from the last point to the current point
                    cv2.line(canvas, last_point, (center_x, center_y), draw_color, brush_sizes[brush_size_index])
                    previous_strokes.append(((last_point, (center_x, center_y)), draw_color))
                elif not eraser_mode:
                    # Draw freehand
                    drawing_points.appendleft((center_x, center_y))
                    is_drawing = True
                last_point = (center_x, center_y)
            else:
                drawing_points.appendleft(None)
                last_point = None
                is_drawing = False

    # Draw previous strokes with their respective colors
    for stroke, color in previous_strokes:
        cv2.line(canvas, stroke[0], stroke[1], color, brush_sizes[brush_size_index])

    # Draw lines between points for freehand drawing
    for i in range(1, len(drawing_points)):
        if drawing_points[i - 1] is None or drawing_points[i] is None:
            continue
        if eraser_mode:
            # Draw white line for eraser mode
            cv2.line(canvas, drawing_points[i - 1], drawing_points[i], (255, 255, 255), brush_sizes[brush_size_index])
        else:
            # Draw with the current color for new strokes
            cv2.line(canvas, drawing_points[i - 1], drawing_points[i], draw_color, brush_sizes[brush_size_index])

    # Show the tutorial if the flag is set
    if show_tutorial:
        # Set the tutorial background as a semi-transparent overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (50, 50), (int(im_width) - 50, int(im_height) - 50), (255, 255, 255), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        
        # Put the tutorial text on the canvas
        font = cv2.FONT_HERSHEY_SIMPLEX
        y0, dy = 50, 35
        for i, line in enumerate(tutorial_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(canvas, line, (50, y), font, 0.8, (0, 0, 0), 2)

    # Show the result
    cv2.imshow('Hand Drawing', canvas)

    # Handle keyboard input for canvas control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        canvas.fill(255)  # Clear canvas
        drawing_points.clear()  # Clear drawing points
        previous_strokes.clear()  # Clear previous strokes
    elif key == ord('l'):
        straight_line_mode = not straight_line_mode
    elif key == ord('r'):  # Change to red
        draw_color = colors[0]
    elif key == ord('g'):  # Change to green
        draw_color = colors[1]
    elif key == ord('b'):  # Change to blue
        draw_color = colors[2]
    elif key == ord('k'):  # Change to black
        draw_color = colors[3]
    elif key == ord('y'):  # Change to yellow
        draw_color = colors[4]
    elif key == ord('+'):  # Increase brush size
        brush_size_index = (brush_size_index + 1) % len(brush_sizes)
    elif key == ord('-'):  # Decrease brush size
        brush_size_index = (brush_size_index - 1) % len(brush_sizes)
    # elif key == ord('e'):  # Toggle Eraser Mode
    #     eraser_mode = not eraser_mode
    #     if eraser_mode:
    #         draw_color = (255, 255, 255)  # Set color to white for eraser
    #         brush_sizes[brush_size_index] *= 4  # Increase brush size by 4 times
    #     else:
    #         brush_sizes[brush_size_index] //= 4  # Reset brush size to original
    elif key == ord('t'):  # Toggle Tutorial
        show_tutorial = not show_tutorial
        if not show_tutorial:
            canvas = np.ones((int(im_height), int(im_width), 3), dtype=np.uint8) * 255  # Reset canvas when tutorial is hidden

# Cleanup
cap.release()
cv2.destroyAllWindows()
