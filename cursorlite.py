import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe Hands with GPU support
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lower for faster processing
)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Smoothing factor and thresholds
smoothening = 2
move_threshold = 5
prev_x, prev_y = 0, 0

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Increase resolution for better accuracy
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30 if supported by your webcam

# Frame skipping
# frame_skip = 2  # Process every 3rd frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Skip frames for performance
    # frame_count += 1
    # if frame_count % frame_skip != 0:
    #     continue

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of index fingertip (landmark 8)
            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Convert normalized coordinates to screen coordinates
            screen_x = np.interp(x, [0, 1], [0, screen_width])
            screen_y = np.interp(y, [0, 1], [0, screen_height])

            # Smooth cursor movement
            cursor_x = prev_x + (screen_x - prev_x) / smoothening
            cursor_y = prev_y + (screen_y - prev_y) / smoothening

            # Move the cursor only if it moves significantly
            if abs(cursor_x - prev_x) > move_threshold or abs(cursor_y - prev_y) > move_threshold:
                pyautogui.moveTo(cursor_x, cursor_y)
                prev_x, prev_y = cursor_x, cursor_y

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
