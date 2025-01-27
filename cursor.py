import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os
import threading
import math
import signal
import sys

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Threaded class for frame capture
class VideoCaptureThread(threading.Thread):
    def __init__(self, cap, stop_event):
        threading.Thread.__init__(self)
        self.cap = cap
        self.frame = None
        self.ret = False
        self.stop_event = stop_event  # Event to signal when to stop

    def run(self):
        while not self.stop_event.is_set():
            self.ret, self.frame = self.cap.read()

    def stop(self):
        self.stop_event.set()  # Signal the thread to stop

# Initialize MediaPipe Hands with GPU support
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.3,  # Lower for faster processing
)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Smoothing factor and thresholds
smoothening = 1  # Reduced smoothening for faster response
move_threshold = 5
prev_x, prev_y = 0, 0

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Increased resolution for accuracy
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Ensure FPS is 30

# Event to control stopping of the capture thread
stop_event = threading.Event()

capture_thread = VideoCaptureThread(cap, stop_event)
capture_thread.start()

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Graceful exit handling for ctrl+c
def signal_handler(sig, frame):
    print("\nTerminating...")
    stop_event.set()  # Stop the capture thread
    capture_thread.stop()
    capture_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register signal handler for graceful termination
signal.signal(signal.SIGINT, signal_handler)

# Flag to allow q to terminate
terminate_on_q = False

while True:
    if capture_thread.ret:
        frame = capture_thread.frame

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get coordinates of index fingertip (landmark 8) and thumb tip (landmark 4)
                x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

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

                # Gesture detection for clicks
                # Calculate distance between thumb and index finger
                distance_thumb_index = calculate_distance(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                                                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])

                # Left click (thumb and index close together)
                if distance_thumb_index < 0.05:  # Threshold for a fist or close thumb-index
                    pyautogui.click()  # Perform left click
                    print("Left Click")

                # Right click (open hand detected)
                # Check if the distance between the thumb and pinky is large to detect open hand
                distance_thumb_pinky = calculate_distance(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                                                          hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP])

                if distance_thumb_pinky > 0.2:  # Threshold for open hand
                    pyautogui.rightClick()  # Perform right click
                    print("Right Click")

    # Exit on 'q' key or ctrl+c (once)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or terminate_on_q:
        stop_event.set()  # Stop the capture thread
        break
    elif key == ord('c') and cv2.waitKey(1) & 0xFF == 17:  # Detect ctrl+c press and toggle to q
        terminate_on_q = True

# Cleanup
capture_thread.stop()
capture_thread.join()
cap.release()
cv2.destroyAllWindows()
