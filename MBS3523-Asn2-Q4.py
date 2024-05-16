import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

# Function to calculate the distance between two points
def calculate_distance(a, b):
    return ((b[0] - a[0])**2 + (b[1] - a[1])**2)**0.5

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize Mediapipe Hands model
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8,
                      min_tracking_confidence=0.6)

# Time variables for FPS calculation
t_old = 0
t_new = 0

# Variable to control screen brightness
brightness = 0

while True:
    success, img = cam.read()
    if not success:
        break

    # Recolor image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Make detection
    results = hands.process(imgRGB)

    # Recolor back to BGR
    imgRGB.flags.writeable = True
    img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

    # Extract landmarks and draw hand landmarks
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLandmarks,
                                  mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(125, 0, 125), thickness=2, circle_radius=4),
                                  mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

            # Get landmarks for thumb and index finger
            thumb = (handLandmarks.landmark[4].x, handLandmarks.landmark[4].y)
            index_finger = (handLandmarks.landmark[8].x, handLandmarks.landmark[8].y)

            # Calculate distance between thumb and index finger
            distance = calculate_distance(thumb, index_finger)

            # Determine hand state
            if distance > 0.1:  # Adjust this threshold as needed
                brightness = 0  # Increase brightness when hand opens
            else:
                brightness = -100  # Decrease brightness when hand closes

    # Adjust screen brightness
    img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)

    # Display the frame
    cv2.imshow('Hand Tracking Counter', img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
