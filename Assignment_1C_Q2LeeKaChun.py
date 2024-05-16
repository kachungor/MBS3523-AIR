import cv2
import numpy as np

default_lowH, default_lowS, default_lowV = 40, 50, 50
default_highH, default_highS, default_highV = 80, 255, 255

def update_HSV_values(lowH, lowS, lowV, highH, highS, highV):
    global new_lowH, new_lowS, new_lowV, new_highH, new_highS, new_highV
    new_lowH, new_lowS, new_lowV = lowH, lowS, lowV
    new_highH, new_highS, new_highV = highH, highS, highV

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Couldn't open webcam")
    exit()

cv2.namedWindow("Tune HSV")
cv2.createTrackbar("LowH", "Tune HSV", default_lowH, 40, lambda x: None)
cv2.createTrackbar("HighH", "Tune HSV", default_highH, 80, lambda x: None)
cv2.createTrackbar("LowS", "Tune HSV", default_lowS, 50, lambda x: None)
cv2.createTrackbar("HighS", "Tune HSV", default_highS, 255, lambda x: None)
cv2.createTrackbar("LowV", "Tune HSV", default_lowV, 50, lambda x: None)
cv2.createTrackbar("HighV", "Tune HSV", default_highV, 255, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame")
        break

    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowH = cv2.getTrackbarPos("LowH", "Tune HSV")
    highH = cv2.getTrackbarPos("HighH", "Tune HSV")
    lowS = cv2.getTrackbarPos("LowS", "Tune HSV")
    highS = cv2.getTrackbarPos("HighS", "Tune HSV")
    lowV = cv2.getTrackbarPos("LowV", "Tune HSV")
    highV = cv2.getTrackbarPos("HighV", "Tune HSV")

    update_HSV_values(lowH, lowS, lowV, highH, highS, highV)

    lowerBound = np.array([new_lowH, new_lowS, new_lowV])
    upperBound = np.array([new_highH, new_highS, new_highV])

    mask = cv2.inRange(hsvFrame, lowerBound, upperBound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Tune HSV", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
