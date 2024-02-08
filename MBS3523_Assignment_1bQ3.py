import cv2
import numpy as np
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    width = cam.get(3)
    height = cam.get(4)

    if not ret:
        break

    frameResize = cv2.resize(frame, (int(width * 0.75), int(height * 0.75)))

    frameFlippedY = cv2.flip(frameResize, 1)
    frameFlippedX = cv2.flip(frameResize, 0)

    frameTop = np.hstack([frameResize, frameFlippedY])
    frameBottom = np.flip(frameTop, 0)
    frameMix = np.vstack([frameTop, frameBottom])

    cv2.imshow('Window', frameMix)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()