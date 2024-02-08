import cv2
# print(cv2.__version__)
# cam  =cv2.VideoCapture(0)


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cam.read()

    width = int(cam.get(3))
    height = int(cam.get(4))
    print(width,height)

    frameResize = cv2.resize(frame, (int(width*0.75),int(height*0.75)))
    frameCanny = cv2.Canny(frameResize, 100, 100)
    frameHSV = cv2.cvtColor(frameResize, cv2.COLOR_BGR2HSV)
    frameBlur = cv2.GaussianBlur(frameResize, (15, 15), 0)

    cv2.imshow('Frame', frameResize)
    cv2.imshow('Window Blur', frameBlur)
    # cv2.imshow('Window Resize', frameResize)
    cv2.imshow('Window Canny', frameCanny)
    cv2.imshow('Window HSV', frameHSV)
    if cv2.waitKey(1) & 0xff == 27:
        break
cam.release()
cv2.destroyAllWindows()