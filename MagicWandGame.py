import math
import cv2
import random
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

class MagicWandGame:
    def __init__(self, pathHope, pathCongrats):
        self.initializeData()
        self.PositiveEnergy = 0
        self.congratsImg = cv2.imread(pathCongrats, cv2.IMREAD_UNCHANGED)
        self.imgHope = cv2.imread(pathHope, cv2.IMREAD_UNCHANGED)
        if self.imgHope is not None:
            self.imgHope = cv2.resize(self.imgHope, (100, 100))
            self.hHope, self.wHope, _ = self.imgHope.shape
        else:
            raise ValueError("Cannot load hope image, please check the path")

    def randomLocation(self):
        self.hopePoint = random.randint(50, 550), random.randint(50, 550)

    def initializeData(self):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = 0, 0
        self.hopePoint = 0, 0
        self.randomLocation()

    def update(self, imgMain, currentHead, hands):
        cx, cy = currentHead
        px, py = self.previousHead
        self.points.append((cx, cy))
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy
        self.reduceLength()

        rx, ry = self.hopePoint
        if rx - self.wHope // 2 < cx < rx + self.wHope // 2 and ry - self.hHope // 2 < cy < ry + self.hHope // 2:
            self.PositiveEnergy += 1
            self.allowedLength += 50
            self.randomLocation()
            if self.PositiveEnergy == 30:
                # Set up the window to be resizable
                cv2.namedWindow("Mental Health Message", cv2.WINDOW_NORMAL)
                # Resize the window to be larger
                cv2.resizeWindow("Mental Health Message", 800, 600)
                cv2.imshow("Mental Health Message", self.congratsImg)

        if self.points:
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 255, 255), 10)  # Yellow color (BGR)
            cv2.circle(imgMain, self.points[-1], 10, (0, 255, 255), cv2.FILLED)  # Yellow color (BGR)

            # Add glitter effect
            for point in self.points:
                if random.random() > 0.7:  # Randomly add glitter effect
                    cv2.circle(imgMain, point, 5, (255, 255, 255), cv2.FILLED)  # White glitter

        cvzone.putTextRect(imgMain, f'Positive Energy: {self.PositiveEnergy}', [50, 80], scale=2, thickness=2, offset=10)
        imgMain = cvzone.overlayPNG(imgMain, self.imgHope, (rx - self.wHope // 2, ry - self.hHope // 2))

        # Check if a hand is making a fist to close the window
        if hands:
            lmList = hands[0]["lmList"]
            bbox = hands[0]["bbox"]
            x, y, w, h = bbox
            fist_detected = self.is_fist(lmList)
            if fist_detected:
                if cv2.getWindowProperty("Mental Health Message", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Mental Health Message")
        return imgMain

    def is_fist(self, lmList):
        # Simple heuristic: check if fingertips are close to palm (x-coordinates)
        if lmList:
            tips = [lmList[i][0] for i in [4, 8, 12, 16, 20]]  # Fingertip x-coordinates
            base = [lmList[i][0] for i in [0, 0, 0, 0, 0]]    # Palm base x-coordinate
            if all(abs(tips[i] - base[i]) < 40 for i in range(5)):
                return True
        return False

    def reduceLength(self):
        if self.currentLength > self.allowedLength:
            while self.currentLength > self.allowedLength and self.lengths:
                self.currentLength -= self.lengths.pop(0)
                self.points.pop(0)