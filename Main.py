import cv2
from cvzone.HandTrackingModule import HandDetector
from MagicWandGame import MagicWandGame

# Constants
FRAME_WIDTH = 1280
FRAME_HEIGHT = 1024

def initialize_camera(frame_width, frame_height):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None
    cap.set(3, frame_width)
    cap.set(4, frame_height)
    return cap

def main():
    cap = initialize_camera(FRAME_WIDTH, FRAME_HEIGHT)
    if cap is None:
        return

    detector = HandDetector(detectionCon=0.8, maxHands=1)
    hope_image_path = "C:/Users/85253/Desktop/Asn2_yolo/pngtree-goldan-3d-star-emoji-icon-png-image_10459560.png"  # Update with the actual path to hope image
    congrats_image_path = "C:/Users/85253/Desktop/Asn2_yolo/images.png"  # Update with the actual path to congrats image
    game = MagicWandGame(hope_image_path, congrats_image_path)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, draw=False, flipType=False)

        if hands:
            lmList = hands[0]["lmList"]
            pointIndex = lmList[8][0:2]
            img = game.update(img, pointIndex, hands)

        cv2.imshow("img", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()