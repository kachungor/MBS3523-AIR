import cv2
import numpy as np

# Confidence threshold
confThreshold = 0.8

# Load COCO classes
classesFile = 'coco80.names'
classes = []
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()

# Load YOLO model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Fruit prices dictionary
fruit_prices = {'banana': 2.50, 'apple': 1.20, 'orange': 1.80}

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize fruit counts and flags to track if a fruit is detected in the current frame
fruit_counts = {'banana': 0, 'apple': 0, 'orange': 0}
detected_fruits = {'banana': False, 'apple': False, 'orange': False}

while True:
    success, img = cam.read()
    if not success:
        print("Failed to read frame")
        break

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    bboxes = []  # Array to store bounding boxes of all detected fruits
    confidences = []  # Array to store all confidence values of matching detected fruits
    class_ids = []  # Array to store all class IDs of matching detected fruits

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]  # Ignore first 5 values
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(bboxes), 3))

    # Reset detected fruits flags
    detected_fruits = {'banana': False, 'apple': False, 'orange': False}

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = bboxes[i]
            label = str(classes[class_ids[i]])
            confidence = round(confidences[i], 2)
            color = colors[i]

            # Check if the fruit is detected in this frame
            if label in detected_fruits and not detected_fruits[label]:
                # Increase fruit count
                fruit_counts[label] += 1
                # Update flag to indicate detected fruit
                detected_fruits[label] = True

                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # Draw text: fruit name and confidence
                text = f"{label}: {confidence}"
                cv2.putText(img, text, (x, y - 5), font, 1, color, 1)

    # Draw total fruit count and price
    total_fruits = sum(fruit_counts.values())
    total_price = sum(fruit_prices[fruit] * count for fruit, count in fruit_counts.items())
    total_text = f"Total fruits: {total_fruits}, Total price: ${total_price:.2f}"
    cv2.putText(img, total_text, (20, 30), font, 1, (0, 255, 0), 2)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
