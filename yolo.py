import cv2
import numpy as np

# Load the YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO class labels
classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "bird", "cat", "dog", "horse"]

# Load the image
img = cv2.imread("image.jpg")

# Convert the image to a blob
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 2), True, False)

# Set the input blob for the network
net.setInput(blob)

# Run the YOLO object detection
outs = net.forward(net.getUnconnectedOutLayers())

# Loop through the detections
for i in range(len(outs)):
    for j in range(len(outs[i])):
        confidence = outs[i][j][2]
        if confidence > 0.5:
            # Get the class ID and confidence
            class_id = int(outs[i][j][0])
            class_name = classes[class_id]

            # Get the bounding box coordinates
            x, y, w, h = outs[i][j][3:7]
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Draw the bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the output
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
