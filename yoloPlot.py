import torch
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
trained_model = YOLO("trained_model.pt")

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model.to(device)  # Move model to device

# Load the image
image_path = "image.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Make predictions
results = trained_model(image_path)

# Process results
for result in results:
    for box in result.boxes:  
        x, y, w, h = box.xywh.tolist()[0]  # Ensure correct unpacking

        # Convert center (x, y) to top-left (x1, y1)
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the result
cv2.imwrite("image_plot.jpg", image)
print("Processed image saved as image_plot.jpg")