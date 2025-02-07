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
results = trained_model.predict(image)

# Define colors for different objects
colors = {
    "person": (0, 255, 0),  # Green
    "bicycle": (0, 0, 255),  # Blue
    "motorcycle": (255, 0, 0),  # Red
    "horse": (128, 0, 128),  # Purple
}

# Process results
for result in results:
    for box in result.boxes:
        if len(result.boxes) > 0:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Get the class name
            class_id = int(box.cls[0])
            class_name = trained_model.names[class_id]  # Map class index to name
            color = colors.get(class_name, (255, 255, 255))  # Default color is white

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            #cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Create a legend of colors
legend_height = len(colors) * 50
legend_width = image.shape[1]  # Make legend width match the image width
legend_image = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)

for i, (class_name, color) in enumerate(colors.items()):
    cv2.rectangle(legend_image, (0, i * 50), (legend_width, (i + 1) * 50), color, -1)
    cv2.putText(legend_image, class_name, (10, i * 50 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Stack the images vertically (processed image on top, legend below)
final_image = cv2.vconcat([image, legend_image])

# Save the result
cv2.imwrite("image_with_legend.jpg", final_image)
print("Processed image with legend saved as image_with_legend.jpg")