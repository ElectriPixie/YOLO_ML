import torch
from ultralytics import YOLO
import cv2
import numpy as np
import random

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

# Create a dictionary to store unique class colors
class_colors = {}

# Extract detected classes and assign colors randomly
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = trained_model.names[class_id]  # Map class index to name
        
        if class_name not in class_colors:
            # Assign a random color (RGB format)
            class_colors[class_name] = tuple(random.randint(0, 255) for _ in range(3))

# Process results and draw bounding boxes
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        class_name = trained_model.names[class_id]
        color = class_colors[class_name]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        #cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Create a left-aligned legend dynamically
legend_width = image.shape[1]  # Match the width of the image
legend_height = len(class_colors) * 50  # Dynamic height based on classes detected
legend_bg_color = (220, 220, 220)  # Light gray background
legend_image = np.full((legend_height, legend_width, 3), legend_bg_color, dtype=np.uint8)

# Draw legend with random class colors
box_size = 30
padding = 10
text_x_offset = box_size + padding * 2  # Text starts after color box

for i, (class_name, color) in enumerate(class_colors.items()):
    y1, y2 = i * 50 + padding, (i + 1) * 50 - padding
    box_x1 = padding
    box_x2 = box_x1 + box_size
    
    # Draw color box
    cv2.rectangle(legend_image, (box_x1, y1), (box_x2, y2), color, -1)
    
    # Draw class name
    cv2.putText(legend_image, class_name, (text_x_offset, y1 + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Convert images to same data type
image = image.astype(np.uint8)
legend_image = legend_image.astype(np.uint8)

# Stack images vertically (image on top, legend below)
final_image = cv2.vconcat([image, legend_image])

# Save the final image
cv2.imwrite("image_with_legend.jpg", final_image)
print("Processed image with dynamic legend saved as image_with_legend.jpg")