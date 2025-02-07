import torch
from ultralytics import YOLO
import cv2
import numpy as np
import random
import argparse

# Create argparse
parser = argparse.ArgumentParser(description="YOLO model image processing")
parser.add_argument("--image_path", type=str, default="image.jpg", help="Path to the input image")
parser.add_argument("--output_image_path", type=str, default="image_with_legend.jpg", help="Path to save the processed image")
parser.add_argument("--legend_bg_color", type=str, default="220,220,220", help="Background color of the legend")
parser.add_argument("--legend_box_size", type=int, default=30, help="Size of the color box in the legend")
parser.add_argument("--legend_padding", type=int, default=10, help="Padding between legend elements")
parser.add_argument("--legend_text_size", type=float, default=0.8, help="Size of the legend text")

# Parse arguments
args = parser.parse_args()

# Load the trained model
trained_model = YOLO("trained_model.pt")

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model.to(device)  # Move model to device

# Load the image
image_path = args.image_path
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Make predictions
results = trained_model.predict(image)

# Function to convert HSV to RGB
def hsv_to_rgb(h, s, v):
    return tuple(int(i) for i in cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0])

# Create a dictionary to store unique class colors
class_colors = {}

# List of hues to ensure distinct colors (every 30 units apart to avoid similarity)
hue_step = 30
hue_values = [i * hue_step % 180 for i in range(12)]  # Limiting to 12 distinct hues, each 30 degrees apart

# Extract detected classes and assign colors randomly with distinct hues
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = trained_model.names[class_id]  # Map class index to name
        
        if class_name is None or class_name == '':
            class_name = 'Unknown Class'
        
        if class_name not in class_colors:
            # Assign a hue from the distinct hue values
            hue = hue_values[len(class_colors) % len(hue_values)]  # Cycle through the hue values
            
            # Random saturation and value to ensure the color is bright and vivid
            saturation = random.randint(150, 255)
            value = random.randint(150, 255)
            
            # Convert HSV to RGB and store the color
            color = hsv_to_rgb(hue, saturation, value)
            
            # Ensure the color is not already assigned (though the hue step should prevent this)
            while color in class_colors.values():
                hue = (hue + hue_step) % 180  # Change hue to get a new color
                color = hsv_to_rgb(hue, saturation, value)
            
            class_colors[class_name] = color

# Process results and draw bounding boxes
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        class_name = trained_model.names[class_id]
        color = class_colors[class_name]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # Optionally add the class label text
        #cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Create a left-aligned legend dynamically
legend_width = image.shape[1]  # Match the width of the image
legend_height = len(class_colors) * 50  # Dynamic height based on classes detected
legend_bg_color = tuple(map(int, args.legend_bg_color.split(",")))[::-1]  # Light gray background
legend_image = np.full((legend_height, legend_width, 3), legend_bg_color, dtype=np.uint8)

# Draw legend with random class colors
box_size = args.legend_box_size
padding = args.legend_padding
text_x_offset = box_size + padding * 2  # Text starts after color box

for i, (class_name, color) in enumerate(class_colors.items()):
    y1, y2 = i * 50 + padding, (i + 1) * 50 - padding
    box_x1 = padding
    box_x2 = box_x1 + box_size
    
    # Draw color box
    cv2.rectangle(legend_image, (box_x1, y1), (box_x2, y2), color, -1)
    
    # Draw class name
    cv2.putText(legend_image, class_name, (text_x_offset, y1 + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, args.legend_text_size, (0, 0, 0), 2)

# Convert images to same data type
image = image.astype(np.uint8)
legend_image = legend_image.astype(np.uint8)

# Stack images vertically (image on top, legend below)
final_image = cv2.vconcat([image, legend_image])

# Save the final image
cv2.imwrite(args.output_image_path, final_image)
print("Processed image with dynamic legend saved as", args.output_image_path)
