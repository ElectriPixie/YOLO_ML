import torch
from ultralytics import YOLO
import argparse

# Create argparse
parser = argparse.ArgumentParser(description="YOLO model prediction")
parser.add_argument("--trained_model", type=str, default="trained_model.pt", help="Path to the trained model")
parser.add_argument("--image_path", type=str, default="image.jpg", help="Path to the image to make predictions on")

# Parse arguments
args = parser.parse_args()

# Load the trained model
trained_model = YOLO(args.trained_model)

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make predictions
predictions = trained_model(args.image_path)

# Print predictions
print(predictions)