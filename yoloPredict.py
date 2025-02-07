import torch
from ultralytics import YOLO

# Load the trained model
trained_model = YOLO("trained_model.pt")

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make predictions
predictions = trained_model("image.jpg")

# Print predictions
print(predictions)