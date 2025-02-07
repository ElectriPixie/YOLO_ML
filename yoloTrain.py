import os
import torch
from ultralytics import YOLO

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Set the training dataset
#dataset = "coco8.yaml"
# Set the training and validation datasets
train_dataset = "coco8-train.yaml"
val_dataset = "coco8-val.yaml"

# Train the model
model.train(
    data=train_dataset,
    val_data=val_dataset,
    epochs=10,
    model="yolo11n.pt",
    lr0=0.01,
    batch=32,
    imgsz=640,
    val_batch=32
    val_imgsz=640
)

# Save the trained model
model.save("trained_model.pt")