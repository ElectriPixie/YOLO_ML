import os
import torch
from ultralytics import YOLO
import argparse

# Create argparse
parser = argparse.ArgumentParser(description="YOLO model training")
parser.add_argument("--trained_model", type=str, default="trained_model.pt", help="Path to save the trained model")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
parser.add_argument("--batch", type=int, default=32, help="Batch size for training")
parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
parser.add_argument("--dataset", type=str, default="coco8.yaml", help="Path to the training dataset")

# Parse arguments
args = parser.parse_args()

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the YOLO model
model = YOLO(args.trained_model)

# Train the model
model.train(
    data=args.dataset,
    epochs=args.epochs,
    model=args.trained_model,
    lr0=args.lr0,
    batch=args.batch,
    imgsz=args.imgsz,
)
model.val()

# Save the trained model
model.save(args.trained_model)