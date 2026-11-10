
from ultralytics import YOLO

# Load the lightweight YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train the book detection model
results = model.train(
    data="dataset/external/Open Shelves.v9i.yolov8/data.yaml",
    epochs=30,
    imgsz=416,
    batch=1,
    device=0,
    workers=0,
    cache=False,
    project="runs/shelfsense",
    name="book_detection_v1",
    patience=10
)

print("Training completed!")
