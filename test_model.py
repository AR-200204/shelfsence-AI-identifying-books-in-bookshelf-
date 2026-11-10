
from ultralytics import YOLO
from pathlib import Path

# Load your trained model
model = YOLO(
    r"C:\Users\User\Desktop\runs\detect\runs\shelfsense\book_detection_v1\weights\best.pt"
)

# Select an image from your test dataset
image_path = (
    r"dataset\external\Open Shelves.v9i.yolov8"
    r"\test\images"
)

# Get the first test image
images = list(Path(image_path).glob("*.jpg"))

if not images:
    raise FileNotFoundError("No JPG test images found!")

# Run book detection
results = model.predict(
    source=str(images[0]),
    conf=0.25,
    save=True,
    device=0
)

print("Test image:", images[0].name)
print("Detected books:", len(results[0].boxes))
print("Detection results saved.")
