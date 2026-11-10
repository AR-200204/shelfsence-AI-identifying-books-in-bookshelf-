
from ultralytics import YOLO
import cv2
from pathlib import Path

# 1. Load your trained YOLO model
model = YOLO(
    r"C:\Users\User\Desktop\runs\detect\runs\shelfsense\book_detection_v1\weights\best.pt"
)

# 2. Input bookshelf image
image_path = Path(
    r"dataset\external\Open Shelves.v9i.yolov8\test\images\20240221_103320_jpg.rf.ea0bf77b689c671e532e16acb774acf2.jpg"
)

# 3. Create output folder
output_folder = Path("outputs/cropped_books")
output_folder.mkdir(parents=True, exist_ok=True)

# 4. Read image
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

height, width = image.shape[:2]

# 5. Detect books
results = model.predict(
    source=str(image_path),
    conf=0.25,
    device=0,
    verbose=False
)

# 6. Crop each detected book
boxes = results[0].boxes
saved_count = 0

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    # Keep coordinates inside the image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # Extract book crop
    cropped_book = image[y1:y2, x1:x2]

    if cropped_book.size == 0:
        continue

    # Save crop
    output_path = output_folder / f"book_{i + 1:02d}.jpg"
    cv2.imwrite(str(output_path), cropped_book)

    saved_count += 1
    print(f"Saved: {output_path}")

print("\nBook cropping completed!")
print(f"Total detected books: {len(boxes)}")
print(f"Total crops saved: {saved_count}")
print(f"Output folder: {output_folder.resolve()}")
