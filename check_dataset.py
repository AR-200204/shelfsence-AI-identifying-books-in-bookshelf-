
from pathlib import Path

DATASET = Path("dataset/external/Open Shelves.v9i.yolov8")

splits = ["train", "valid", "test"]

for split in splits:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = {
        p.stem for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    }

    labels = {p.stem for p in label_dir.glob("*.txt")}

    missing_labels = images - labels
    missing_images = labels - images

    print(f"\n--- {split.upper()} ---")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(missing_images))

    if missing_labels:
        print("Examples:", list(missing_labels)[:5])
