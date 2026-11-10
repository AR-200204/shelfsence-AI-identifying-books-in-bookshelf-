
import easyocr
import cv2
from pathlib import Path
import csv

# 1. Initialize OCR (CPU mode to save GPU memory)
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

# 2. Folder containing cropped books
input_folder = Path("outputs/cropped_books")

# 3. Output folder
output_folder = Path("outputs/ocr_results")
output_folder.mkdir(parents=True, exist_ok=True)

# 4. Find all cropped images
images = list(input_folder.glob("*.jpg"))

if not images:
    raise FileNotFoundError("No cropped book images found!")

all_results = []

# 5. Read text from each book
for index, image_path in enumerate(images, start=1):

    print(f"\nProcessing {index}/{len(images)}: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image. Skipping.")
        continue

    # Extract text
    results = reader.readtext(image)

    extracted_text = []

    for bbox, text, confidence in results:
        extracted_text.append(text)

        print(f"Text: {text} | Confidence: {confidence:.2f}")

    # Save results
    all_results.append({
        "image": image_path.name,
        "extracted_text": " ".join(extracted_text)
    })

# 6. Export results to CSV
csv_path = output_folder / "book_text_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["image", "extracted_text"]
    )
    writer.writeheader()
    writer.writerows(all_results)

print("\nOCR completed!")
print(f"Images processed: {len(all_results)}")
print(f"Results saved to: {csv_path.resolve()}")
