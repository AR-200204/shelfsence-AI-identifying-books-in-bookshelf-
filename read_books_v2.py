
import easyocr
import cv2
import csv

from pathlib import Path

# 1. Initialize EasyOCR
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

# 2. Input and output folders
input_folder = Path("outputs/cropped_books")
output_folder = Path("outputs/ocr_results_v2")

output_folder.mkdir(parents=True, exist_ok=True)

images = list(input_folder.glob("*.jpg"))

if not images:
    raise FileNotFoundError("No cropped images found!")

all_results = []


# 3. Image preprocessing function
def preprocess_image(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Enlarge the image
    enlarged = cv2.resize(
        enhanced,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Sharpen the image
    blurred = cv2.GaussianBlur(
        enlarged,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        enlarged,
        1.5,
        blurred,
        -0.5,
        0
    )

    return sharpened


# 4. Process every book
for index, image_path in enumerate(images, start=1):

    print(f"\nProcessing {index}/{len(images)}: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    # Create different orientations
    orientations = {
        "original": image,
        "clockwise": cv2.rotate(
            image, cv2.ROTATE_90_CLOCKWISE
        ),
        "counterclockwise": cv2.rotate(
            image, cv2.ROTATE_90_COUNTERCLOCKWISE
        )
    }

    candidates = []

    # 5. Try OCR on each orientation
    for orientation_name, rotated_image in orientations.items():

        processed = preprocess_image(rotated_image)

        results = reader.readtext(processed)

        detected_text = []
        confidences = []

        for bbox, text, confidence in results:

            text = text.strip()

            if text and confidence >= 0.15:
                detected_text.append(text)
                confidences.append(confidence)

        combined_text = " ".join(detected_text)

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0
        )

        candidates.append({
            "orientation": orientation_name,
            "text": combined_text,
            "confidence": average_confidence
        })

        print(
            f"{orientation_name}: "
            f"{combined_text} "
            f"(confidence: {average_confidence:.2f})"
        )

    # 6. Select the candidate with the highest average confidence
    best_result = max(
        candidates,
        key=lambda item: item["confidence"]
    )

    all_results.append({
        "image": image_path.name,
        "best_orientation": best_result["orientation"],
        "extracted_text": best_result["text"],
        "confidence": round(best_result["confidence"], 3)
    })

    print(f"BEST RESULT: {best_result['text']}")


# 7. Save results as CSV
csv_path = output_folder / "book_text_results_v2.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "best_orientation",
            "extracted_text",
            "confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


print("\nOCR V2 completed!")
print(f"Books processed: {len(all_results)}")
print(f"Results saved: {csv_path.resolve()}")
