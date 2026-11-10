
import easyocr
import cv2
import csv

from pathlib import Path

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

input_folder = Path("outputs/cropped_books")
output_folder = Path("outputs/ocr_results_v3")
output_folder.mkdir(parents=True, exist_ok=True)

images = sorted(input_folder.glob("*.jpg"))

if not images:
    raise FileNotFoundError("No cropped book images found!")


def create_variations(image):

    # Rotate the book spine counterclockwise
    rotated = cv2.rotate(
        image,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )

    gray = cv2.cvtColor(
        rotated,
        cv2.COLOR_BGR2GRAY
    )

    # Enlarge image
    enlarged = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(enlarged)

    # Adaptive threshold
    threshold = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # Sharpening
    blurred = cv2.GaussianBlur(
        contrast,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        contrast,
        1.5,
        blurred,
        -0.5,
        0
    )

    return {
        "rotated": enlarged,
        "contrast": contrast,
        "threshold": threshold,
        "sharpened": sharpened
    }


all_results = []

for index, image_path in enumerate(images, start=1):

    print(f"\nProcessing {index}/{len(images)}: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    variations = create_variations(image)

    for method, processed_image in variations.items():

        results = reader.readtext(processed_image)

        detected_text = []
        confidences = []

        for bbox, text, confidence in results:

            text = text.strip()

            if text:
                detected_text.append(text)
                confidences.append(confidence)

        combined_text = " ".join(detected_text)

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0
        )

        print(
            f"{method}: {combined_text} "
            f"(confidence: {average_confidence:.2f})"
        )

        all_results.append({
            "image": image_path.name,
            "method": method,
            "extracted_text": combined_text,
            "confidence": round(average_confidence, 3)
        })


# Save all OCR attempts
csv_path = output_folder / "ocr_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "method",
            "extracted_text",
            "confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


print("\nOCR V3 completed!")
print(f"Total OCR attempts: {len(all_results)}")
print(f"Results saved to: {csv_path.resolve()}")
