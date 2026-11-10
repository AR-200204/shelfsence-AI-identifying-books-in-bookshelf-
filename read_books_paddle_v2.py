
import os
import csv
import json
import cv2
from paddleocr import PaddleOCR

INPUT_FOLDER = "outputs/cropped_books"
OUTPUT_FOLDER = "outputs/ocr_results_paddle_v2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Initializing PaddleOCR...")

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)


def extract_results(results):
    detected = []

    for result in results:
        data = result.json

        if callable(data):
            data = data()

        if isinstance(data, str):
            data = json.loads(data)

        payload = data.get("res", data)

        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])

        for text, score in zip(texts, scores):
            detected.append({
                "text": str(text),
                "confidence": float(score)
            })

    return detected


def create_variants(image):
    variants = {}

    # Original
    variants["original"] = image

    # Rotate clockwise
    variants["clockwise"] = cv2.rotate(
        image, cv2.ROTATE_90_CLOCKWISE
    )

    # Rotate counterclockwise
    rotated = cv2.rotate(
        image, cv2.ROTATE_90_COUNTERCLOCKWISE
    )

    variants["counterclockwise"] = rotated

    # Contrast enhancement
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Convert grayscale back to BGR
    enhanced = cv2.cvtColor(
        enhanced, cv2.COLOR_GRAY2BGR
    )

    variants["counterclockwise_contrast"] = enhanced

    return variants


all_results = []

image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for filename in image_files:

    image_path = os.path.join(INPUT_FOLDER, filename)

    print(f"\nProcessing: {filename}")

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        continue

    variants = create_variants(image)

    for method, processed_image in variants.items():

        print(f"  Testing: {method}")

        temp_path = os.path.join(
            OUTPUT_FOLDER,
            f"temp_{method}_{filename}"
        )

        cv2.imwrite(temp_path, processed_image)

        try:
            results = ocr.predict(temp_path)
            detections = extract_results(results)

            if not detections:
                all_results.append({
                    "image": filename,
                    "method": method,
                    "detected_text": "",
                    "confidence": ""
                })

                print("    No text detected.")

            for item in detections:

                print(
                    f"    {item['text']} "
                    f"| Confidence: {item['confidence']:.3f}"
                )

                all_results.append({
                    "image": filename,
                    "method": method,
                    "detected_text": item["text"],
                    "confidence": round(item["confidence"], 3)
                })

        except Exception as error:
            print(f"    Error: {error}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


csv_path = os.path.join(
    OUTPUT_FOLDER,
    "paddleocr_v2_comparison.csv"
)

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "method",
            "detected_text",
            "confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


print("\nPaddleOCR V2 completed!")
print(f"Total images: {len(image_files)}")
print(f"Total OCR attempts: {len(image_files) * 4}")
print(f"Results saved to: {csv_path}")
