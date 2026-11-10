
import os
import json
import csv
from paddleocr import PaddleOCR

# Input and output folders
INPUT_FOLDER = "outputs/cropped_books"
OUTPUT_FOLDER = "outputs/ocr_results_paddle"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize PaddleOCR
print("Initializing PaddleOCR...")

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

results_data = []

# Process each cropped book
for filename in sorted(os.listdir(INPUT_FOLDER)):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)

    print(f"\nProcessing: {filename}")

    try:
        results = ocr.predict(image_path)

        for result in results:
            data = result.json

            if callable(data):
                data = data()

            if isinstance(data, str):
                data = json.loads(data)

            payload = data.get("res", data)

            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])

            if not texts:
                print("No text detected.")
                results_data.append({
                    "image": filename,
                    "detected_text": "",
                    "confidence": ""
                })
                continue

            for text, score in zip(texts, scores):
                print(f"Text: {text} | Confidence: {score:.3f}")

                results_data.append({
                    "image": filename,
                    "detected_text": text,
                    "confidence": round(float(score), 3)
                })

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save CSV
csv_path = os.path.join(OUTPUT_FOLDER, "paddleocr_results.csv")

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["image", "detected_text", "confidence"]
    )
    writer.writeheader()
    writer.writerows(results_data)

print("\nOCR processing completed!")
print(f"Results saved to: {csv_path}")
