
import csv
import os
import re
import time
import requests

from difflib import SequenceMatcher
from collections import defaultdict

INPUT_CSV = "outputs/ocr_results_paddle_v2/paddleocr_v2_comparison.csv"
IMAGE_FOLDER = "outputs/cropped_books"

OUTPUT_FOLDER = "outputs/book_identification"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_FOLDER,
    "identified_books_v2.csv"
)

API_URL = "https://openlibrary.org/search.json"

session = requests.Session()
session.headers.update({
    "User-Agent": "ShelfSenseAI/1.0 (educational project)"
})


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def similarity(text1, text2):
    a = normalize(text1)
    b = normalize(text2)

    if not a or not b:
        return 0.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    overlap = len(tokens_a & tokens_b) / max(
        len(tokens_a), 1
    )

    return max(sequence_score, overlap)


def load_ocr_results():
    grouped = defaultdict(lambda: defaultdict(list))

    with open(INPUT_CSV, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image = row["image"]
            method = row["method"]
            text = row["detected_text"].strip()

            try:
                confidence = float(row["confidence"] or 0)
            except ValueError:
                confidence = 0

            if text:
                grouped[image][method].append({
                    "text": text,
                    "confidence": confidence
                })

    return grouped


def build_queries(methods):

    priority = [
        "counterclockwise",
        "original",
        "counterclockwise_contrast",
        "clockwise"
    ]

    queries = []
    seen = set()

    for method in priority:

        detections = methods.get(method, [])

        # Prefer meaningful, relatively confident text
        detections = sorted(
            detections,
            key=lambda item: item["confidence"],
            reverse=True
        )

        for item in detections:

            text = item["text"].strip()
            cleaned = normalize(text)

            # Ignore very short OCR fragments
            if len(cleaned) < 4:
                continue

            if len(cleaned.split()) == 1 and len(cleaned) < 5:
                continue

            if cleaned not in seen:
                queries.append(cleaned)
                seen.add(cleaned)

            # Limit requests per image
            if len(queries) >= 4:
                return queries

    return queries


def search_books(query):

    params = {
        "q": query,
        "limit": 10,
        "fields": (
            "key,title,author_name,first_publish_year,"
            "isbn,publisher,cover_i,edition_count"
        )
    }

    try:
        response = session.get(
            API_URL,
            params=params,
            timeout=25
        )

        response.raise_for_status()

        return response.json().get("docs", [])

    except requests.RequestException as error:
        print("API error:", error)
        return []


def get_book_information(book):

    title = book.get("title", "")
    authors = book.get("author_name", [])

    return {
        "title": title,
        "authors": ", ".join(authors),
        "year": book.get("first_publish_year", ""),
        "isbn": (book.get("isbn") or [""])[0],
        "publisher": (book.get("publisher") or [""])[0],
        "cover_url": (
            f"https://covers.openlibrary.org/b/id/{book['cover_i']}-L.jpg"
            if book.get("cover_i") else ""
        )
    }


def main():

    grouped = load_ocr_results()

    # Use crop folder to preserve all 15 images,
    # including images with no OCR text.
    image_files = sorted([
        filename
        for filename in os.listdir(IMAGE_FOLDER)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    output_rows = []

    print("\nShelfSense AI - Book Identification V2")
    print("=" * 55)

    for filename in image_files:

        print(f"\nProcessing: {filename}")

        methods = grouped.get(filename, {})
        queries = build_queries(methods)

        if not queries:
            print("No usable OCR text.")

            output_rows.append({
                "image": filename,
                "search_query": "",
                "matched_title": "",
                "authors": "",
                "publication_year": "",
                "isbn": "",
                "publisher": "",
                "cover_url": "",
                "match_score": 0,
                "status": "No OCR text"
            })

            continue

        candidates = {}

        for query in queries:

            print("Searching:", query)

            books = search_books(query)

            for book in books:

                key = book.get("key")

                if not key:
                    continue

                candidates[key] = book

            # Respect API request limits
            time.sleep(1)

        # Compare each result with all useful OCR phrases
        ranked = []

        for book in candidates.values():

            info = get_book_information(book)

            candidate_text = (
                info["title"] + " " + info["authors"]
            )

            best_score = max(
                similarity(query, candidate_text)
                for query in queries
            )

            ranked.append((best_score, info))

        ranked.sort(
            key=lambda item: item[0],
            reverse=True
        )

        if not ranked:
            print("No catalogue candidates found.")

            output_rows.append({
                "image": filename,
                "search_query": " | ".join(queries),
                "matched_title": "",
                "authors": "",
                "publication_year": "",
                "isbn": "",
                "publisher": "",
                "cover_url": "",
                "match_score": 0,
                "status": "No catalogue results"
            })

            continue

        for rank, (score, info) in enumerate(ranked[:5], start=1):

            # These are ranking aids, not verified probabilities.
            if score >= 0.65:
                status = "Strong candidate - verify"
            elif score >= 0.35:
                status = "Possible candidate - verify"
            else:
                status = "Weak candidate - verify"

            print(f"\nCandidate {rank}")
            print("Title:", info["title"])
            print("Author:", info["authors"])
            print("Similarity:", round(score, 3))
            print("Status:", status)

            output_rows.append({
                "image": filename,
                "search_query": " | ".join(queries),
                "matched_title": info["title"],
                "authors": info["authors"],
                "publication_year": info["year"],
                "isbn": info["isbn"],
                "publisher": info["publisher"],
                "cover_url": info["cover_url"],
                "match_score": round(score, 3),
                "status": status
            })

    fields = [
        "image",
        "search_query",
        "matched_title",
        "authors",
        "publication_year",
        "isbn",
        "publisher",
        "cover_url",
        "match_score",
        "status"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print("\n" + "=" * 55)
    print("Identification V2 completed!")
    print("Images processed:", len(image_files))
    print("Candidate records:", len(output_rows))
    print("Saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
