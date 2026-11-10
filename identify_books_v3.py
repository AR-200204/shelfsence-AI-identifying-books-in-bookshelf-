
import csv
import re
import time
import requests

from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


# =====================================================
# SHELFSENSE AI - BOOK IDENTIFICATION V3
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

OCR_FILE = (
    BASE_DIR
    / "outputs"
    / "ocr_results_paddle_v2"
    / "paddleocr_v2_comparison.csv"
)

CROP_FOLDER = BASE_DIR / "outputs" / "cropped_books"

OUTPUT_DIR = BASE_DIR / "outputs" / "book_identification"

OUTPUT_FILE = OUTPUT_DIR / "identified_books_v3.csv"

API_URL = "https://openlibrary.org/search.json"

MAX_QUERIES = 8
MAX_RESULTS = 15

METHOD_PRIORITY = [
    "counterclockwise",
    "original",
    "counterclockwise_contrast",
    "clockwise"
]


# =====================================================
# 1. NORMALIZE TEXT
# =====================================================

def normalize(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9 ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# 2. CALCULATE SIMILARITY
# =====================================================

def similarity(text1, text2):

    a = normalize(text1)
    b = normalize(text2)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    words_a = set(a.split())
    words_b = set(b.split())

    # Ignore generic single-word matches.
    if len(words_a) == 1 and len(a) < 6:
        return 0.0

    if len(words_b) == 1 and len(b) < 6:
        return 0.0

    sequence_score = SequenceMatcher(
        None, a, b
    ).ratio()

    common_words = words_a.intersection(words_b)

    overlap_score = (
        len(common_words) / len(words_a)
        if words_a else 0
    )

    return max(
        sequence_score,
        overlap_score * 0.85
    )


# =====================================================
# 3. LOAD OCR CSV
# =====================================================

def load_ocr_results():

    grouped = defaultdict(list)

    if not OCR_FILE.exists():
        raise FileNotFoundError(
            f"OCR file not found: {OCR_FILE}"
        )

    with open(
        OCR_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        print("OCR CSV columns:", reader.fieldnames)

        required_columns = {
            "image",
            "method",
            "detected_text",
            "confidence"
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}"
            )

        for row in reader:

            image = row.get("image", "").strip()

            method = row.get("method", "").strip()

            detected_text = row.get(
                "detected_text", ""
            ).strip()

            try:
                confidence = float(
                    row.get("confidence", 0)
                )
            except (ValueError, TypeError):
                confidence = 0.0

            if not image or not detected_text:
                continue

            grouped[image].append({
                "method": method,
                "text": detected_text,
                "confidence": confidence
            })

    return grouped


# =====================================================
# 4. BUILD SEARCH QUERIES
# =====================================================

def build_queries(ocr_items):

    queries = []
    seen = set()

    for method in METHOD_PRIORITY:

        method_items = [
            item for item in ocr_items
            if item["method"] == method
        ]

        method_items.sort(
            key=lambda item: item["confidence"],
            reverse=True
        )

        for item in method_items:

            text = item["text"]

            cleaned = normalize(text)

            if len(cleaned) < 6:
                continue

            words = cleaned.split()

            # Avoid searching isolated short words.
            if len(words) == 1 and len(cleaned) < 9:
                continue

            if cleaned in seen:
                continue

            seen.add(cleaned)

            queries.append({
                "text": text,
                "method": method,
                "confidence": item["confidence"]
            })

            if len(queries) >= MAX_QUERIES:
                return queries

    return queries


# =====================================================
# 5. SEARCH OPEN LIBRARY
# =====================================================

def search_open_library(query):

    params = {
        "q": query,
        "limit": MAX_RESULTS,
        "fields": (
            "key,title,author_name,first_publish_year,"
            "isbn,publisher,cover_i,edition_count"
        )
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=20,
            headers={
                "User-Agent": "ShelfSenseAI/1.0"
            }
        )

        response.raise_for_status()

        return response.json().get("docs", [])

    except requests.RequestException as error:

        print("Open Library API error:", error)

        return []


# =====================================================
# 6. RANK CATALOGUE CANDIDATES
# =====================================================

def rank_candidates(queries):

    candidates = {}

    for query in queries:

        query_text = query["text"]

        print("Searching:", query_text)

        documents = search_open_library(query_text)

        for doc in documents:

            key = doc.get("key")

            if not key:
                continue

            title = doc.get("title", "")

            authors = doc.get("author_name", [])

            if isinstance(authors, str):
                authors = [authors]

            author_text = " ".join(authors)

            title_score = similarity(
                query_text,
                title
            )

            author_score = similarity(
                query_text,
                author_text
            )

            # Title evidence is more valuable than author evidence.
            score = max(
                title_score,
                author_score * 0.75
            )

            # Penalize generic one-word searches.
            if len(normalize(query_text).split()) == 1:
                score = min(score, 0.35)

            evidence = {
                "matched_query": query_text,
                "ocr_method": query["method"],
                "ocr_confidence": round(
                    query["confidence"], 3
                ),
                "title_similarity": round(
                    title_score, 3
                ),
                "author_similarity": round(
                    author_score, 3
                )
            }

            if (
                key not in candidates
                or score > candidates[key]["score"]
            ):

                candidates[key] = {
                    "key": key,
                    "title": title,
                    "authors": authors,
                    "year": doc.get(
                        "first_publish_year", ""
                    ),
                    "isbn": doc.get("isbn", []),
                    "publisher": doc.get("publisher", []),
                    "cover_id": doc.get("cover_i", ""),
                    "edition_count": doc.get(
                        "edition_count", ""
                    ),
                    "score": score,
                    "evidence": evidence
                }

        time.sleep(1)

    ranked = sorted(
        candidates.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked[:5]


# =====================================================
# 7. MATCH STATUS
# =====================================================

def match_status(score):

    if score >= 0.65:
        return "Promising candidate - verify"

    elif score >= 0.40:
        return "Possible candidate - verify"

    else:
        return "Low confidence - verify manually"


# =====================================================
# 8. PREPARE OUTPUT ROW
# =====================================================

def empty_result(image_name, queries, status):

    return {
        "image": image_name,
        "search_queries": " | ".join(
            item["text"] for item in queries
        ),
        "matched_title": "",
        "authors": "",
        "publication_year": "",
        "isbn": "",
        "publisher": "",
        "cover_url": "",
        "match_score": 0,
        "status": status,
        "evidence": ""
    }


# =====================================================
# 9. MAIN PROGRAM
# =====================================================

def main():

    print("\nShelfSense AI - Book Identification V3")

    print("=" * 55)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not CROP_FOLDER.exists():
        raise FileNotFoundError(
            f"Crop folder not found: {CROP_FOLDER}"
        )

    ocr_results = load_ocr_results()

    image_files = sorted(
        list(CROP_FOLDER.glob("*.jpg"))
        + list(CROP_FOLDER.glob("*.jpeg"))
        + list(CROP_FOLDER.glob("*.png"))
    )

    if not image_files:
        raise FileNotFoundError(
            "No cropped book images found."
        )

    output_rows = []

    for image_path in image_files:

        image_name = image_path.name

        print(f"\nProcessing: {image_name}")

        ocr_items = ocr_results.get(
            image_name, []
        )

        queries = build_queries(ocr_items)

        if not queries:

            print("No usable OCR text.")

            output_rows.append(
                empty_result(
                    image_name,
                    queries,
                    "No usable OCR text"
                )
            )

            continue

        ranked = rank_candidates(queries)

        if not ranked:

            print("No catalogue candidates found.")

            output_rows.append(
                empty_result(
                    image_name,
                    queries,
                    "No catalogue results"
                )
            )

            continue

        query_texts = " | ".join(
            item["text"] for item in queries
        )

        for index, candidate in enumerate(
            ranked, start=1
        ):

            score = candidate["score"]

            status = match_status(score)

            cover_id = candidate["cover_id"]

            cover_url = ""

            if cover_id:
                cover_url = (
                    "https://covers.openlibrary.org/b/id/"
                    f"{cover_id}-L.jpg"
                )

            isbn_list = candidate["isbn"]

            isbn_text = (
                str(isbn_list[0])
                if isbn_list else ""
            )

            publisher_list = candidate["publisher"]

            row = {
                "image": image_name,
                "search_queries": query_texts,
                "matched_title": candidate["title"],
                "authors": ", ".join(
                    candidate["authors"]
                ),
                "publication_year": candidate["year"],
                "isbn": isbn_text,
                "publisher": ", ".join(
                    publisher_list[:3]
                ),
                "cover_url": cover_url,
                "match_score": round(score, 3),
                "status": status,
                "evidence": str(
                    candidate["evidence"]
                )
            }

            output_rows.append(row)

            print(f"\nCandidate {index}")

            print("Title:", candidate["title"])

            print(
                "Authors:",
                ", ".join(candidate["authors"])
            )

            print("Score:", round(score, 3))

            print("Evidence:", candidate["evidence"])

            print("Status:", status)

    # =================================================
    # 10. SAVE CSV
    # =================================================

    fieldnames = [
        "image",
        "search_queries",
        "matched_title",
        "authors",
        "publication_year",
        "isbn",
        "publisher",
        "cover_url",
        "match_score",
        "status",
        "evidence"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(output_rows)

    print("\n" + "=" * 55)

    print("Identification V3 completed!")

    print("Images processed:", len(image_files))

    print("Candidate records:", len(output_rows))

    print("Saved:", OUTPUT_FILE)


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    main()
