
import csv
import re
import time
import requests

from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from urllib.parse import quote


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OCR_FILE = (
    BASE_DIR
    / "outputs"
    / "ocr_results_paddle_v2"
    / "paddleocr_v2_comparison.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "book_identification"
OUTPUT_FILE = OUTPUT_DIR / "identified_books_v4.csv"

OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

MAX_QUERIES_PER_IMAGE = 6
MAX_RESULTS_PER_QUERY = 10
MAX_CANDIDATES_PER_IMAGE = 5

REQUEST_DELAY = 0.3
TIMEOUT = 20

HEADERS = {
    "User-Agent": "ShelfSenseAI/1.0 (Book identification research)"
}


# ============================================================
# TEXT PROCESSING
# ============================================================

def normalize(text):
    """Normalize text for comparison."""

    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def similarity(text1, text2):
    """Calculate similarity between two text strings."""

    a = normalize(text1)
    b = normalize(text2)

    if not a or not b:
        return 0.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    words_a = set(a.split())
    words_b = set(b.split())

    if words_a and words_b:
        overlap = len(words_a & words_b) / len(words_a | words_b)
    else:
        overlap = 0.0

    return max(sequence_score, overlap)


def is_useful_text(text):
    """Reject OCR output that is unlikely to be useful."""

    cleaned = normalize(text)

    if len(cleaned) < 4:
        return False

    if not re.search(r"[a-z]", cleaned):
        return False

    return True


# ============================================================
# LOAD OCR RESULTS
# ============================================================

def load_ocr_results():

    if not OCR_FILE.exists():
        raise FileNotFoundError(
            f"OCR file not found: {OCR_FILE}"
        )

    grouped = defaultdict(list)

    with open(
        OCR_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "image",
            "method",
            "detected_text",
            "confidence"
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "OCR CSV must contain these columns: "
                "image, method, detected_text, confidence"
            )

        for row in reader:

            image = row["image"].strip()
            method = row["method"].strip()
            detected_text = row["detected_text"].strip()

            if not image or not is_useful_text(detected_text):
                continue

            try:
                confidence = float(row["confidence"])
            except (ValueError, TypeError):
                confidence = 0.0

            grouped[image].append({
                "method": method,
                "text": detected_text,
                "confidence": confidence
            })

    return grouped


# ============================================================
# BUILD SEARCH QUERIES
# ============================================================

def build_queries(ocr_records):

    method_priority = {
        "counterclockwise": 1,
        "original": 2,
        "counterclockwise_contrast": 3,
        "clockwise": 4
    }

    queries = []
    seen = set()

    sorted_records = sorted(
        ocr_records,
        key=lambda x: (
            method_priority.get(x["method"], 99),
            -x["confidence"]
        )
    )

    # First: individual OCR text detections
    for record in sorted_records:

        text = record["text"].strip()
        key = normalize(text)

        if not is_useful_text(text) or key in seen:
            continue

        queries.append({
            "text": text,
            "method": record["method"],
            "confidence": record["confidence"],
            "query_type": "individual"
        })

        seen.add(key)

    # Second: combine text detections from each OCR method
    method_groups = defaultdict(list)

    for record in sorted_records:
        method_groups[record["method"]].append(record)

    for method in sorted(
        method_groups,
        key=lambda m: method_priority.get(m, 99)
    ):

        records = sorted(
            method_groups[method],
            key=lambda x: -x["confidence"]
        )

        combined_text = " ".join(
            record["text"] for record in records
        )

        key = normalize(combined_text)

        if (
            is_useful_text(combined_text)
            and key not in seen
        ):
            queries.append({
                "text": combined_text,
                "method": method,
                "confidence": sum(
                    r["confidence"] for r in records
                ) / len(records),
                "query_type": "combined"
            })

            seen.add(key)

    return queries[:MAX_QUERIES_PER_IMAGE]


# ============================================================
# OPEN LIBRARY SEARCH
# ============================================================

def search_open_library(query):

    try:

        response = requests.get(
            OPEN_LIBRARY_URL,
            params={
                "q": query,
                "limit": MAX_RESULTS_PER_QUERY,
                "fields": (
                    "key,title,author_name,"
                    "first_publish_year,isbn,publisher,"
                    "edition_count"
                )
            },
            headers=HEADERS,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        candidates = []

        for book in data.get("docs", []):

            title = book.get("title", "")

            authors = book.get("author_name", [])

            candidates.append({
                "title": title,
                "authors": ", ".join(authors),
                "year": book.get("first_publish_year", ""),
                "isbn": book.get("isbn", [""])[0]
                if book.get("isbn") else "",
                "publisher": book.get("publisher", [""])[0]
                if book.get("publisher") else "",
                "source": "Open Library",
                "source_id": book.get("key", "")
            })

        return candidates

    except requests.RequestException as error:

        print(f"Open Library error: {error}")
        return []


# ============================================================
# GOOGLE BOOKS SEARCH
# ============================================================

def search_google_books(query):

    try:

        response = requests.get(
            GOOGLE_BOOKS_URL,
            params={
                "q": query,
                "maxResults": MAX_RESULTS_PER_QUERY,
                "printType": "books"
            },
            headers=HEADERS,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        candidates = []

        for item in data.get("items", []):

            volume = item.get("volumeInfo", {})

            title = volume.get("title", "")

            authors = volume.get("authors", [])

            industry_ids = volume.get(
                "industryIdentifiers", []
            )

            isbn = ""

            for identifier in industry_ids:

                if identifier.get("type") == "ISBN_13":
                    isbn = identifier.get("identifier", "")
                    break

            if not isbn:
                for identifier in industry_ids:
                    if identifier.get("type") == "ISBN_10":
                        isbn = identifier.get("identifier", "")
                        break

            candidates.append({
                "title": title,
                "authors": ", ".join(authors),
                "year": volume.get("publishedDate", ""),
                "isbn": isbn,
                "publisher": volume.get("publisher", ""),
                "source": "Google Books",
                "source_id": item.get("id", "")
            })

        return candidates

    except requests.RequestException as error:

        print(f"Google Books error: {error}")
        return []


# ============================================================
# RANK CANDIDATES
# ============================================================

def score_candidate(candidate, query):

    title_score = similarity(
        query,
        candidate["title"]
    )

    author_score = similarity(
        query,
        candidate["authors"]
    )

    # Title matches are more valuable than author-only matches.
    score = max(
        title_score,
        author_score * 0.65
    )

    # Single-word queries are less reliable.
    if len(normalize(query).split()) == 1:
        score = min(score, 0.65)

    return round(score, 3), round(title_score, 3), round(
        author_score, 3
    )


def rank_candidates(candidates, queries):

    ranked = {}

    for candidate in candidates:

        best_score = 0.0
        best_evidence = {}

        for query_info in queries:

            query = query_info["text"]

            score, title_score, author_score = score_candidate(
                candidate,
                query
            )

            if score > best_score:

                best_score = score

                best_evidence = {
                    "matched_query": query,
                    "ocr_method": query_info["method"],
                    "ocr_confidence": query_info["confidence"],
                    "title_similarity": title_score,
                    "author_similarity": author_score,
                    "query_type": query_info["query_type"]
                }

        candidate_key = (
            normalize(candidate["title"]),
            normalize(candidate["authors"]),
            candidate["isbn"]
        )

        if candidate_key not in ranked or (
            best_score > ranked[candidate_key]["score"]
        ):

            ranked[candidate_key] = {
                "candidate": candidate,
                "score": best_score,
                "evidence": best_evidence
            }

    results = sorted(
        ranked.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:MAX_CANDIDATES_PER_IMAGE]


# ============================================================
# CONFIDENCE LABEL
# ============================================================

def get_status(score, evidence):

    title_score = evidence.get("title_similarity", 0)

    # An author-only match should not be called a likely book match.
    if title_score >= 0.80 and score >= 0.80:
        return "Likely title match - verify"

    if title_score >= 0.60 and score >= 0.60:
        return "Possible title match - verify"

    if score >= 0.45:
        return "Weak candidate - verify manually"

    return "Low confidence - verify manually"


# ============================================================
# MAIN IDENTIFICATION PROCESS
# ============================================================

def main():

    print("\nShelfSense AI - Book Identification V4")
    print("=" * 55)

    ocr_data = load_ocr_results()

    print(f"OCR images found: {len(ocr_data)}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Include all crop images, even when OCR found no text.
    crop_dir = BASE_DIR / "outputs" / "cropped_books"

    if crop_dir.exists():

        image_names = sorted(
            path.name
            for path in crop_dir.glob("*")
            if path.suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }
        )

    else:
        image_names = sorted(ocr_data.keys())

    output_rows = []

    total_candidates = 0

    for image_name in image_names:

        print("\n" + "-" * 55)
        print(f"Processing: {image_name}")

        records = ocr_data.get(image_name, [])

        queries = build_queries(records)

        if not queries:

            print("No usable OCR text.")

            output_rows.append({
                "image": image_name,
                "rank": "",
                "title": "",
                "authors": "",
                "score": "",
                "status": "No usable OCR text",
                "year": "",
                "isbn": "",
                "publisher": "",
                "catalogue": "",
                "matched_query": "",
                "ocr_method": "",
                "ocr_confidence": "",
                "title_similarity": "",
                "author_similarity": ""
            })

            continue

        all_candidates = []

        for query_info in queries:

            query = query_info["text"]

            print(f"Searching: {query}")

            open_library_results = search_open_library(query)

            google_books_results = search_google_books(query)

            all_candidates.extend(open_library_results)
            all_candidates.extend(google_books_results)

            time.sleep(REQUEST_DELAY)

        ranked = rank_candidates(
            all_candidates,
            queries
        )

        if not ranked:

            print("No catalogue candidates found.")

            output_rows.append({
                "image": image_name,
                "rank": "",
                "title": "",
                "authors": "",
                "score": "",
                "status": "No catalogue candidates",
                "year": "",
                "isbn": "",
                "publisher": "",
                "catalogue": "",
                "matched_query": "",
                "ocr_method": "",
                "ocr_confidence": "",
                "title_similarity": "",
                "author_similarity": ""
            })

            continue

        for rank, result in enumerate(ranked, start=1):

            candidate = result["candidate"]
            score = result["score"]
            evidence = result["evidence"]

            status = get_status(score, evidence)

            print(f"\nCandidate {rank}")
            print(f"Title: {candidate['title']}")
            print(f"Authors: {candidate['authors']}")
            print(f"Score: {score}")
            print(f"Status: {status}")
            print(f"Catalogue: {candidate['source']}")

            output_rows.append({
                "image": image_name,
                "rank": rank,
                "title": candidate["title"],
                "authors": candidate["authors"],
                "score": score,
                "status": status,
                "year": candidate["year"],
                "isbn": candidate["isbn"],
                "publisher": candidate["publisher"],
                "catalogue": candidate["source"],
                "matched_query": evidence.get("matched_query", ""),
                "ocr_method": evidence.get("ocr_method", ""),
                "ocr_confidence": evidence.get("ocr_confidence", ""),
                "title_similarity": evidence.get("title_similarity", ""),
                "author_similarity": evidence.get("author_similarity", "")
            })

            total_candidates += 1

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "image",
        "rank",
        "title",
        "authors",
        "score",
        "status",
        "year",
        "isbn",
        "publisher",
        "catalogue",
        "matched_query",
        "ocr_method",
        "ocr_confidence",
        "title_similarity",
        "author_similarity"
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
    print("Identification V4 completed!")
    print(f"Images processed: {len(image_names)}")
    print(f"Candidate records: {total_candidates}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
