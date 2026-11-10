
import csv
import re
import time
import requests

from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


# ============================================================
# SHELFSENSE AI - BOOK IDENTIFICATION V4.1
# Uses Open Library only. Google Books temporarily disabled.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OCR_CSV = (
    BASE_DIR
    / "outputs"
    / "ocr_results_paddle_v2"
    / "paddleocr_v2_comparison.csv"
)

CROPS_DIR = BASE_DIR / "outputs" / "cropped_books"

OUTPUT_DIR = BASE_DIR / "outputs" / "book_identification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "identified_books_v4_1.csv"

OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"

MAX_QUERIES_PER_IMAGE = 5
RESULTS_PER_QUERY = 20
TOP_RESULTS_PER_IMAGE = 5

REQUEST_DELAY = 0.6
REQUEST_TIMEOUT = 20

METHOD_PRIORITY = {
    "counterclockwise": 1,
    "original": 2,
    "counterclockwise_contrast": 3,
    "clockwise": 4
}

session = requests.Session()
session.headers.update({
    "User-Agent": "ShelfSenseAI/1.0 (book identification experiment)"
})


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize(text):
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text):
    return set(normalize(text).split())


def similarity(a, b):
    """Compare two text strings. Returns a score between 0 and 1."""

    a = normalize(a)
    b = normalize(b)

    if not a or not b:
        return 0.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    a_tokens = tokens(a)
    b_tokens = tokens(b)

    if not a_tokens or not b_tokens:
        return sequence_score

    overlap = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    return max(sequence_score, overlap)


def split_known_concatenation(text):
    """
    Handle a common OCR issue where AND is joined to other words.

    Example:
    MUNCHANDPHOTOGRAPHY -> MUNCH AND PHOTOGRAPHY
    """

    text = str(text).strip()

    match = re.fullmatch(
        r"([A-Za-z]{3,})AND([A-Za-z]{4,})",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1) + " AND " + match.group(2)

    return text


# ============================================================
# LOAD OCR DATA
# ============================================================

def load_ocr_data():

    if not OCR_CSV.exists():
        raise FileNotFoundError(
            f"OCR CSV not found:\n{OCR_CSV}"
        )

    grouped = defaultdict(list)

    with open(
        OCR_CSV,
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

        if not reader.fieldnames:
            raise ValueError("The OCR CSV has no column headers.")

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}\n"
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:

            image = row.get("image", "").strip()
            method = row.get("method", "").strip()
            detected_text = row.get("detected_text", "").strip()

            try:
                confidence = float(row.get("confidence", 0))
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


# ============================================================
# BUILD SEARCH QUERIES
# ============================================================

def build_queries(image, ocr_records):

    sorted_records = sorted(
        ocr_records,
        key=lambda item: (
            METHOD_PRIORITY.get(item["method"], 99),
            -item["confidence"]
        )
    )

    queries = []
    seen = set()

    for record in sorted_records:

        original_text = record["text"]
        cleaned_text = split_known_concatenation(original_text)
        cleaned = normalize(cleaned_text)

        words = cleaned.split()

        # Ignore very short OCR fragments.
        if len(cleaned) < 4:
            continue

        # Single words are weak title evidence.
        # Keep them only as possible author/subject searches.
        if len(words) == 1 and len(words[0]) < 5:
            continue

        if cleaned in seen:
            continue

        seen.add(cleaned)

        if len(words) == 1:
            query_type = "author_or_subject"
        else:
            query_type = "phrase"

        queries.append({
            "query": cleaned_text,
            "query_type": query_type,
            "method": record["method"],
            "ocr_confidence": record["confidence"]
        })

        if len(queries) >= MAX_QUERIES_PER_IMAGE:
            break

    return queries


# ============================================================
# OPEN LIBRARY SEARCH
# ============================================================

def search_open_library(query):

    params = {
        "q": query,
        "limit": RESULTS_PER_QUERY,
        "fields": (
            "key,title,author_name,"
            "first_publish_year,isbn,edition_count"
        )
    }

    try:

        response = session.get(
            OPEN_LIBRARY_URL,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        return data.get("docs", [])

    except requests.RequestException as error:

        print(f"Open Library request failed: {error}")
        return []

    except ValueError:

        print("Open Library returned invalid JSON.")
        return []


# ============================================================
# SCORE A CANDIDATE AGAINST ITS ORIGINAL QUERY
# ============================================================

def score_candidate(query_info, document):

    query = query_info["query"]
    query_type = query_info["query_type"]

    title = document.get("title") or ""

    authors = document.get("author_name") or []

    author_text = " ".join(authors)

    # A single word such as PICASSO or MUNCH must not
    # count as evidence that the book title matches.
    if query_type == "author_or_subject":
        title_score = 0.0
    else:
        title_score = similarity(query, title)

    author_score = similarity(query, author_text)

    # Keep title and author evidence separate.
    # Author matches are useful but weaker than title matches.
    overall_score = max(
        title_score,
        author_score * 0.80
    )

    if title_score >= 0.75:
        status = "Strong title evidence"

    elif title_score >= 0.50:
        status = "Possible title evidence"

    elif author_score >= 0.65:
        status = "Author/subject evidence only"

    elif overall_score >= 0.35:
        status = "Weak evidence"

    else:
        status = "Very weak evidence"

    return {
        "title_score": round(title_score, 3),
        "author_score": round(author_score, 3),
        "overall_score": round(overall_score, 3),
        "status": status
    }


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def identify_image(image, ocr_records):

    print("\n" + "=" * 65)
    print(f"Processing: {image}")
    print("=" * 65)

    queries = build_queries(image, ocr_records)

    if not queries:

        print("No usable OCR search queries.")

        return [{
            "image": image,
            "rank": "",
            "matched_title": "",
            "authors": "",
            "year": "",
            "isbn": "",
            "matched_ocr_query": "",
            "query_type": "",
            "ocr_method": "",
            "ocr_confidence": "",
            "title_score": "",
            "author_score": "",
            "overall_score": "",
            "status": "No usable OCR"
        }]

    print("Search queries:")

    for item in queries:
        print(
            f"  - {item['query']} "
            f"({item['query_type']}, {item['method']})"
        )

    candidates = {}

    for query_info in queries:

        query = query_info["query"]

        print(f"\nSearching Open Library: {query}")

        documents = search_open_library(query)

        print(f"Catalogue results: {len(documents)}")

        for document in documents:

            key = document.get("key")

            if not key:
                continue

            scores = score_candidate(query_info, document)

            candidate = {
                "image": image,
                "matched_title": document.get("title", ""),
                "authors": "; ".join(
                    document.get("author_name", [])
                ),
                "year": document.get("first_publish_year", ""),
                "isbn": "; ".join(
                    document.get("isbn", [])[:5]
                ),
                "matched_ocr_query": query,
                "query_type": query_info["query_type"],
                "ocr_method": query_info["method"],
                "ocr_confidence": query_info["ocr_confidence"],
                **scores
            }

            # Keep the best evidence for each catalogue record.
            # Crucially, each score is calculated against the
            # exact OCR query that retrieved this record.
            existing = candidates.get(key)

            if (
                existing is None
                or candidate["overall_score"]
                > existing["overall_score"]
            ):
                candidates[key] = candidate

        time.sleep(REQUEST_DELAY)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            item["overall_score"],
            item["title_score"],
            item["author_score"]
        ),
        reverse=True
    )

    if not ranked:

        print("\nNo catalogue candidates found.")

        return [{
            "image": image,
            "rank": "",
            "matched_title": "",
            "authors": "",
            "year": "",
            "isbn": "",
            "matched_ocr_query": "",
            "query_type": "",
            "ocr_method": "",
            "ocr_confidence": "",
            "title_score": "",
            "author_score": "",
            "overall_score": "",
            "status": "No catalogue candidates"
        }]

    output_rows = []

    print("\nTop catalogue candidates:")

    for rank, candidate in enumerate(
        ranked[:TOP_RESULTS_PER_IMAGE],
        start=1
    ):

        candidate["rank"] = rank
        output_rows.append(candidate)

        print(
            f"\n{rank}. {candidate['matched_title']}"
            f"\n   Authors: {candidate['authors']}"
            f"\n   OCR query: {candidate['matched_ocr_query']}"
            f"\n   Title score: {candidate['title_score']}"
            f"\n   Author score: {candidate['author_score']}"
            f"\n   Overall score: {candidate['overall_score']}"
            f"\n   Assessment: {candidate['status']}"
        )

    return output_rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nShelfSense AI - Book Identification V4.1")
    print("=" * 65)

    if not CROPS_DIR.exists():
        raise FileNotFoundError(
            f"Book crop folder not found:\n{CROPS_DIR}"
        )

    ocr_data = load_ocr_data()

    image_files = sorted(
        file.name
        for file in CROPS_DIR.iterdir()
        if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    print(f"OCR images with text: {len(ocr_data)}")
    print(f"Book crops found: {len(image_files)}")

    all_results = []

    for image in image_files:

        records = ocr_data.get(image, [])

        results = identify_image(image, records)

        all_results.extend(results)

    fieldnames = [
        "image",
        "rank",
        "matched_title",
        "authors",
        "year",
        "isbn",
        "matched_ocr_query",
        "query_type",
        "ocr_method",
        "ocr_confidence",
        "title_score",
        "author_score",
        "overall_score",
        "status"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 65)
    print("IDENTIFICATION V4.1 COMPLETED")
    print("=" * 65)

    print(f"Images processed: {len(image_files)}")
    print(f"Candidate records: {len(all_results)}")
    print(f"Results saved to:\n{OUTPUT_CSV}")

    print("\nImportant:")
    print("These are catalogue candidates, not verified book identities.")
    print("Check the physical book cover/spine before confirming a match.")


if __name__ == "__main__":
    main()
