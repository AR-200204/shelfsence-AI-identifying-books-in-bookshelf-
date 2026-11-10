
import csv
import re
import time
import requests
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

# ============================================================
# SHELFSENSE AI - BOOK IDENTIFICATION V4.2
# Uses PaddleOCR V2 results + Open Library
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OCR_CSV = (
    BASE_DIR
    / "outputs"
    / "ocr_results_paddle_v2"
    / "paddleocr_v2_comparison.csv"
)

CROP_DIR = BASE_DIR / "outputs" / "cropped_books"

OUTPUT_DIR = BASE_DIR / "outputs" / "book_identification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "identified_books_v4_2.csv"

OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"

# Be gentle with the public Open Library API.
REQUEST_DELAY = 0.7
MAX_RESULTS_PER_QUERY = 10

METHOD_PRIORITY = {
    "counterclockwise": 1,
    "original": 2,
    "counterclockwise_contrast": 3,
    "clockwise": 4
}

# Common OCR corrections.
OCR_CORRECTIONS = {
    "willtam": "william",
    "mckean": "mckean",
    "picasse": "picasso",
    "hibmer": "hirmer",
    "pa iry": "fairy",
    "fai ry": "fairy",
}

# Known concatenated words that frequently appear in this dataset.
WORD_SPACING_FIXES = {
    "americangrotesque": "american grotesque",
    "picturesthattick": "pictures that tick",
    "postmortem": "post mortem",
    "post-mortem": "post mortem",
    "edvardmunch": "edvard munch",
    "william mortensen": "william mortensen",
    "dave mckean": "dave mckean",
    "johnpiper": "john piper",
    "mothmeister": "mothmeister",
    "munchandphotography": "munch and photography",
    "william morris": "william morris",
}

# Words that usually indicate a title rather than an author's name.
TITLE_WORDS = {
    "the", "art", "life", "and", "of", "love", "angst",
    "pictures", "that", "tick", "american", "grotesque",
    "weird", "wonderful", "post", "mortem", "fairy",
    "tales", "prints", "early", "years", "photography",
    "book", "volume", "vol"
}

# Publishers and other terms that should not be treated as author names.
PUBLISHER_WORDS = {
    "dark horse", "darkhorse", "hirmer", "lannoo",
    "penguin", "random house", "university press"
}

session = requests.Session()
session.headers.update({
    "User-Agent": "ShelfSenseAI/0.1 (educational book identification project)"
})


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Clean OCR text and apply known corrections."""

    text = str(text).lower().strip()

    text = text.replace("&", " and ")
    text = text.replace(":", " ")
    text = text.replace(",", " ")
    text = text.replace(".", " ")
    text = text.replace("-", " ")

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Apply common OCR spelling corrections.
    words = text.split()
    corrected_words = [
        OCR_CORRECTIONS.get(word, word)
        for word in words
    ]
    text = " ".join(corrected_words)

    # Restore spaces in known concatenated phrases.
    compact = text.replace(" ", "")

    if compact in WORD_SPACING_FIXES:
        text = WORD_SPACING_FIXES[compact]

    return text.strip()


def similarity(a, b):
    """Calculate similarity between two normalized strings."""

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    a_words = set(a.split())
    b_words = set(b.split())

    if not a_words or not b_words:
        token_score = 0.0
    else:
        token_score = len(a_words & b_words) / len(a_words | b_words)

    return max(sequence_score, token_score)


def token_coverage(query, candidate):
    """How much of the OCR query appears in the catalogue text."""

    query_words = set(normalize_text(query).split())
    candidate_words = set(normalize_text(candidate).split())

    if not query_words:
        return 0.0

    return len(query_words & candidate_words) / len(query_words)


# ============================================================
# OCR DATA LOADING
# ============================================================

def load_ocr_data():
    if not OCR_CSV.exists():
        raise FileNotFoundError(
            f"OCR CSV not found:\n{OCR_CSV}"
        )

    grouped = defaultdict(lambda: defaultdict(list))

    with open(OCR_CSV, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "image", "method", "detected_text", "confidence"
        }

        if not required_columns.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "The OCR CSV must contain these columns: "
                "image, method, detected_text, confidence\n"
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            image = row.get("image", "").strip()
            method = row.get("method", "").strip()
            text = row.get("detected_text", "").strip()

            if not image or not text:
                continue

            try:
                confidence = float(row.get("confidence", 0))
            except (ValueError, TypeError):
                confidence = 0.0

            grouped[image][method].append({
                "text": text,
                "normalized": normalize_text(text),
                "confidence": confidence
            })

    return grouped


# ============================================================
# SELECT USEFUL OCR EVIDENCE
# ============================================================

def get_image_evidence(methods):
    """Select useful OCR text while preserving method information."""

    evidence = []

    sorted_methods = sorted(
        methods.items(),
        key=lambda item: METHOD_PRIORITY.get(item[0], 99)
    )

    for method, detections in sorted_methods:
        cleaned = []

        for detection in detections:
            text = detection["normalized"]

            if not text or len(text) < 2:
                continue

            # Ignore mostly numeric OCR results.
            if not re.search(r"[a-z]", text):
                continue

            if text not in cleaned:
                cleaned.append(text)

            evidence.append({
                "text": text,
                "method": method,
                "confidence": detection["confidence"]
            })

        # Add the combined text from this OCR method.
        if cleaned:
            combined = " ".join(cleaned)

            if len(combined.split()) >= 2:
                evidence.append({
                    "text": combined,
                    "method": method,
                    "confidence": 0.0
                })

    # Remove duplicate evidence.
    unique = {}
    for item in evidence:
        key = (item["text"], item["method"])
        if key not in unique:
            unique[key] = item

    return list(unique.values())


# ============================================================
# CLASSIFY OCR CLUES
# ============================================================

def looks_like_publisher(text):
    text = normalize_text(text)

    return any(
        publisher in text
        for publisher in PUBLISHER_WORDS
    )


def looks_like_author_name(text):
    """A cautious heuristic, not a definitive classification."""

    text = normalize_text(text)
    words = text.split()

    if not words or looks_like_publisher(text):
        return False

    if len(words) > 3:
        return False

    if any(word in TITLE_WORDS for word in words):
        return False

    # One-word names can be useful, but only if reasonably distinctive.
    if len(words) == 1:
        return len(words[0]) >= 7

    return len(words) in (2, 3)


def build_search_queries(evidence):
    """Create separate title and author search queries."""

    title_queries = []
    author_queries = []

    # Highest priority evidence first.
    evidence = sorted(
        evidence,
        key=lambda item: (
            METHOD_PRIORITY.get(item["method"], 99),
            -item["confidence"]
        )
    )

    for item in evidence:
        text = normalize_text(item["text"])

        if len(text) < 3:
            continue

        if looks_like_publisher(text):
            continue

        words = text.split()

        if len(words) >= 2:
            if text not in title_queries:
                title_queries.append(text)

        if looks_like_author_name(text):
            if text not in author_queries:
                author_queries.append(text)

    # Limit the number of API calls.
    return title_queries[:3], author_queries[:2]


# ============================================================
# OPEN LIBRARY SEARCH
# ============================================================

def search_open_library(field, query):
    """Search a specific Open Library field."""

    params = {
        field: query,
        "limit": MAX_RESULTS_PER_QUERY
    }

    try:
        response = session.get(
            OPEN_LIBRARY_URL,
            params=params,
            timeout=20
        )

        if response.status_code == 429:
            print("Open Library rate limit reached. Waiting...")
            time.sleep(3)
            return []

        response.raise_for_status()

        data = response.json()
        return data.get("docs", [])

    except requests.RequestException as error:
        print(f"API error for {field}='{query}': {error}")
        return []

    except ValueError:
        print(f"Invalid JSON response for query: {query}")
        return []

    finally:
        time.sleep(REQUEST_DELAY)


# ============================================================
# CANDIDATE SCORING
# ============================================================

def get_candidate_title(record):
    return record.get("title", "") or ""


def get_candidate_authors(record):
    authors = record.get("author_name", [])
    return "; ".join(authors) if authors else ""


def get_candidate_publishers(record):
    publishers = record.get("publisher", [])
    return "; ".join(publishers) if publishers else ""


def score_candidate(record, query, evidence_type):
    """
    Score a candidate only against the query that retrieved it.

    This avoids the V4 problem where unrelated OCR queries
    could increase the score of the same catalogue record.
    """

    title = get_candidate_title(record)
    authors = get_candidate_authors(record)
    publishers = get_candidate_publishers(record)

    title_score = similarity(query, title)
    author_score = similarity(query, authors)
    publisher_score = similarity(query, publishers)

    title_coverage = token_coverage(query, title)
    author_coverage = token_coverage(query, authors)

    if evidence_type == "title":
        # Title evidence should be the strongest signal.
        score = (
            title_score * 0.75
            + title_coverage * 0.25
        )

        # A title query should not be rescued by a publisher match.
        status = "Title evidence"

    else:
        # Author query: author similarity is the main signal.
        score = (
            author_score * 0.85
            + author_coverage * 0.15
        )

        status = "Author evidence"

    # Publisher match is recorded, but never boosts the main score.
    if publisher_score >= 0.75:
        status += " + publisher clue"

    return {
        "score": round(score, 4),
        "title_score": round(title_score, 4),
        "author_score": round(author_score, 4),
        "publisher_score": round(publisher_score, 4),
        "status": status
    }


def confidence_label(score, evidence_type):
    if evidence_type == "title":
        if score >= 0.85:
            return "Strong title candidate"
        elif score >= 0.65:
            return "Possible title candidate"
        else:
            return "Needs verification"

    else:
        if score >= 0.90:
            return "Strong author candidate"
        elif score >= 0.70:
            return "Possible author candidate"
        else:
            return "Needs verification"


# ============================================================
# IDENTIFY ONE BOOK
# ============================================================

def identify_book(image_name, methods):
    evidence = get_image_evidence(methods)

    title_queries, author_queries = build_search_queries(evidence)

    candidates = {}

    # Search title evidence.
    for query in title_queries:
        records = search_open_library("title", query)

        for record in records:
            result = score_candidate(
                record,
                query,
                "title"
            )

            add_candidate(
                candidates,
                record,
                query,
                "title",
                result
            )

    # Search author evidence separately.
    for query in author_queries:
        records = search_open_library("author", query)

        for record in records:
            result = score_candidate(
                record,
                query,
                "author"
            )

            add_candidate(
                candidates,
                record,
                query,
                "author",
                result
            )

    ranked = sorted(
        candidates.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    rows = []

    for rank, candidate in enumerate(ranked[:5], start=1):
        rows.append({
            "image": image_name,
            "rank": rank,
            "title": candidate["title"],
            "authors": candidate["authors"],
            "first_publish_year": candidate["year"],
            "publishers": candidate["publishers"],
            "isbn": candidate["isbn"],
            "matched_query": candidate["matched_query"],
            "evidence_type": candidate["evidence_type"],
            "score": candidate["score"],
            "title_score": candidate["title_score"],
            "author_score": candidate["author_score"],
            "publisher_score": candidate["publisher_score"],
            "status": candidate["status"],
            "confidence_label": candidate["confidence_label"],
            "open_library_key": candidate["key"]
        })

    return rows, evidence, title_queries, author_queries


def add_candidate(candidates, record, query, evidence_type, result):
    key = record.get("key")

    if not key:
        key = (
            normalize_text(get_candidate_title(record))
            + "|"
            + normalize_text(get_candidate_authors(record))
        )

    candidate_id = f"{key}|{evidence_type}"

    candidate = {
        "key": key,
        "title": get_candidate_title(record),
        "authors": get_candidate_authors(record),
        "year": record.get("first_publish_year", ""),
        "publishers": get_candidate_publishers(record),
        "isbn": "; ".join(record.get("isbn", [])[:5]),
        "matched_query": query,
        "evidence_type": evidence_type,
        "score": result["score"],
        "title_score": result["title_score"],
        "author_score": result["author_score"],
        "publisher_score": result["publisher_score"],
        "status": result["status"],
        "confidence_label": confidence_label(
            result["score"],
            evidence_type
        )
    }

    existing = candidates.get(candidate_id)

    # Keep the strongest result for this record and evidence type.
    if existing is None or candidate["score"] > existing["score"]:
        candidates[candidate_id] = candidate


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    print("\nShelfSense AI - Book Identification V4.2")
    print("=" * 55)

    if not CROP_DIR.exists():
        raise FileNotFoundError(
            f"Crop folder not found:\n{CROP_DIR}"
        )

    crop_images = sorted(CROP_DIR.glob("*.jpg"))

    if not crop_images:
        crop_images = sorted(CROP_DIR.glob("*.jpeg"))

    if not crop_images:
        raise FileNotFoundError(
            f"No JPG crops found in:\n{CROP_DIR}"
        )

    ocr_data = load_ocr_data()

    print(f"OCR images with text: {len(ocr_data)}")
    print(f"Book crops found: {len(crop_images)}")
    print("\nStarting catalogue identification...\n")

    all_rows = []

    for index, image_path in enumerate(crop_images, start=1):
        image_name = image_path.name

        print("-" * 55)
        print(f"[{index}/{len(crop_images)}] Processing {image_name}")

        methods = ocr_data.get(image_name, {})

        if not methods:
            print("No OCR text found for this image.")
            continue

        rows, evidence, title_queries, author_queries = identify_book(
            image_name,
            methods
        )

        print("Title queries:", title_queries)
        print("Author queries:", author_queries)

        if rows:
            for row in rows[:3]:
                print(
                    f"  {row['rank']}. {row['title']} "
                    f"| {row['authors']} "
                    f"| Score: {row['score']} "
                    f"| {row['confidence_label']}"
                )

            all_rows.extend(rows)
        else:
            print("No catalogue candidates found.")

    # Save results.
    fieldnames = [
        "image",
        "rank",
        "title",
        "authors",
        "first_publish_year",
        "publishers",
        "isbn",
        "matched_query",
        "evidence_type",
        "score",
        "title_score",
        "author_score",
        "publisher_score",
        "status",
        "confidence_label",
        "open_library_key"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 55)
    print("IDENTIFICATION COMPLETED")
    print("=" * 55)

    print(f"Images processed: {len(crop_images)}")
    print(f"Candidate records: {len(all_rows)}")
    print(f"Results saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
