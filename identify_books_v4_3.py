
import csv
import re
import time
import requests
import unicodedata

from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

# ============================================================
# SHELFSENSE AI - BOOK IDENTIFICATION V4.3
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OCR_CSV = (
    BASE_DIR / "outputs" / "ocr_results_paddle_v2"
    / "paddleocr_v2_comparison.csv"
)

CROPS_DIR = BASE_DIR / "outputs" / "cropped_books"

OUTPUT_DIR = BASE_DIR / "outputs" / "book_identification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "identified_books_v4_3.csv"

API_URL = "https://openlibrary.org/search.json"

HEADERS = {
    "User-Agent": "ShelfSenseAI/4.3 (educational book identification project)"
}

MAX_RESULTS = 20
REQUEST_DELAY = 0.15

# Known corrections based on observed OCR errors.
OCR_CORRECTIONS = {
    "willtam mortensen": "william mortensen",
    "faited by giuna bartrum": "edited by giulia bartrum",
    "pairy tales": "fairy tales",
    "pictures that tick": "pictures that tick",
    "munchandphotography": "munch and photography",
    "theartof john piper": "the art of john piper",
    "theartor john piper": "the art of john piper",
}

# Words that should not be treated as book titles or authors.
PUBLISHER_WORDS = {
    "darkhorse",
    "dark horse",
    "hirmer",
    "lannoo",
    "penguin",
    "random house",
    "publisher",
}

STOPWORDS = {
    "the", "a", "an", "and", "of", "by", "for", "with",
    "edited", "illustrated", "volume", "vol"
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text if not unicodedata.combining(c)
    )

    text = text.replace("&", " and ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def correct_ocr(text):
    cleaned = normalize(text)

    for wrong, right in OCR_CORRECTIONS.items():
        if normalize(wrong) in cleaned:
            cleaned = cleaned.replace(
                normalize(wrong), normalize(right)
            )

    # Restore spaces in common concatenated phrases.
    replacements = {
        "american grotesque": "american grotesque",
        "pictures that tick": "pictures that tick",
        "munch and photography": "munch and photography",
        "post mortem fairy tales": "post mortem fairy tales",
        "otto mueller": "otto mueller",
        "william morris": "william morris",
    }

    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    return re.sub(r"\s+", " ", cleaned).strip()


def tokens(text):
    return {
        word for word in normalize(text).split()
        if word not in STOPWORDS and len(word) > 1
    }


def similarity(a, b):
    a = normalize(a)
    b = normalize(b)

    if not a or not b:
        return 0.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    a_tokens = tokens(a)
    b_tokens = tokens(b)

    if not a_tokens or not b_tokens:
        token_score = 0.0
    else:
        token_score = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    return max(sequence_score, token_score)


def token_coverage(query, candidate):
    q = tokens(query)
    c = tokens(candidate)

    if not q:
        return 0.0

    return len(q & c) / len(q)


# ============================================================
# LOAD OCR RESULTS
# ============================================================

def load_ocr():
    grouped = defaultdict(list)

    with open(OCR_CSV, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {"image", "detected_text"}

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "OCR CSV must contain image and detected_text columns."
            )

        for row in reader:
            image = Path(row.get("image", "")).name
            text = row.get("detected_text", "").strip()
            method = row.get("method", "").strip().lower()

            if image and text:
                grouped[image].append({
                    "text": text,
                    "method": method
                })

    return grouped


# ============================================================
# BUILD SEARCH QUERIES
# ============================================================

def build_queries(rows):
    title_queries = []
    author_queries = []

    for row in rows:
        raw = row["text"]
        cleaned = correct_ocr(raw)
        normalized = normalize(cleaned)

        if not normalized:
            continue

        words = normalized.split()

        if any(
            publisher in normalized
            for publisher in PUBLISHER_WORDS
        ):
            continue

        # Ignore extremely short and generic text.
        if len(words) == 1 and len(words[0]) < 5:
            continue

        # Remove obvious publisher-only detections.
        if normalized in PUBLISHER_WORDS:
            continue

        if len(words) >= 2:
            title_queries.append(cleaned)

            # Detect likely author names.
            if len(words) <= 4 and not any(
                word in STOPWORDS for word in words
            ):
                author_queries.append(cleaned)

        else:
            # Single words are only weak author clues.
            author_queries.append(cleaned)

    # Keep unique queries in original order.
    title_queries = list(dict.fromkeys(title_queries))
    author_queries = list(dict.fromkeys(author_queries))

    return title_queries[:5], author_queries[:5]


# ============================================================
# OPEN LIBRARY SEARCH
# ============================================================

def search_open_library(session, query, field=None):
    params = {
        "limit": MAX_RESULTS,
        "fields": (
            "key,title,author_name,first_publish_year,"
            "publisher,isbn,edition_count"
        )
    }

    if field == "title":
        params["title"] = query
    elif field == "author":
        params["author"] = query
    else:
        params["q"] = query

    try:
        response = session.get(
            API_URL,
            params=params,
            headers=HEADERS,
            timeout=20
        )

        response.raise_for_status()
        return response.json().get("docs", [])

    except requests.RequestException as error:
        print("Open Library error:", error)
        return []


# ============================================================
# SCORE CANDIDATES
# ============================================================

def score_candidate(query, doc, evidence_type):
    title = doc.get("title", "")
    authors = doc.get("author_name", []) or []
    publishers = doc.get("publisher", []) or []

    author_text = " ".join(authors)
    publisher_text = " ".join(publishers)

    title_score = similarity(query, title)
    author_score = similarity(query, author_text)
    publisher_score = similarity(query, publisher_text)

    coverage = token_coverage(query, title)

    # For title evidence, require meaningful title similarity.
    if evidence_type == "title":
        score = (
            0.70 * title_score
            + 0.20 * coverage
            + 0.10 * author_score
        )

    else:
        score = (
            0.75 * author_score
            + 0.20 * title_score
            + 0.05 * publisher_score
        )

    # Avoid publisher-only evidence becoming a false match.
    if evidence_type == "author" and author_score < 0.45:
        score *= 0.55

    # An exact title match is useful but does not prove the edition.
    if normalize(query) == normalize(title):
        score = max(score, 0.95)

    return {
        "score": round(min(score, 1.0), 4),
        "title_score": round(title_score, 4),
        "author_score": round(author_score, 4),
        "publisher_score": round(publisher_score, 4),
        "coverage": round(coverage, 4),
    }


def classify_candidate(score, title_score, author_score, evidence_type):
    if evidence_type == "title" and title_score >= 0.90:
        return "Strong title candidate"

    if evidence_type == "title" and title_score >= 0.70:
        return "Possible title candidate"

    if evidence_type == "author" and author_score >= 0.80:
        return "Author evidence only"

    if score >= 0.55:
        return "Needs verification"

    return "Low confidence"


# ============================================================
# MAIN IDENTIFICATION
# ============================================================

def main():
    print("\nShelfSense AI - Book Identification V4.3")
    print("=" * 60)

    if not OCR_CSV.exists():
        print("ERROR: OCR CSV not found:")
        print(OCR_CSV)
        return

    if not CROPS_DIR.exists():
        print("ERROR: Cropped books folder not found:")
        print(CROPS_DIR)
        return

    ocr_data = load_ocr()
    crop_files = sorted(CROPS_DIR.glob("*.jpg"))

    print("OCR images:", len(ocr_data))
    print("Book crops:", len(crop_files))

    session = requests.Session()
    all_results = []

    for index, crop in enumerate(crop_files, start=1):
        image_name = crop.name

        print("\n" + "-" * 60)
        print(f"[{index}/{len(crop_files)}] {image_name}")

        rows = ocr_data.get(image_name, [])

        if not rows:
            print("No OCR text.")
            continue

        title_queries, author_queries = build_queries(rows)

        print("Title queries:", title_queries)
        print("Author queries:", author_queries)

        candidate_records = {}

        searches = []

        for query in title_queries:
            searches.append((query, "title", "title"))

        for query in author_queries:
            searches.append((query, "author", "author"))

        # Use structured title/author searches.
        for query, field, evidence_type in searches:
            docs = search_open_library(
                session, query, field
            )

            time.sleep(REQUEST_DELAY)

            for doc in docs:
                key = doc.get("key")

                if not key:
                    continue

                scores = score_candidate(
                    query, doc, evidence_type
                )

                candidate = {
                    "image": image_name,
                    "title": doc.get("title", ""),
                    "authors": "; ".join(
                        doc.get("author_name", []) or []
                    ),
                    "first_publish_year": doc.get(
                        "first_publish_year", ""
                    ),
                    "publishers": "; ".join(
                        doc.get("publisher", []) or []
                    ),
                    "isbn": "; ".join(
                        (doc.get("isbn", []) or [])[:5]
                    ),
                    "matched_query": query,
                    "evidence_type": evidence_type,
                    **scores,
                    "open_library_key": key,
                }

                old = candidate_records.get(key)

                if old is None or candidate["score"] > old["score"]:
                    candidate_records[key] = candidate

        ranked = sorted(
            candidate_records.values(),
            key=lambda item: (
                item["score"],
                item["title_score"],
                item["author_score"]
            ),
            reverse=True
        )

        for rank, candidate in enumerate(ranked[:5], start=1):
            candidate["rank"] = rank

            candidate["confidence_label"] = classify_candidate(
                candidate["score"],
                candidate["title_score"],
                candidate["author_score"],
                candidate["evidence_type"]
            )

            all_results.append(candidate)

            print(
                f"{rank}. {candidate['title']} | "
                f"{candidate['authors']} | "
                f"Score: {candidate['score']} | "
                f"{candidate['confidence_label']}"
            )

        if not ranked:
            print("No catalogue candidates found.")

    # Save results.
    fieldnames = [
        "image", "rank", "title", "authors",
        "first_publish_year", "publishers", "isbn",
        "matched_query", "evidence_type", "score",
        "title_score", "author_score", "publisher_score",
        "coverage", "confidence_label", "open_library_key"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 60)
    print("V4.3 IDENTIFICATION COMPLETED")
    print("=" * 60)
    print("Images processed:", len(crop_files))
    print("Candidate records:", len(all_results))
    print("Saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
