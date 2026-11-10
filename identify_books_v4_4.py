
import csv
import re
import time
import unicodedata
import requests

from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


# ============================================================
# SHELFSENSE AI - BOOK IDENTIFICATION V4.4
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

OUTPUT_CSV = OUTPUT_DIR / "identified_books_v4_4.csv"

API_URL = "https://openlibrary.org/search.json"

HEADERS = {
    "User-Agent": "ShelfSenseAI/4.4 (educational project)"
}

MAX_RESULTS = 30
REQUEST_DELAY = 0.15


# ============================================================
# OCR CORRECTIONS
# ============================================================

# Exact compact forms that OCR may produce without spaces.
COMPACT_CORRECTIONS = {
    "americangrotesque": "american grotesque",
    "picturesthattick": "pictures that tick",
    "theartofjohnpiper": "the art of john piper",
    "munchandphotography": "munch and photography",
    "postmortem": "post mortem",
    "williammortensen": "william mortensen",
    "williammorris": "william morris",
    "ottomueller": "otto mueller",
    "davidmckean": "dave mckean",
}

WORD_CORRECTIONS = {
    "willtam": "william",
    "pairy": "fairy",
    "fowter": "fowler",
    "muncn": "munch",
    "mckean": "mckean",
}

STOPWORDS = {
    "the", "a", "an", "and", "of", "by", "for",
    "with", "from", "in", "on", "to", "at",
    "edited", "illustrated", "volume", "vol"
}

GENERIC_AUTHOR_WORDS = {
    "angst", "art", "book", "books", "photography",
    "pictures", "picture", "life", "years", "volume",
    "munch", "the", "and", "edition"
}

PUBLISHERS = {
    "dark horse",
    "darkhorse",
    "hirmer",
    "penguin",
    "lannoo",
    "harper collins",
    "random house"
}


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize(text):
    text = str(text or "").lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def correct_ocr(text):
    text = str(text or "").strip()

    # Fix known joined phrases before removing punctuation.
    compact = re.sub(r"[^a-z0-9]", "", text.lower())

    if compact in COMPACT_CORRECTIONS:
        return COMPACT_CORRECTIONS[compact]

    # Handle joined words occurring inside longer OCR lines.
    replacements = {
        "picturesthattick": "pictures that tick",
        "americangrotesque": "american grotesque",
        "theartof": "the art of",
        "munchandphotography": "munch and photography",
        "postmortem": "post mortem",
    }

    lower = text.lower()

    for old, new in replacements.items():
        lower = lower.replace(old, new)

    words = normalize(lower).split()

    corrected = [
        WORD_CORRECTIONS.get(word, word)
        for word in words
    ]

    return " ".join(corrected)


def meaningful_tokens(text):
    return [
        word
        for word in normalize(text).split()
        if word not in STOPWORDS and len(word) > 1
    ]


def similarity(a, b):
    a = normalize(a)
    b = normalize(b)

    if not a or not b:
        return 0.0

    sequence_score = SequenceMatcher(
        None, a, b
    ).ratio()

    a_tokens = set(meaningful_tokens(a))
    b_tokens = set(meaningful_tokens(b))

    if not a_tokens or not b_tokens:
        token_score = 0.0
    else:
        token_score = (
            len(a_tokens & b_tokens)
            / len(a_tokens | b_tokens)
        )

    return max(sequence_score, token_score)


def coverage_score(query, candidate):
    query_tokens = set(meaningful_tokens(query))
    candidate_tokens = set(meaningful_tokens(candidate))

    if not query_tokens:
        return 0.0

    return (
        len(query_tokens & candidate_tokens)
        / len(query_tokens)
    )


# ============================================================
# OCR QUALITY FILTER
# ============================================================

def valid_title_query(text):
    text = normalize(text)

    if not text:
        return False

    letters = re.findall(r"[a-z]", text)
    digits = re.findall(r"[0-9]", text)
    words = meaningful_tokens(text)

    if len(letters) < 6:
        return False

    if len(letters) / max(len(text), 1) < 0.55:
        return False

    if len(words) < 2:
        return False

    if len(digits) > len(letters):
        return False

    # Ignore publisher-only detections.
    if text in PUBLISHERS:
        return False

    return True


def valid_author_query(text):
    text = normalize(text)

    if not text:
        return False

    words = meaningful_tokens(text)

    if len(text.replace(" ", "")) < 5:
        return False

    if len(words) > 4:
        return False

    if len(words) == 1 and words[0] in GENERIC_AUTHOR_WORDS:
        return False

    if text in PUBLISHERS:
        return False

    return True


# ============================================================
# LOAD OCR DATA
# ============================================================

def load_ocr():
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
            "detected_text",
            "method"
        }

        available_columns = set(reader.fieldnames or [])

        if not required_columns.issubset(available_columns):
            raise ValueError(
                "OCR CSV must contain these columns: "
                "image, method, detected_text"
            )

        for row in reader:
            image_name = Path(
                row.get("image", "")
            ).name

            raw_text = row.get(
                "detected_text", ""
            ).strip()

            method = row.get(
                "method", ""
            ).strip()

            if not image_name or not raw_text:
                continue

            grouped[image_name].append({
                "raw_text": raw_text,
                "corrected_text": correct_ocr(raw_text),
                "method": method
            })

    return grouped


# ============================================================
# BUILD TITLE AND AUTHOR QUERIES
# ============================================================

def build_queries(ocr_rows):
    title_queries = []
    author_queries = []

    for row in ocr_rows:
        corrected = row["corrected_text"]
        cleaned = normalize(corrected)

        if not cleaned:
            continue

        # Title queries require multiple meaningful words.
        if valid_title_query(cleaned):
            title_queries.append(cleaned)

        # Author clues are kept separately.
        # Split combined names such as:
        # "David Fraser Jenkins and Hugh Fowler Wright".
        author_parts = re.split(
            r"\s+and\s+|[;,]",
            cleaned
        )

        for part in author_parts:
            part = part.strip()

            # Remove common role prefixes.
            part = re.sub(
                r"^(edited by|written by|by)\s+",
                "",
                part
            )

            if valid_author_query(part):
                author_queries.append(part)

    # Remove duplicate queries without changing order.
    title_queries = list(dict.fromkeys(title_queries))
    author_queries = list(dict.fromkeys(author_queries))

    return title_queries[:8], author_queries[:8]


# ============================================================
# OPEN LIBRARY SEARCH
# ============================================================

def search_open_library(session, query, search_type):
    params = {
        "limit": MAX_RESULTS,
        "fields": (
            "key,title,author_name,first_publish_year,"
            "publisher,isbn,edition_count"
        )
    }

    if search_type == "title":
        params["title"] = query

    elif search_type == "author":
        params["author"] = query

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
        print(
            f"Open Library search failed for "
            f"'{query}': {error}"
        )

        return []


# ============================================================
# EVIDENCE SCORING
# ============================================================

def get_candidate_evidence(candidate, query, evidence_type):
    title = candidate.get("title", "")

    authors = candidate.get(
        "author_name", []
    ) or []

    author_text = " ".join(authors)

    if evidence_type == "title":
        title_similarity = similarity(query, title)
        title_coverage = coverage_score(query, title)

        return {
            "title_score": title_similarity,
            "title_coverage": title_coverage,
            "title_query": query
        }

    author_similarity = similarity(query, author_text)
    author_coverage = coverage_score(query, author_text)

    return {
        "author_score": author_similarity,
        "author_coverage": author_coverage,
        "author_query": query
    }


def calculate_final_score(title_score, title_coverage,
                          author_score, has_title_evidence):

    if not has_title_evidence:
        # Author-only evidence cannot confirm a book.
        return round(
            0.70 * author_score,
            4
        )

    if title_coverage < 0.20 or title_score < 0.35:
        return 0.0

    score = (
        0.65 * title_score
        + 0.25 * title_coverage
        + 0.10 * author_score
    )

    # Penalize weak title coverage.
    if title_coverage < 0.40:
        score *= 0.50

    elif title_coverage < 0.60:
        score *= 0.80

    return round(score, 4)


# ============================================================
# CONFIDENCE LABELS
# ============================================================

def classify_candidate(score, title_score,
                       title_coverage, has_title_evidence):

    if not has_title_evidence:
        return "Author clue only"

    if title_coverage < 0.30 or title_score < 0.35:
        return "Reject"

    if (
        score >= 0.85
        and title_score >= 0.90
        and title_coverage >= 0.80
    ):
        return "High"

    if (
        score >= 0.65
        and title_score >= 0.70
        and title_coverage >= 0.60
    ):
        return "Medium"

    return "Low"


# ============================================================
# PROCESS ONE BOOK IMAGE
# ============================================================

def identify_one_book(session, image_name, ocr_rows):
    title_queries, author_queries = build_queries(ocr_rows)

    print("Title queries:", title_queries)
    print("Author queries:", author_queries)

    candidates = {}

    # --------------------------------------------------------
    # TITLE SEARCHES
    # --------------------------------------------------------

    for query in title_queries:
        docs = search_open_library(
            session,
            query,
            "title"
        )

        time.sleep(REQUEST_DELAY)

        for doc in docs:
            key = doc.get("key")

            if not key:
                continue

            record = candidates.setdefault(
                key,
                {
                    "doc": doc,
                    "title_score": 0.0,
                    "title_coverage": 0.0,
                    "author_score": 0.0,
                    "author_coverage": 0.0,
                    "title_query": "",
                    "author_query": "",
                    "has_title_evidence": False,
                    "has_author_evidence": False
                }
            )

            evidence = get_candidate_evidence(
                doc,
                query,
                "title"
            )

            record["has_title_evidence"] = True

            if evidence["title_score"] > record["title_score"]:
                record["title_score"] = evidence["title_score"]
                record["title_query"] = query

            record["title_coverage"] = max(
                record["title_coverage"],
                evidence["title_coverage"]
            )

    # --------------------------------------------------------
    # AUTHOR SEARCHES
    # --------------------------------------------------------

    for query in author_queries:
        docs = search_open_library(
            session,
            query,
            "author"
        )

        time.sleep(REQUEST_DELAY)

        for doc in docs:
            key = doc.get("key")

            if not key:
                continue

            record = candidates.setdefault(
                key,
                {
                    "doc": doc,
                    "title_score": 0.0,
                    "title_coverage": 0.0,
                    "author_score": 0.0,
                    "author_coverage": 0.0,
                    "title_query": "",
                    "author_query": "",
                    "has_title_evidence": False,
                    "has_author_evidence": False
                }
            )

            evidence = get_candidate_evidence(
                doc,
                query,
                "author"
            )

            record["has_author_evidence"] = True

            if evidence["author_score"] > record["author_score"]:
                record["author_score"] = evidence["author_score"]
                record["author_query"] = query

            record["author_coverage"] = max(
                record["author_coverage"],
                evidence["author_coverage"]
            )

    # --------------------------------------------------------
    # FINAL SCORING
    # --------------------------------------------------------

    results = []

    for key, record in candidates.items():
        doc = record["doc"]

        title_score = record["title_score"]
        title_coverage = record["title_coverage"]
        author_score = record["author_score"]

        has_title = record["has_title_evidence"]
        has_author = record["has_author_evidence"]

        final_score = calculate_final_score(
            title_score,
            title_coverage,
            author_score,
            has_title
        )

        confidence = classify_candidate(
            final_score,
            title_score,
            title_coverage,
            has_title
        )

        if has_title and has_author:
            evidence_type = "title+author"
        elif has_title:
            evidence_type = "title"
        else:
            evidence_type = "author"

        result = {
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
            "matched_title_query": record["title_query"],
            "matched_author_query": record["author_query"],
            "evidence_type": evidence_type,
            "title_score": round(title_score, 4),
            "title_coverage": round(title_coverage, 4),
            "author_score": round(author_score, 4),
            "author_coverage": round(
                record["author_coverage"], 4
            ),
            "score": final_score,
            "confidence_label": confidence,
            "open_library_key": key
        }

        results.append(result)

    # Title-supported candidates always rank before author-only clues.
    results.sort(
        key=lambda item: (
            item["evidence_type"] != "author",
            item["score"],
            item["title_score"],
            item["title_coverage"],
            item["author_score"]
        ),
        reverse=True
    )

    return results[:5]


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nShelfSense AI - Book Identification V4.4")
    print("=" * 65)

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

        print("\n" + "-" * 65)
        print(f"[{index}/{len(crop_files)}] {image_name}")

        rows = ocr_data.get(image_name, [])

        if not rows:
            print("No OCR text. Skipping.")
            continue

        results = identify_one_book(
            session,
            image_name,
            rows
        )

        if not results:
            print("No catalogue candidates found.")
            continue

        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
            all_results.append(result)

            print(
                f"{rank}. {result['title']} | "
                f"{result['authors']} | "
                f"Score: {result['score']} | "
                f"Title score: {result['title_score']} | "
                f"Coverage: {result['title_coverage']} | "
                f"{result['confidence_label']}"
            )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    fieldnames = [
        "image",
        "rank",
        "title",
        "authors",
        "first_publish_year",
        "publishers",
        "isbn",
        "matched_title_query",
        "matched_author_query",
        "evidence_type",
        "title_score",
        "title_coverage",
        "author_score",
        "author_coverage",
        "score",
        "confidence_label",
        "open_library_key"
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
    print("V4.4 IDENTIFICATION COMPLETED")
    print("=" * 65)
    print("Images processed:", len(crop_files))
    print("Candidate records:", len(all_results))
    print("Saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
