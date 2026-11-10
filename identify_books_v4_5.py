import csv
import re
import time
import unicodedata
import requests

from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


# ============================================================
# SHELFSENSE AI - BOOK IDENTIFICATION V4.5
# Evidence-based identification using Open Library + Google Books
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

OUTPUT_CSV = OUTPUT_DIR / "identified_books_v4_5.csv"

OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

HEADERS = {
    "User-Agent": "ShelfSenseAI/4.5 (educational book-identification project)"
}

OPEN_LIBRARY_LIMIT = 30
GOOGLE_BOOKS_LIMIT = 20
REQUEST_DELAY = 0.25
MAX_RETRIES = 3

# Words that often indicate a title phrase. A two-word personal name
# alone is deliberately NOT considered a reliable title query.
TITLE_CUES = {
    "art", "love", "angst", "photography", "grotesque",
    "pictures", "picture", "tick", "early", "years",
    "weird", "wonderful", "fairy", "tales", "post",
    "mortem", "life", "history", "story", "stories",
    "illustrated", "paintings", "drawings", "catalogue",
    "catalog", "exhibition", "volume"
}

STOPWORDS = {
    "the", "a", "an", "and", "of", "by", "for", "with",
    "from", "in", "on", "to", "at", "edited", "written",
    "illustrated", "volume", "vol"
}

PUBLISHERS = {
    "dark horse", "darkhorse", "hirmer", "penguin",
    "lannoo", "harper collins", "random house",
    "yale university press", "dark horse books"
}

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
    "giuna": "giulia",
    "faited": "edited",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def correct_ocr(text):
    text = str(text or "").strip()
    compact = re.sub(r"[^a-z0-9]", "", text.lower())

    if compact in COMPACT_CORRECTIONS:
        return COMPACT_CORRECTIONS[compact]

    lower = text.lower()

    joined_replacements = {
        "picturesthattick": "pictures that tick",
        "americangrotesque": "american grotesque",
        "theartof": "the art of",
        "munchandphotography": "munch and photography",
        "postmortem": "post mortem",
    }

    for old, new in joined_replacements.items():
        lower = lower.replace(old, new)

    words = normalize(lower).split()
    return " ".join(
        WORD_CORRECTIONS.get(word, word)
        for word in words
    )


def meaningful_tokens(text):
    return [
        word for word in normalize(text).split()
        if word not in STOPWORDS and len(word) > 1
    ]


def title_similarity(query, candidate):
    """Compare a query with a catalogue title, preserving both precision and recall."""
    q_norm = normalize(query)
    c_norm = normalize(candidate)

    if not q_norm or not c_norm:
        return 0.0, 0.0, 0.0

    q_tokens = set(meaningful_tokens(q_norm))
    c_tokens = set(meaningful_tokens(c_norm))

    if not q_tokens or not c_tokens:
        return 0.0, 0.0, 0.0

    overlap = len(q_tokens & c_tokens)
    coverage = overlap / len(q_tokens)       # How much of OCR query is present?
    precision = overlap / len(c_tokens)      # How much of catalogue title is explained?

    sequence = SequenceMatcher(None, q_norm, c_norm).ratio()
    token_f1 = (
        2 * coverage * precision / (coverage + precision)
        if coverage + precision else 0.0
    )

    score = max(sequence, token_f1)
    return score, coverage, precision


def text_similarity(query, candidate):
    q_tokens = set(meaningful_tokens(query))
    c_tokens = set(meaningful_tokens(candidate))

    if not q_tokens or not c_tokens:
        return 0.0

    sequence = SequenceMatcher(
        None, normalize(query), normalize(candidate)
    ).ratio()

    overlap = len(q_tokens & c_tokens)
    coverage = overlap / len(q_tokens)
    precision = overlap / len(c_tokens)
    token_f1 = (
        2 * coverage * precision / (coverage + precision)
        if coverage + precision else 0.0
    )

    return max(sequence, token_f1)


# ============================================================
# OCR LOADING AND QUERY CLASSIFICATION
# ============================================================

def load_ocr():
    grouped = defaultdict(list)

    with open(OCR_CSV, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {"image", "detected_text", "method"}
        available = set(reader.fieldnames or [])

        if not required.issubset(available):
            raise ValueError(
                "OCR CSV must contain: image, method, detected_text"
            )

        for row in reader:
            image_name = Path(row.get("image", "")).name
            raw_text = (row.get("detected_text") or "").strip()

            if not image_name or not raw_text:
                continue

            grouped[image_name].append({
                "raw_text": raw_text,
                "corrected_text": correct_ocr(raw_text),
                "method": (row.get("method") or "").strip()
            })

    return grouped


def looks_like_title(text):
    """
    Conservative title filter:
    - A short personal name alone is not a title query.
    - Two-word phrases are accepted only when they contain a title cue.
    - Longer phrases need a title cue or at least four meaningful words.
    """
    cleaned = normalize(text)

    if not cleaned:
        return False

    if cleaned in PUBLISHERS:
        return False

    words = meaningful_tokens(cleaned)
    if len(words) < 2:
        return False

    if any(cue in words for cue in TITLE_CUES):
        return True

    # Avoid treating likely two/three-word names as titles.
    if len(words) <= 3:
        return False

    return len(words) >= 4


def clean_author_phrase(text):
    text = normalize(text)

    # Handles OCR mistakes such as "Faited by Giuna Bartrum".
    text = re.sub(r"^[a-z]+ by\s+", "", text)
    text = re.sub(r"^(edited|written|illustrated)\s+by\s+", "", text)

    return text.strip()


def looks_like_author(text):
    cleaned = normalize(text)

    if not cleaned or cleaned in PUBLISHERS:
        return False

    words = meaningful_tokens(cleaned)

    if not 1 <= len(words) <= 4:
        return False

    # Do not reuse an obvious title phrase as an author query.
    if any(cue in words for cue in TITLE_CUES):
        return False

    if len(cleaned.replace(" ", "")) < 4:
        return False

    return True


def split_multiple_authors(text):
    """
    Split 'David Fraser Jenkins and Hugh Fowler Wright' into two names,
    but do not split title phrases such as 'Weird and Wonderful'.
    """
    parts = re.split(r"\s+and\s+", normalize(text))

    if len(parts) == 2:
        left_words = meaningful_tokens(parts[0])
        right_words = meaningful_tokens(parts[1])

        if (
            len(left_words) >= 2
            and len(right_words) >= 2
            and not any(cue in left_words for cue in TITLE_CUES)
            and not any(cue in right_words for cue in TITLE_CUES)
        ):
            return [parts[0], parts[1]]

    return [normalize(text)]


def build_queries(ocr_rows):
    title_queries = []
    author_queries = []

    for row in ocr_rows:
        corrected = row["corrected_text"]
        cleaned = normalize(corrected)

        if not cleaned:
            continue

        # Keep complete title phrases intact. Never split every occurrence
        # of "and"; it may be part of the actual title.
        if looks_like_title(cleaned):
            title_queries.append(cleaned)

        author_text = clean_author_phrase(cleaned)

        for part in split_multiple_authors(author_text):
            part = clean_author_phrase(part)

            if looks_like_author(part):
                author_queries.append(part)

    title_queries = list(dict.fromkeys(title_queries))
    author_queries = list(dict.fromkeys(author_queries))

    return title_queries[:10], author_queries[:10]


# ============================================================
# API REQUESTS WITH RETRIES
# ============================================================

def request_json(session, url, params, service_name):
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=20
            )

            if response.status_code == 429 or response.status_code >= 500:
                last_error = (
                    f"HTTP {response.status_code} from {service_name}"
                )
                if attempt < MAX_RETRIES:
                    wait_seconds = 2 ** (attempt - 1)
                    print(
                        f"{service_name}: {last_error}; "
                        f"retrying in {wait_seconds}s "
                        f"({attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(wait_seconds)
                    continue

                return None, last_error

            response.raise_for_status()
            return response.json(), ""

        except requests.RequestException as error:
            last_error = str(error)

            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
            else:
                return None, last_error

        except ValueError as error:
            return None, f"Invalid JSON from {service_name}: {error}"

    return None, last_error or "Unknown API error"


def search_open_library(session, query, search_type):
    params = {
        "limit": OPEN_LIBRARY_LIMIT,
        "fields": (
            "key,title,author_name,first_publish_year,"
            "publisher,isbn,edition_count"
        )
    }

    if search_type == "title":
        params["title"] = query
    else:
        params["author"] = query

    data, error = request_json(
        session, OPEN_LIBRARY_URL, params, "Open Library"
    )

    if data is None:
        return [], "ERROR", error

    return data.get("docs", []), "OK", ""


def search_google_books(session, query, search_type):
    if search_type == "title":
        google_query = f'intitle:"{query}"'
    else:
        google_query = f'inauthor:"{query}"'

    params = {
        "q": google_query,
        "maxResults": GOOGLE_BOOKS_LIMIT,
        "printType": "books"
    }

    data, error = request_json(
        session, GOOGLE_BOOKS_URL, params, "Google Books"
    )

    if data is None:
        return [], "ERROR", error

    docs = []

    for item in data.get("items", []):
        info = item.get("volumeInfo", {}) or {}

        docs.append({
            "key": item.get("id", ""),
            "title": info.get("title", ""),
            "author_name": info.get("authors", []) or [],
            "first_publish_year": (
                str(info.get("publishedDate", ""))[:4]
            ),
            "publisher": (
                [info.get("publisher", "")]
                if info.get("publisher") else []
            ),
            "isbn": [
                identifier.get("identifier", "")
                for identifier in info.get("industryIdentifiers", [])
                if identifier.get("identifier")
            ],
            "source_url": info.get("infoLink", ""),
            "source": "Google Books"
        })

    return docs, "OK", ""


# ============================================================
# CANDIDATE EVIDENCE
# ============================================================

def candidate_id(source, doc):
    return f"{source}:{doc.get('key', '')}"


def new_record(doc, source):
    return {
        "doc": doc,
        "source": source,
        "title_evidence": None,
        "author_evidence": None
    }


def update_title_evidence(record, query):
    doc = record["doc"]
    score, coverage, precision = title_similarity(
        query, doc.get("title", "")
    )

    evidence = {
        "score": score,
        "coverage": coverage,
        "precision": precision,
        "query": query
    }

    current = record["title_evidence"]

    # Keep the best complete evidence tuple from ONE query.
    # Do not take max score and max coverage from different queries.
    if current is None or (
        score + coverage + precision
        > current["score"] + current["coverage"] + current["precision"]
    ):
        record["title_evidence"] = evidence


def update_author_evidence(record, query):
    authors = record["doc"].get("author_name", []) or []
    author_text = " ".join(authors)

    score = text_similarity(query, author_text)
    evidence = {
        "score": score,
        "query": query
    }

    current = record["author_evidence"]

    if current is None or score > current["score"]:
        record["author_evidence"] = evidence


def is_reliable_match(title_evidence, author_evidence):
    if not title_evidence:
        return False

    title_score = title_evidence["score"]
    coverage = title_evidence["coverage"]
    precision = title_evidence["precision"]

    # A strong title match is mandatory.
    if title_score < 0.82 or coverage < 0.80 or precision < 0.60:
        return False

    author_score = author_evidence["score"] if author_evidence else 0.0

    # Near-exact title can stand alone as a catalogue match.
    # Otherwise require independent author support.
    if title_score >= 0.96 and coverage >= 0.90 and precision >= 0.75:
        return True

    return author_score >= 0.60


def confidence_label(title_evidence, author_evidence, reliable):
    if not title_evidence:
        return "Author clue only"

    if not reliable:
        return "Low"

    title_score = title_evidence["score"]
    author_score = author_evidence["score"] if author_evidence else 0.0

    if (
        title_score >= 0.92
        and title_evidence["coverage"] >= 0.90
        and title_evidence["precision"] >= 0.75
        and author_score >= 0.60
    ):
        return "High"

    return "Medium"


# ============================================================
# IDENTIFY ONE BOOK
# ============================================================

def identify_one_book(session, image_name, ocr_rows):
    title_queries, author_queries = build_queries(ocr_rows)

    print("Title queries:", title_queries)
    print("Author queries:", author_queries)

    candidates = {}
    service_status = {
        "Open Library": "NOT SEARCHED",
        "Google Books": "NOT SEARCHED"
    }
    service_errors = []

    # Search title clues in both catalogue services.
    for query in title_queries:
        for source, search_function in (
            ("Open Library", search_open_library),
            ("Google Books", search_google_books)
        ):
            docs, status, error = search_function(
                session, query, "title"
            )

            service_status[source] = status

            if error:
                service_errors.append(f"{source}: {error}")

            for doc in docs:
                key = doc.get("key")
                if not key:
                    continue

                cid = candidate_id(source, doc)
                record = candidates.setdefault(
                    cid, new_record(doc, source)
                )
                update_title_evidence(record, query)

            time.sleep(REQUEST_DELAY)

    # Search author clues separately. These searches can support a title
    # candidate, but author-only records are never accepted.
    for query in author_queries:
        docs, status, error = search_open_library(
            session, query, "author"
        )
        service_status["Open Library"] = status

        if error:
            service_errors.append(f"Open Library: {error}")

        for doc in docs:
            key = doc.get("key")
            if not key:
                continue

            cid = candidate_id("Open Library", doc)
            record = candidates.setdefault(
                cid, new_record(doc, "Open Library")
            )
            update_author_evidence(record, query)

        time.sleep(REQUEST_DELAY)

        docs, status, error = search_google_books(
            session, query, "author"
        )
        service_status["Google Books"] = status

        if error:
            service_errors.append(f"Google Books: {error}")

        for doc in docs:
            key = doc.get("key")
            if not key:
                continue

            cid = candidate_id("Google Books", doc)
            record = candidates.setdefault(
                cid, new_record(doc, "Google Books")
            )
            update_author_evidence(record, query)

        time.sleep(REQUEST_DELAY)

    results = []

    for record in candidates.values():
        doc = record["doc"]
        title_ev = record["title_evidence"]
        author_ev = record["author_evidence"]

        reliable = is_reliable_match(title_ev, author_ev)
        confidence = confidence_label(
            title_ev, author_ev, reliable
        )

        title_score = title_ev["score"] if title_ev else 0.0
        title_coverage = title_ev["coverage"] if title_ev else 0.0
        title_precision = title_ev["precision"] if title_ev else 0.0
        author_score = author_ev["score"] if author_ev else 0.0

        # Ranking score is diagnostic only; acceptance uses the explicit
        # reliability rules above.
        combined_score = (
            0.75 * title_score + 0.25 * author_score
            if title_ev else 0.0
        )

        results.append({
            "image": image_name,
            "identification_status": (
                "Identified" if reliable else "Not Found"
            ),
            "candidate_decision": (
                "ACCEPTED" if reliable else "REJECTED"
            ),
            "source": record["source"],
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
            "matched_title_query": (
                title_ev["query"] if title_ev else ""
            ),
            "matched_author_query": (
                author_ev["query"] if author_ev else ""
            ),
            "title_score": round(title_score, 4),
            "title_coverage": round(title_coverage, 4),
            "title_precision": round(title_precision, 4),
            "author_score": round(author_score, 4),
            "score": round(combined_score, 4),
            "confidence_label": confidence,
            "source_key": doc.get("key", ""),
            "source_url": doc.get("source_url", ""),
            "open_library_key": (
                doc.get("key", "")
                if record["source"] == "Open Library" else ""
            ),
            "open_library_status": service_status["Open Library"],
            "google_books_status": service_status["Google Books"],
            "service_errors": " | ".join(dict.fromkeys(service_errors))
        })

    results.sort(
        key=lambda item: (
            item["candidate_decision"] == "ACCEPTED",
            item["score"],
            item["title_score"],
            item["author_score"]
        ),
        reverse=True
    )

    # If no reliable match exists, keep up to five rejected candidates
    # for debugging, but explicitly label the image Not Found.
    if results:
        accepted = any(
            item["candidate_decision"] == "ACCEPTED"
            for item in results
        )

        if not accepted:
            for item in results:
                item["identification_status"] = "Not Found"

        return results[:5], service_status, service_errors

    # No catalogue candidates at all.
    if not title_queries:
        reason = "No usable title clue in OCR"
    elif (
        service_status["Open Library"] == "ERROR"
        and service_status["Google Books"] == "ERROR"
    ):
        reason = "Both catalogue services unavailable; search incomplete"
    else:
        reason = "No catalogue candidates returned"

    empty_result = {
        "image": image_name,
        "identification_status": (
            "Search Error"
            if (
                service_status["Open Library"] == "ERROR"
                and service_status["Google Books"] == "ERROR"
            )
            else "Not Found"
        ),
        "candidate_decision": "NO CANDIDATE",
        "source": "",
        "title": "",
        "authors": "",
        "first_publish_year": "",
        "publishers": "",
        "isbn": "",
        "matched_title_query": "; ".join(title_queries),
        "matched_author_query": "; ".join(author_queries),
        "title_score": 0,
        "title_coverage": 0,
        "title_precision": 0,
        "author_score": 0,
        "score": 0,
        "confidence_label": "Not identified",
        "source_key": "",
        "source_url": "",
        "open_library_key": "",
        "open_library_status": service_status["Open Library"],
        "google_books_status": service_status["Google Books"],
        "service_errors": reason + (
            " | " + " | ".join(dict.fromkeys(service_errors))
            if service_errors else ""
        )
    }

    return [empty_result], service_status, service_errors


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nShelfSense AI - Book Identification V4.5")
    print("=" * 70)

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
    print("Catalogue services: Open Library + Google Books")
    print("Output:", OUTPUT_CSV)

    session = requests.Session()
    all_results = []

    for index, crop in enumerate(crop_files, start=1):
        image_name = crop.name

        print("\n" + "-" * 70)
        print(f"[{index}/{len(crop_files)}] {image_name}")

        rows = ocr_data.get(image_name, [])

        if not rows:
            print("No OCR text. Marking Not Found.")

            all_results.append({
                "image": image_name,
                "identification_status": "Not Found",
                "candidate_decision": "NO OCR",
                "source": "",
                "title": "",
                "authors": "",
                "first_publish_year": "",
                "publishers": "",
                "isbn": "",
                "matched_title_query": "",
                "matched_author_query": "",
                "title_score": 0,
                "title_coverage": 0,
                "title_precision": 0,
                "author_score": 0,
                "score": 0,
                "confidence_label": "Not identified",
                "source_key": "",
                "source_url": "",
                "open_library_key": "",
                "open_library_status": "NOT SEARCHED",
                "google_books_status": "NOT SEARCHED",
                "service_errors": "No OCR text"
            })
            continue

        results, statuses, errors = identify_one_book(
            session, image_name, rows
        )

        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
            all_results.append(result)

            if result["candidate_decision"] == "ACCEPTED":
                print(
                    f"{rank}. ACCEPTED: {result['title']} | "
                    f"{result['authors']} | {result['source']} | "
                    f"{result['confidence_label']} | "
                    f"score={result['score']}"
                )
            elif result["title"]:
                print(
                    f"{rank}. REJECTED: {result['title']} | "
                    f"{result['authors']} | "
                    f"title={result['title_score']} | "
                    f"author={result['author_score']}"
                )
            else:
                print(
                    f"Result: {result['identification_status']} | "
                    f"{result['service_errors']}"
                )

        accepted = any(
            result["candidate_decision"] == "ACCEPTED"
            for result in results
        )
        if not accepted:
            print("FINAL DECISION: NOT IDENTIFIED")

    fieldnames = [
        "image",
        "rank",
        "identification_status",
        "candidate_decision",
        "source",
        "title",
        "authors",
        "first_publish_year",
        "publishers",
        "isbn",
        "matched_title_query",
        "matched_author_query",
        "title_score",
        "title_coverage",
        "title_precision",
        "author_score",
        "score",
        "confidence_label",
        "source_key",
        "source_url",
        "open_library_key",
        "open_library_status",
        "google_books_status",
        "service_errors"
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

    identified_images = {
        row["image"]
        for row in all_results
        if row["candidate_decision"] == "ACCEPTED"
    }

    print("\n" + "=" * 70)
    print("V4.5 IDENTIFICATION COMPLETED")
    print("=" * 70)
    print("Book crops processed:", len(crop_files))
    print("Images identified:", len(identified_images))
    print("Images not identified:", len(crop_files) - len(identified_images))
    print("Candidate/output rows:", len(all_results))
    print("Saved to:", OUTPUT_CSV)
    print("\nIMPORTANT:")
    print("- Not Found means no reliable match was returned by the searches.")
    print("- Search Error means both catalogue services were unavailable.")
    print("- Rejected candidates are retained only for evaluation/debugging.")


if __name__ == "__main__":
    main()
