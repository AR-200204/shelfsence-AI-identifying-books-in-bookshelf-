"""
ShelfSense AI - Book Identification V4.6.1

This is a hotfix wrapper around identify_books_v4_6.py.

It does NOT modify V4.6.
It imports V4.6, applies targeted improvements, and writes separate output files:

    outputs/book_identification/identified_books_v4_6_1.csv
    outputs/book_identification/identified_books_v4_6_1_summary.csv
    outputs/book_identification/v4_6_1_run_report.json

Improvements over V4.6:
1. Better OCR correction for glued spine text.
2. Shorter title subqueries for long noisy titles.
3. Better publisher matching, e.g. darkhorse vs Dark Horse Books.
4. Optional Google Books API key support from .env.
5. Separate V4.6.1 output files so V4.6 results are preserved.
"""

import os
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


# ----------------------------------------------------------------------
# Load the original V4.6 script as a module.
# ----------------------------------------------------------------------
try:
    import identify_books_v4_6 as v46
except Exception as exc:
    raise SystemExit(
        "Could not import identify_books_v4_6.py.\n"
        "Make sure identify_books_v4_6.py is in the same folder as this file.\n"
        f"Import error: {exc}"
    )


# V4.6 may not import os. Inject it safely.
v46.os = os


# ----------------------------------------------------------------------
# V4.6.1 settings
# ----------------------------------------------------------------------

# Allow more title queries so long noisy OCR titles can generate shorter subqueries.
v46.MAX_TITLE_QUERIES = 6


def configure_v4_6_1_outputs():
    """
    Redirect V4.6 output paths to V4.6.1 files.

    We reuse the same API cache as V4.6 to avoid repeating successful
    Open Library requests unnecessarily.
    """
    v46.OUTPUT_CSV = v46.OUTPUT_DIR / "identified_books_v4_6_1.csv"
    v46.SUMMARY_CSV = v46.OUTPUT_DIR / "identified_books_v4_6_1_summary.csv"
    v46.REPORT_JSON = v46.OUTPUT_DIR / "v4_6_1_run_report.json"

    # Reuse V4.6 cache. If you want a completely separate cache, change this to:
    # v46.CACHE_FILE = v46.OUTPUT_DIR / "api_cache_v4_6_1.json"
    v46.CACHE_FILE = v46.OUTPUT_DIR / "api_cache_v4_6.json"


# ----------------------------------------------------------------------
# Simple .env loader
# ----------------------------------------------------------------------
def load_env_file(env_path=None):
    """
    Load simple KEY=VALUE pairs from .env.

    Does not overwrite variables already present in the environment.
    Never commits secrets. Keeps Google Books API key outside the code.
    """
    path = Path(env_path) if env_path else BASE_DIR / ".env"

    if not path.exists():
        return

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        print("Warning: could not read .env file:", error)
        return

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            os.environ.setdefault(key, value)


# ----------------------------------------------------------------------
# Helper: compact text for fuzzy publisher/title comparison
# ----------------------------------------------------------------------
def compact_text(text):
    """
    Return lowercase alphanumeric-only text.

    Examples:
        Dark Horse Books -> darkhorsebooks
        darkhorse        -> darkhorse
        Hirmer Verlag    -> hirmerverlag
    """
    return re.sub(r"[^a-z0-9]", "", v46.normalize(text))


# ----------------------------------------------------------------------
# Hotfix 1: better OCR correction
# ----------------------------------------------------------------------
ORIGINAL_CORRECT_OCR = v46.correct_ocr

EXTRA_OCR_REPLACEMENTS = {
    # Common glued / misread spine fragments.
    "theartor": "the art of",
    "theartorjohnpiper": "the art of john piper",
    "theartofjohnpiper": "the art of john piper",

    "picturesthattick": "pictures that tick",
    "americangrotesque": "american grotesque",
    "munchandphotography": "munch and photography",
    "postmortem": "post mortem",

    "samuelpalmer": "samuel palmer",
    "arthurrackham": "arthur rackham",
    "edvardmunch": "edvard munch",
    "ottomueller": "otto mueller",
    "williammorris": "william morris",
    "williammortensen": "william mortensen",
    "davidmckean": "dave mckean",
}


def patched_correct_ocr(text):
    """
    Improve OCR correction before calling the original V4.6 correct_ocr().
    """
    s = str(text or "")
    lower = s.lower()

    for old, new in EXTRA_OCR_REPLACEMENTS.items():
        lower = lower.replace(old, new)

    return ORIGINAL_CORRECT_OCR(lower)


v46.correct_ocr = patched_correct_ocr


# ----------------------------------------------------------------------
# Hotfix 2: better publisher matching
# ----------------------------------------------------------------------
def patched_publisher_score(ocr_publishers, cand_publishers):
    """
    Improved publisher evidence.

    Important rule:
        Publisher can support a candidate, but can never prove the title.

    Handles compact publisher names such as:
        darkhorse       vs Dark Horse Books
        hirmer          vs Hirmer Verlag
        yaleuniversitypress vs Yale University Press
    """
    if not ocr_publishers or not cand_publishers:
        return None

    best = 0.0

    for ocr_pub in ocr_publishers:
        o_norm = v46.normalize(ocr_pub)
        o_tokens = [t for t in o_norm.split() if t]
        o_compact = compact_text(o_norm)

        if not o_tokens and not o_compact:
            continue

        for cand_pub in cand_publishers:
            c_norm = v46.normalize(cand_pub)
            c_tokens = [t for t in c_norm.split() if t]
            c_compact = compact_text(c_norm)

            score = 0.0

            # Compact comparison survives missing spaces and punctuation.
            if o_compact and c_compact:
                if o_compact == c_compact:
                    score = 1.0
                elif o_compact in c_compact or c_compact in o_compact:
                    score = 0.95
                else:
                    score = max(
                        score,
                        SequenceMatcher(None, o_compact, c_compact).ratio()
                    )

            # Token comparison still helps for normal multi-word publishers.
            if o_tokens and c_tokens:
                score = max(
                    score,
                    v46.match_weight(o_tokens, c_tokens) / len(o_tokens)
                )

            best = max(best, score)

    return best if best > 0 else 0.0


v46.publisher_score = patched_publisher_score


# ----------------------------------------------------------------------
# Hotfix 3: generate shorter title subqueries for long noisy OCR titles
# ----------------------------------------------------------------------
def query_rarity_score(tokens):
    """
    Rank generated subqueries.

    Prefer subqueries containing longer / rarer / title-like words.
    This is only for choosing which query variants to search.
    """
    score = 0.0

    for t in tokens:
        if t in v46.STOPWORDS:
            continue

        if len(t) >= 8:
            score += 2.0
        elif len(t) >= 5:
            score += 1.0

        if t in v46.TITLE_CUES:
            score += 0.5

    return score


def generate_subqueries(query, max_meaningful_span=5):
    """
    Create shorter contiguous title spans from a long OCR title query.

    Example:
        weird and wonderful mothmeister post mortem fairy tales

    Can produce useful shorter queries such as:
        mothmeister post mortem fairy tales
        weird and wonderful mothmeister
        post mortem fairy tales
    """
    words = query["text"].split()

    if len(words) < 4:
        return []

    meaningful_idx = [
        i for i, w in enumerate(words)
        if w not in v46.STOPWORDS and len(w) > 1
    ]

    if len(meaningful_idx) < 3:
        return []

    out = []
    original_tokens = tuple(query["tokens"])
    max_size = min(max_meaningful_span, len(meaningful_idx))

    # Try longer useful spans first, then shorter ones.
    for size in range(max_size, 1, -1):
        for start in range(0, len(meaningful_idx) - size + 1):
            first = meaningful_idx[start]
            last = meaningful_idx[start + size - 1]

            sub_text = " ".join(words[first:last + 1])
            sub_tokens = tuple(v46.meaningful_tokens(sub_text))

            if len(sub_tokens) < 2:
                continue

            if sub_tokens == original_tokens:
                continue

            out.append(v46.make_query(sub_text, "subquery"))

    return out


def expand_title_queries(unique_queries, limit):
    """
    Keep original best queries first, then add shorter subqueries if needed.
    """
    base = list(unique_queries[:limit])

    subs = []
    for q in unique_queries:
        subs.extend(generate_subqueries(q))

    def is_duplicate(q, existing):
        compact = q["text"].replace(" ", "")
        for u in existing:
            if SequenceMatcher(
                None,
                compact,
                u["text"].replace(" ", "")
            ).ratio() >= 0.92:
                return True
        return False

    # Remove duplicate subqueries among themselves.
    seen_subs = []
    deduped_subs = []

    subs.sort(
        key=lambda q: (
            len(q["tokens"]),
            query_rarity_score(q["tokens"]),
            -len(q["text"]),
        ),
        reverse=True
    )

    for q in subs:
        if not is_duplicate(q, seen_subs):
            seen_subs.append(q)
            deduped_subs.append(q)

    # Add best subqueries only if we still have room.
    for q in deduped_subs:
        if len(base) >= limit:
            break

        if not is_duplicate(q, base):
            base.append(q)

    return base[:limit]


ORIGINAL_BUILD_CLUES = v46.build_clues


def patched_build_clues(ocr_rows):
    """
    Use original V4.6 clue building, then expand title queries with subqueries.
    """
    clues = ORIGINAL_BUILD_CLUES(ocr_rows)
    clues["title_queries"] = expand_title_queries(
        clues["title_queries"],
        v46.MAX_TITLE_QUERIES
    )
    return clues


v46.build_clues = patched_build_clues


# ----------------------------------------------------------------------
# Hotfix 4: optional Google Books API key support
# ----------------------------------------------------------------------
try:
    ORIGINAL_BUILD_REQUEST_FUNC = v46.CatalogueClient.__dict__["_build_request"].__func__
except AttributeError:
    ORIGINAL_BUILD_REQUEST_FUNC = v46.CatalogueClient._build_request


def patched_build_request(service, title, author):
    """
    Add optional GOOGLE_BOOKS_API_KEY from environment / .env.

    This does not bypass rate limits.
    It only allows Google to recognise your project if you provide a key.
    """
    url, params = ORIGINAL_BUILD_REQUEST_FUNC(service, title, author)

    if service == v46.GB:
        params = dict(params)
        api_key = os.getenv("GOOGLE_BOOKS_API_KEY", "").strip()

        if api_key:
            params["key"] = api_key

    return url, params


v46.CatalogueClient._build_request = staticmethod(patched_build_request)


# ----------------------------------------------------------------------
# Main wrapper
# ----------------------------------------------------------------------
def main():
    load_env_file()
    configure_v4_6_1_outputs()

    print("\nShelfSense AI - Book Identification V4.6.1")
    print("=" * 70)
    print("Base module :", getattr(v46, "__file__", "identify_books_v4_6.py"))
    print("Output CSV  :", v46.OUTPUT_CSV)
    print("Summary CSV :", v46.SUMMARY_CSV)
    print("Report JSON :", v46.REPORT_JSON)
    print("Cache file  :", v46.CACHE_FILE)
    print("Google API key:", "detected" if os.getenv("GOOGLE_BOOKS_API_KEY") else "not set")
    print("Note: the V4.6 base script may still print its own V4.6 header.")
    print("V4.6.1 outputs are written to separate files.")

    # Run the original V4.6 main(), but with patched functions and output paths.
    v46.main()


if __name__ == "__main__":
    main()