import argparse
import csv
import json
import re
import time
import unicodedata
import requests

from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher


# ============================================================
# SHELFSENSE AI - BOOK IDENTIFICATION V4.6
#
# What is new compared with V4.5:
#   1. One final status per book: IDENTIFIED / AMBIGUOUS / NOT IDENTIFIED
#   2. Only ONE candidate can be SELECTED per book. Everything else is
#      ALTERNATIVE - REVIEW or REJECTED.
#   3. OCR text is split into title / author / publisher / year pieces.
#      Author-only and publisher-only text can never identify a book.
#   4. Candidates that are records of the same work are grouped, so
#      different editions are not treated as different books and
#      different books with the same title are detected.
#   5. Better partial-author matching (surname, initials, OCR typos).
#   6. Google Books HTTP 429 => stop Google for the rest of the run.
#   7. Response cache, so re-running does not repeat the same requests.
#   8. Optional evaluation against a manually verified ground-truth CSV.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OCR_CSV = (
    BASE_DIR / "outputs" / "ocr_results_paddle_v2"
    / "paddleocr_v2_comparison.csv"
)
CROPS_DIR = BASE_DIR / "outputs" / "cropped_books"

OUTPUT_DIR = BASE_DIR / "outputs" / "book_identification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "identified_books_v4_6.csv"           # one row per candidate
SUMMARY_CSV = OUTPUT_DIR / "identified_books_v4_6_summary.csv"  # one row per book
REPORT_JSON = OUTPUT_DIR / "v4_6_run_report.json"
CACHE_FILE = OUTPUT_DIR / "api_cache_v4_6.json"

EVAL_DIR = BASE_DIR / "outputs" / "evaluation"
GROUND_TRUTH_CSV = EVAL_DIR / "ground_truth.csv"
GROUND_TRUTH_TEMPLATE = EVAL_DIR / "ground_truth_template.csv"

OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
OL = "Open Library"
GB = "Google Books"

HEADERS = {
    "User-Agent": "ShelfSenseAI/4.6 (educational book-identification project)"
}

OPEN_LIBRARY_LIMIT = 30
GOOGLE_BOOKS_LIMIT = 20
REQUEST_DELAY = 0.35          # pause after every real (non-cached) request
MAX_RETRIES = 3               # only for network errors / HTTP 5xx
MAX_CONSECUTIVE_ERRORS = 3    # then the service is switched off for the run

MAX_TITLE_QUERIES = 3         # title queries per book
MAX_AUTHOR_COMBO_QUERIES = 2  # title+author searches per book

# ---- decision thresholds (explicit, easy to tune) ----
TITLE_SCORE_MIN = 0.85        # overall title similarity
TITLE_COVERAGE_MIN = 0.85     # share of OCR title words found in the record
TITLE_PRECISION_MIN = 0.60    # share of record title words explained by OCR
AUTHOR_SUPPORT_MIN = 0.60     # author score that counts as real support
STANDALONE_MIN_TOKENS = 3     # a title this long may be accepted without author
STANDALONE_MIN_SCORE = 0.95   # ...if it is a near-exact match
COMPETITOR_MARGIN = 0.10      # another work this close = ambiguity

# False  : IDENTIFIED means "same title + author" (the work). The edition is
#          reported separately in `edition_confirmed`.
# True   : if the edition cannot be confirmed (no publisher/year/ISBN match)
#          and the catalogue has several editions, return AMBIGUOUS.
STRICT_EDITION_MODE = False

MAX_ALTERNATIVES_PER_BOOK = 8
MAX_REJECTED_PER_BOOK = 4


# ============================================================
# WORD LISTS
# ============================================================

# Words that make a short phrase look like a title instead of a person's name.
TITLE_CUES = {
    "art", "love", "angst", "photography", "grotesque",
    "pictures", "picture", "tick", "early", "years",
    "weird", "wonderful", "fairy", "tales", "post",
    "mortem", "life", "history", "story", "stories",
    "illustrated", "paintings", "drawings", "catalogue",
    "catalog", "exhibition", "volume",
    # generic additions
    "book", "books", "guide", "introduction", "world", "night",
    "dark", "light", "complete", "collected", "works", "letters",
    "essays", "poems", "poetry", "selected", "portraits", "portrait",
    "modern", "design", "theory", "science", "physics", "mathematics",
    "quantum", "handbook", "principles", "fundamentals", "analysis",
    "engineering", "cookbook", "atlas", "dictionary", "encyclopedia",
    "war", "peace", "death", "dreams", "secret", "secrets",
}

STOPWORDS = {
    "the", "a", "an", "and", "of", "by", "for", "with",
    "from", "in", "on", "to", "at", "edited", "written",
    "illustrated", "volume", "vol"
}

PUBLISHERS = {
    "dark horse", "darkhorse", "hirmer", "penguin",
    "lannoo", "harper collins", "harpercollins", "random house",
    "yale university press", "dark horse books",
    "thames and hudson", "thames hudson", "taschen", "phaidon",
    "prestel", "rizzoli", "abrams", "harry n abrams", "skira",
    "electa", "steidl", "gestalten", "hatje cantz", "tate publishing",
    "faber", "bloomsbury", "image comics", "dc comics", "marvel",
    "mit press", "springer", "wiley", "pearson", "oreilly", "o reilly",
}

# A short segment ending in one of these words is treated as a publisher.
PUBLISHER_ENDINGS = {
    "press", "publishing", "publishers", "publisher", "verlag",
    "editions", "edition", "books", "comics",
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

# Small vocabulary used ONLY to split joined words such as "ARTOFLIFE".
# A split is accepted only when every letter is covered by known words.
SPLIT_VOCAB = set(STOPWORDS) | set(TITLE_CUES) | {
    "that", "this", "new", "old", "great", "little", "big", "man", "men",
    "woman", "women", "time", "day", "year", "late", "red", "black", "white",
    "blue", "green", "golden", "lost", "found", "work", "essay", "painting",
    "drawing", "photographs", "landscape", "nature", "graphic", "comic",
    "comics", "fairy", "tale", "tales", "weird", "wonderful", "post",
    "mortem", "american", "british", "english", "european", "french",
    "german", "years", "early", "pictures", "tick", "city", "house",
    "home", "king", "queen", "sea", "sky", "fire", "water", "earth",
}

AUTHOR_NOISE_TOKENS = {
    "edited", "editor", "editors", "various", "unknown", "anonymous",
    "et", "al", "and", "by", "with", "the", "of", "introduction",
}

SMALL_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
ROMAN_NUMBERS = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
    "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
}
VOLUME_RE = re.compile(
    r"\b(?:vol|volume|part|book|band|tome|no|number)\s*"
    r"(\d{1,2}|[ivx]{1,4}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_joined_token(token):
    """Split a long glued token using SPLIT_VOCAB. Full coverage or nothing."""
    if len(token) < 9 or token in SPLIT_VOCAB:
        return [token]

    n = len(token)
    best = [None] * (n + 1)   # best[i] = fewest words covering token[:i]
    best[0] = []

    for i in range(1, n + 1):
        for j in range(max(0, i - 12), i):
            word = token[j:i]
            if best[j] is not None and word in SPLIT_VOCAB and len(word) >= 2:
                candidate = best[j] + [word]
                if best[i] is None or len(candidate) < len(best[i]):
                    best[i] = candidate

    if best[n] and len(best[n]) >= 2:
        return best[n]
    return [token]


def correct_ocr(text):
    """Normalize OCR text and fix known / glued-word errors."""
    text = str(text or "").strip()
    compact = re.sub(r"[^a-z0-9]", "", text.lower())

    if compact in COMPACT_CORRECTIONS:
        return COMPACT_CORRECTIONS[compact]

    lower = text.lower()
    for old, new in {
        "picturesthattick": "pictures that tick",
        "americangrotesque": "american grotesque",
        "theartof": "the art of",
        "munchandphotography": "munch and photography",
        "postmortem": "post mortem",
    }.items():
        lower = lower.replace(old, new)

    words = []
    for word in normalize(lower).split():
        word = WORD_CORRECTIONS.get(word, word)
        words.extend(split_joined_token(word))
    return " ".join(words)


def meaningful_tokens(text):
    return [
        w for w in normalize(text).split()
        if w not in STOPWORDS and len(w) > 1
    ]


def token_weight(a, b):
    """How well do two single words match? 1.0 exact ... 0.0 no match."""
    if a == b:
        return 1.0
    if a.isdigit() or b.isdigit():
        return 0.0

    la, lb = len(a), len(b)
    if min(la, lb) == 1:                     # initial, e.g. "w" vs "william"
        return 0.5 if a[0] == b[0] else 0.0
    if la == lb and la >= 5:                 # one wrong character (OCR)
        if sum(x != y for x, y in zip(a, b)) == 1:
            return 0.9
    if min(la, lb) >= 6 and SequenceMatcher(None, a, b).ratio() >= 0.88:
        return 0.85
    return 0.0


def token_in(token, collection):
    return any(token_weight(token, other) >= 0.85 for other in collection)


def match_weight(q_tokens, c_tokens):
    """Greedy one-to-one matching of query words to candidate words."""
    used = set()
    total = 0.0
    for q in q_tokens:
        best_w, best_i = 0.0, None
        for i, c in enumerate(c_tokens):
            if i in used:
                continue
            w = token_weight(q, c)
            if w > best_w:
                best_w, best_i = w, i
        if best_i is not None and best_w >= 0.85:
            used.add(best_i)
            total += best_w
    return total


def token_title_score(q_tokens, c_tokens):
    """Returns (score, coverage, precision) for two token lists."""
    if not q_tokens or not c_tokens:
        return 0.0, 0.0, 0.0

    weight = match_weight(q_tokens, c_tokens)
    coverage = weight / len(q_tokens)     # how much of the OCR is explained
    precision = weight / len(c_tokens)    # how much of the record is explained
    f1 = (
        2 * coverage * precision / (coverage + precision)
        if coverage + precision else 0.0
    )

    q_text, c_text = " ".join(q_tokens), " ".join(c_tokens)
    seq = SequenceMatcher(None, q_text, c_text).ratio()
    # Spaceless comparison survives glued / split words from OCR.
    compact = SequenceMatcher(
        None, q_text.replace(" ", ""), c_text.replace(" ", "")
    ).ratio()

    return max(seq, compact, f1), coverage, precision


def parse_volume(text):
    """Returns (volume_number_or_None, text_without_volume_marker)."""
    norm = normalize(text)
    match = VOLUME_RE.search(norm)

    if match:
        raw = match.group(1)
        if raw.isdigit():
            value = int(raw)
        else:
            value = SMALL_NUMBER_WORDS.get(raw) or ROMAN_NUMBERS.get(raw)
        rest = (norm[:match.start()] + " " + norm[match.end():]).strip()
        return value, re.sub(r"\s+", " ", rest)

    trailing = re.search(r"\s(\d{1,2})$", norm)
    if trailing and len(norm.split()) >= 2:
        return int(trailing.group(1)), norm[:trailing.start()].strip()

    return None, norm


def title_key(main_title):
    """Key used to decide whether two records are the same work."""
    _, base = parse_volume(main_title)
    base = re.sub(r"^(the|a|an)\s+", "", base)
    return base


# ============================================================
# AUTHOR MATCHING
# ============================================================

def flip_last_first(name):
    """'Picasso, Pablo' -> 'Pablo Picasso'."""
    name = str(name or "").strip()
    if name.count(",") == 1:
        last, first = [p.strip() for p in name.split(",")]
        if last and first:
            return f"{first} {last}"
    return name


def author_token_set(authors):
    tokens = set()
    for name in authors:
        for t in normalize(name).split():
            if len(t) >= 3 and t not in AUTHOR_NOISE_TOKENS:
                tokens.add(t)
    return tokens


def author_similarity(clue, authors):
    """
    Compare one OCR author clue with a list of catalogue authors.
    Handles: surname only, partial names, initials, OCR typos,
    several authors (best author wins).
    A surname-only clue is capped at 0.80: it can support, never prove.
    """
    clue_tokens = [
        t for t in normalize(clue).split()
        if t not in AUTHOR_NOISE_TOKENS
    ]
    if not clue_tokens:
        return 0.0

    best = 0.0
    for author in authors:
        a_tokens = [
            t for t in normalize(author).split()
            if t not in AUTHOR_NOISE_TOKENS
        ]
        if not a_tokens:
            continue

        weights = [
            max(token_weight(ct, at) for at in a_tokens)
            for ct in clue_tokens
        ]
        clue_cov = sum(w if w >= 0.5 else 0.0 for w in weights) / len(clue_tokens)

        matched_author_tokens = sum(
            1 for at in a_tokens
            if any(token_weight(ct, at) >= 0.5 for ct in clue_tokens)
        )
        author_cov = matched_author_tokens / len(a_tokens)

        score = 0.85 * clue_cov + 0.15 * author_cov
        if len(clue_tokens) == 1:
            score = min(score, 0.80)
        best = max(best, score)

    return best


def author_in_text(authors, token_pool):
    """
    Does a catalogue author name appear anywhere in the OCR text?
      all words of a 2+ word name found -> 0.95
      only the surname found           -> 0.70 (support only)
    """
    best = 0.0
    for author in authors:
        tokens = [
            t for t in normalize(author).split()
            if len(t) >= 2 and t not in AUTHOR_NOISE_TOKENS
        ]
        if not tokens:
            continue

        found = [token_in(t, token_pool) for t in tokens]
        if len(tokens) >= 2 and all(found):
            best = max(best, 0.95)
        elif len(tokens[-1]) >= 4 and found[-1]:
            best = max(best, 0.70)
    return best


# ============================================================
# OCR LOADING AND SEGMENT CLASSIFICATION
# ============================================================

def load_ocr():
    grouped = defaultdict(list)

    with open(OCR_CSV, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {"image", "detected_text", "method"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("OCR CSV must contain: image, method, detected_text")

        for row in reader:
            image_name = Path(row.get("image", "")).name
            raw_text = (row.get("detected_text") or "").strip()
            if not image_name or not raw_text:
                continue

            try:
                confidence = float(row.get("confidence") or 0.0)
            except ValueError:
                confidence = 0.0

            grouped[image_name].append({
                "raw_text": raw_text,
                "method": (row.get("method") or "").strip(),
                "confidence": confidence,
            })

    return grouped


def split_segments(raw_text):
    """
    PaddleOCR text pieces are separated by ';' in the CSV. Split on those,
    on '|', on new lines and on spaced dashes (' - '), but NOT on hyphens
    inside words (Post-Mortem) or date ranges (1892-1906).
    """
    parts = re.split(r"\s*[;|\n]+\s*|\s+[-\u2013\u2014]\s+", raw_text)
    return [p.strip() for p in parts if p.strip()]


def is_publisher_text(norm_text):
    if not norm_text:
        return False
    if norm_text in PUBLISHERS:
        return True
    words = norm_text.split()
    return 1 <= len(words) <= 4 and words[-1] in PUBLISHER_ENDINGS


CREDIT_WORDS = r"(edited|written|illustrated|photographed|translated|compiled)"


def clean_author_phrase(text):
    """Remove 'edited by' style prefixes (OCR fixes happen in correct_ocr)."""
    text = normalize(text)
    text = re.sub(r"^" + CREDIT_WORDS + r"\s+by\s+", "", text)
    return text.strip()


def split_multiple_authors(text):
    """'A B and C D' -> two names, but never split 'Weird and Wonderful'."""
    parts = re.split(r"\s+and\s+", normalize(text))
    if len(parts) == 2:
        left, right = meaningful_tokens(parts[0]), meaningful_tokens(parts[1])
        if (
            len(left) >= 2 and len(right) >= 2
            and not any(c in left for c in TITLE_CUES)
            and not any(c in right for c in TITLE_CUES)
        ):
            return [parts[0], parts[1]]
    return [normalize(text)]


def classify_segment(raw_segment):
    """Decide what one OCR piece is: publisher / year / isbn / author / title."""
    corrected = correct_ocr(raw_segment)
    info = {"raw": raw_segment, "text": corrected, "kind": "junk",
            "tokens": [], "authors": []}

    digits_only = re.sub(r"[^0-9xX]", "", raw_segment)
    if (
        len(digits_only) in (10, 13)
        and sum(ch.isdigit() for ch in raw_segment) >= 10
        and not re.search(r"[a-wyzA-WYZ]", raw_segment)
    ):
        info["kind"], info["isbn"] = "isbn", digits_only.upper()
        return info

    if re.fullmatch(r"(1[5-9]\d\d|20\d\d)", corrected):
        info["kind"], info["year"] = "year", corrected
        return info

    if is_publisher_text(corrected):
        info["kind"] = "publisher"
        return info

    if re.fullmatch(VOLUME_RE.pattern, normalize(corrected)):
        info["kind"] = "volume"
        return info

    tokens = meaningful_tokens(corrected)
    info["tokens"] = tokens
    if not tokens:
        return info

    # "edited by X" style: author even if the rest looks odd.
    if re.match(r"^" + CREDIT_WORDS + r" by\s+", corrected):
        names = [
            n for part in split_multiple_authors(clean_author_phrase(corrected))
            if (n := clean_author_phrase(part))
        ]
        if names:
            info["kind"], info["authors"] = "author", names
            return info

    has_cue = any(t in TITLE_CUES for t in tokens)
    has_digit = any(t.isdigit() for t in tokens)

    # Two clear person names joined by "and"?
    multi = split_multiple_authors(corrected)
    if len(multi) == 2 and not has_digit:
        info["kind"], info["authors"] = "author", multi
        return info

    # Short, cue-free, digit-free phrase = most likely a person's name.
    if 1 <= len(tokens) <= 4 and not has_cue and not has_digit:
        info["kind"], info["authors"] = "author", [corrected]
        return info

    # Title-like: >=2 meaningful words and (a title cue or >=4 words).
    if len(tokens) >= 2 and (has_cue or len(tokens) >= 4):
        info["kind"] = "title"
        return info

    return info


def make_query(text, kind):
    return {
        "text": normalize(text),
        "kind": kind,
        "tokens": meaningful_tokens(text),
    }


def strip_numbers(text):
    return " ".join(w for w in normalize(text).split() if not w.isdigit())


def build_clues(ocr_rows):
    """Turn all OCR rows of one crop into separated, typed clues."""
    rows = sorted(ocr_rows, key=lambda r: -r["confidence"])

    publishers, authors, years, isbns = [], [], [], []
    token_pool = set()
    queries = []
    ocr_volume = None

    for row in rows:
        infos = [classify_segment(s) for s in split_segments(row["raw_text"])]

        for info in infos:
            if info["kind"] == "publisher":
                publishers.append(info["text"])
            elif info["kind"] == "year":
                years.append(info["year"])
            elif info["kind"] == "isbn":
                isbns.append(info["isbn"])
            elif info["kind"] == "author":
                authors.extend(info["authors"])

            if info["kind"] in ("title", "author"):
                token_pool.update(normalize(info["text"]).split())

            # Volume markers such as "VOL 2" have no title words, so look at
            # every segment, whatever kind it was classified as.
            if ocr_volume is None and VOLUME_RE.search(normalize(info["text"])):
                ocr_volume, _ = parse_volume(info["text"])

        title_infos = [i for i in infos if i["kind"] == "title"]
        body_infos = [i for i in infos if i["kind"] in ("title", "author")]

        # A book needs at least one title-like piece before we search.
        if not title_infos:
            continue

        row_queries = [make_query(i["text"], "segment") for i in title_infos]

        if len(title_infos) > 1:
            row_queries.append(make_query(
                " ".join(i["text"] for i in title_infos), "segments_joined"))

        joined = " ".join(i["text"] for i in body_infos)
        if len(meaningful_tokens(joined)) >= 3:
            row_queries.append(make_query(joined, "joined"))

        # Variants without years (a record may not carry "1892-1906").
        extra = []
        for q in row_queries:
            if any(t.isdigit() and len(t) == 4 for t in q["tokens"]):
                stripped = strip_numbers(q["text"])
                if len(meaningful_tokens(stripped)) >= 2:
                    extra.append(make_query(stripped, q["kind"]))
        row_queries.extend(extra)

        queries.extend(row_queries)

    # De-duplicate near-identical queries (OCR variants), keep best-first order.
    unique = []
    for q in queries:
        if len(q["tokens"]) < 2:
            continue
        compact = q["text"].replace(" ", "")
        if any(
            SequenceMatcher(None, compact, u["text"].replace(" ", "")).ratio() >= 0.92
            for u in unique
        ):
            continue
        unique.append(q)

    author_clues = list(dict.fromkeys(
        a for a in (clean_author_phrase(x) for x in authors) if len(a) >= 3
    ))

    return {
        "title_queries": unique[:MAX_TITLE_QUERIES],
        "author_clues": author_clues[:4],
        "publisher_clues": list(dict.fromkeys(publishers))[:3],
        "years": list(dict.fromkeys(years))[:2],
        "isbns": list(dict.fromkeys(isbns))[:2],
        "ocr_volume": ocr_volume,
        "token_pool": token_pool,
        "best_ocr_text": rows[0]["raw_text"] if rows else "",
    }


# ============================================================
# CATALOGUE CANDIDATES (common format for both services)
# ============================================================

def make_candidate(source, key, title, subtitle, authors, year,
                   publishers, isbns, url="", edition_count=""):
    title = str(title or "").strip()
    subtitle = str(subtitle or "").strip()

    if subtitle:
        main, full = title, f"{title}: {subtitle}"
    elif ":" in title:
        main, full = title.split(":")[0].strip(), title
    else:
        main, full = title, title

    clean_authors = [
        flip_last_first(a) for a in (authors or []) if str(a).strip()
    ]

    return {
        "id": f"{source}:{key}",
        "source": source,
        "key": key,
        "title": title,
        "subtitle": subtitle,
        "title_main": main,
        "title_full": full,
        "authors": clean_authors,
        "year": str(year or "")[:4],
        "publishers": list(dict.fromkeys(p for p in (publishers or []) if p))[:8],
        "isbns": list(dict.fromkeys(i for i in (isbns or []) if i))[:10],
        "url": url,
        "edition_count": edition_count,
    }


def parse_open_library(data):
    docs = []
    for doc in data.get("docs", []):
        key = doc.get("key")
        if not key:
            continue
        docs.append(make_candidate(
            OL, key, doc.get("title"), doc.get("subtitle"),
            doc.get("author_name"), doc.get("first_publish_year"),
            doc.get("publisher"), doc.get("isbn"),
            url="https://openlibrary.org" + key,
            edition_count=doc.get("edition_count", ""),
        ))
    return docs


def parse_google_books(data):
    docs = []
    for item in data.get("items", []) or []:
        info = item.get("volumeInfo", {}) or {}
        key = item.get("id")
        if not key:
            continue
        docs.append(make_candidate(
            GB, key, info.get("title"), info.get("subtitle"),
            info.get("authors"), str(info.get("publishedDate", ""))[:4],
            [info["publisher"]] if info.get("publisher") else [],
            [i.get("identifier", "") for i in info.get("industryIdentifiers", [])],
            url=info.get("infoLink", ""),
        ))
    return docs


# ============================================================
# API CLIENT (rate limits, errors, cache)
# ============================================================

class CatalogueClient:
    """
    Outcomes returned for every search:
      OK            real request succeeded
      CACHED        answer came from the cache (no request)
      RATE_LIMITED  HTTP 429 happened (now or earlier in the run)
      ERROR         network / server failure (now or service switched off)
      SKIPPED       service disabled by the user (--skip-google)
    A failure is never interpreted as "the book does not exist".
    """

    def __init__(self, use_cache=True, skip_google=False):
        self.session = requests.Session()
        self.use_cache = use_cache
        self.cache = self._load_cache() if use_cache else {}
        self.disabled = {OL: "", GB: "USER" if skip_google else ""}
        self.consecutive_errors = {OL: 0, GB: 0}
        self.stats = {
            svc: Counter() for svc in (OL, GB)
        }

    # ---- cache ----
    @staticmethod
    def _load_cache():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save_cache(self):
        if not self.use_cache:
            return
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except OSError as error:
            print("Warning: could not save cache:", error)

    # ---- one HTTP request with limited retries ----
    def _http_get(self, service, url, params):
        """Returns (json_or_None, outcome, error_text, retry_after_seconds)."""
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    url, params=params, headers=HEADERS, timeout=20
                )
            except requests.RequestException as error:      # e.g. WinError 10054
                last_error = str(error)
                self.stats[service]["retries"] += 1
                if attempt < MAX_RETRIES:
                    time.sleep(attempt)                      # 1 s, 2 s
                continue

            if response.status_code == 429:
                try:
                    retry_after = float(response.headers.get("Retry-After", 0))
                except ValueError:
                    retry_after = 0.0
                return None, "RATE_LIMITED", "HTTP 429", retry_after

            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                self.stats[service]["retries"] += 1
                if attempt < MAX_RETRIES:
                    time.sleep(attempt)
                continue

            if response.status_code != 200:
                return None, "ERROR", f"HTTP {response.status_code}", 0.0

            try:
                return response.json(), "OK", "", 0.0
            except ValueError as error:
                return None, "ERROR", f"Invalid JSON: {error}", 0.0

        return None, "ERROR", last_error or "Unknown error", 0.0

    # ---- public search ----
    def search(self, service, title=None, author=None):
        stats = self.stats[service]
        stats["searches"] += 1
        key = f"{service}|{normalize(title or '')}|{normalize(author or '')}"

        if self.use_cache and key in self.cache:
            stats["cached"] += 1
            return self.cache[key], "CACHED"

        if self.disabled[service]:
            stats["skipped"] += 1
            reason = self.disabled[service]
            return [], reason if reason in ("RATE_LIMITED", "ERROR") else "SKIPPED"

        url, params = self._build_request(service, title, author)

        data, outcome, error, retry_after = self._http_get(service, url, params)
        stats["requests"] += 1

        # Open Library: one polite retry after a 429. Google: never retry.
        if outcome == "RATE_LIMITED" and service == OL:
            wait = min(max(retry_after, 2.0), 10.0)
            print(f"  {service}: HTTP 429, waiting {wait:.0f}s (one retry)")
            time.sleep(wait)
            data, outcome, error, retry_after = self._http_get(service, url, params)
            stats["requests"] += 1

        time.sleep(REQUEST_DELAY)

        if outcome == "RATE_LIMITED":
            stats["http_429"] += 1
            self.disabled[service] = "RATE_LIMITED"
            print(f"  {service}: RATE_LIMITED -> no more {service} "
                  f"requests in this run.")
            return [], "RATE_LIMITED"

        if outcome == "ERROR":
            stats["errors"] += 1
            self.consecutive_errors[service] += 1
            if self.consecutive_errors[service] >= MAX_CONSECUTIVE_ERRORS:
                self.disabled[service] = "ERROR"
                print(f"  {service}: {MAX_CONSECUTIVE_ERRORS} failures in a row "
                      f"({error}) -> switched off for this run.")
            return [], "ERROR"

        self.consecutive_errors[service] = 0
        stats["ok"] += 1
        docs = (
            parse_open_library(data) if service == OL
            else parse_google_books(data)
        )
        if self.use_cache:
            self.cache[key] = docs          # only successful answers are cached
        return docs, "OK"

    @staticmethod
    def _build_request(service, title, author):
        if service == OL:
            params = {
                "limit": OPEN_LIBRARY_LIMIT,
                "fields": ("key,title,subtitle,author_name,first_publish_year,"
                           "publisher,isbn,edition_count"),
            }
            if title:
                params["title"] = title
            if author:
                params["author"] = author
            return OPEN_LIBRARY_URL, params

        parts = []
        if title:
            parts.append(f'intitle:"{title}"')
        if author:
            parts.append(f'inauthor:"{author}"')
        return GOOGLE_BOOKS_URL, {
            "q": " ".join(parts),
            "maxResults": GOOGLE_BOOKS_LIMIT,
            "printType": "books",
        }


def summarize_service(counter):
    """Per-book status of one service from its outcome counter."""
    total = sum(counter.values())
    if total == 0:
        return "NOT SEARCHED"
    if counter["SKIPPED"] == total:
        return "NOT USED (disabled by user)"

    good = counter["OK"] + counter["CACHED"]
    if good == total:
        return "OK" if counter["OK"] else "OK (cached)"

    problem = "RATE_LIMITED" if counter["RATE_LIMITED"] else "ERROR"
    return problem if good == 0 else f"PARTIAL - {problem}"


# ============================================================
# EVIDENCE FOR ONE CANDIDATE
# ============================================================

def publisher_score(ocr_publishers, cand_publishers):
    """None = no evidence either way. Publisher is support only."""
    if not ocr_publishers or not cand_publishers:
        return None
    best = 0.0
    for ocr_pub in ocr_publishers:
        p_tokens = normalize(ocr_pub).split()
        if not p_tokens:
            continue
        for cand_pub in cand_publishers:
            c_tokens = normalize(cand_pub).split()
            if c_tokens:
                best = max(best, match_weight(p_tokens, c_tokens) / len(p_tokens))
    return best


def evaluate_candidate(clues, cand):
    """Collect all evidence for one candidate. Nothing is accepted here."""
    ev = {"title_score": 0.0, "title_coverage": 0.0, "title_precision": 0.0,
          "title_query": "", "title_view": "", "n_query_tokens": 0,
          "name_only": False}

    cand_author_tokens = author_token_set(cand["authors"])
    best = None

    for q in clues["title_queries"]:
        for view, text in (("main", cand["title_main"]),
                           ("full", cand["title_full"])):
            c_tokens = meaningful_tokens(text)
            if not c_tokens:
                continue

            # OCR words that only repeat the author's name (and are not in
            # the title) do not count against the title.
            adjusted = [
                t for t in q["tokens"]
                if not (token_in(t, cand_author_tokens)
                        and not token_in(t, c_tokens))
            ]
            if not adjusted:
                continue

            score, cov, prec = token_title_score(adjusted, c_tokens)
            rank = score + cov + prec
            if best is None or rank > best["rank"]:
                best = {"rank": rank, "score": score, "cov": cov, "prec": prec,
                        "query": q["text"], "view": view,
                        "n": len(adjusted), "c_tokens": c_tokens}

    if best:
        ev.update({
            "title_score": best["score"], "title_coverage": best["cov"],
            "title_precision": best["prec"], "title_query": best["query"],
            "title_view": best["view"], "n_query_tokens": best["n"],
        })
        # The matched title is just a person's name (e.g. "Otto Mueller"
        # by Otto Mueller): not proof of a book title.
        ev["name_only"] = bool(cand_author_tokens) and all(
            token_in(t, cand_author_tokens) for t in best["c_tokens"]
        )

    ev["title_ok"] = (
        best is not None
        and ev["title_score"] >= TITLE_SCORE_MIN
        and ev["title_coverage"] >= TITLE_COVERAGE_MIN
        and ev["title_precision"] >= TITLE_PRECISION_MIN
    )

    # ---- author evidence (supports a title, never replaces it) ----
    a_clue = max(
        (author_similarity(c, cand["authors"]) for c in clues["author_clues"]),
        default=0.0,
    )
    a_text = author_in_text(cand["authors"], clues["token_pool"])

    # A name printed on the spine that also appears inside the record's
    # title/subtitle (e.g. "...The Life and Art of William Mortensen") is
    # consistent with the record. Counted as support (0.70), never proof.
    full_tokens = meaningful_tokens(cand["title_full"])
    a_title = 0.0
    for clue in clues["author_clues"]:
        clue_tokens = [t for t in normalize(clue).split() if len(t) >= 2]
        if clue_tokens and all(token_in(t, full_tokens) for t in clue_tokens):
            a_title = 0.70

    ev["author_score"] = max(a_clue, a_text, a_title)
    if ev["author_score"] == 0:
        ev["author_source"] = ""
    elif ev["author_score"] == a_clue:
        ev["author_source"] = "author clue on spine matches record authors"
    elif ev["author_score"] == a_text:
        ev["author_source"] = "record author name found in OCR text"
    else:
        ev["author_source"] = "spine name appears in record title"

    # Conflict: spine shows an author, record has authors, none match,
    # and the OCR name is not simply part of the record's title.
    title_tokens = meaningful_tokens(cand["title_full"])
    ev["author_conflict"] = False
    if clues["author_clues"] and cand["authors"]:
        explained_or_matched = False
        for clue in clues["author_clues"]:
            clue_tokens = [t for t in normalize(clue).split() if len(t) >= 2]
            if author_similarity(clue, cand["authors"]) >= 0.5:
                explained_or_matched = True
            elif clue_tokens and all(token_in(t, title_tokens) for t in clue_tokens):
                explained_or_matched = True
        ev["author_conflict"] = not explained_or_matched and a_text < 0.5

    # ---- publisher, year, ISBN (support / edition information) ----
    ev["publisher_score"] = publisher_score(
        clues["publisher_clues"], cand["publishers"])
    ev["publisher_match"] = (
        ev["publisher_score"] is not None and ev["publisher_score"] >= 0.8)

    ev["year_match"] = bool(
        clues["years"] and cand["year"] and cand["year"] in clues["years"])
    cand_isbns = {re.sub(r"[^0-9Xx]", "", i).upper() for i in cand["isbns"]}
    ev["isbn_match"] = bool(cand_isbns & set(clues["isbns"]))

    # ---- volume ----
    vol, _ = parse_volume(cand["title_full"])
    ev["volume"] = vol
    ev["volume_conflict"] = (
        clues["ocr_volume"] is not None and vol is not None
        and vol != clues["ocr_volume"]
    )

    # Diagnostic ranking score (decisions use explicit rules, not this).
    ev["total_score"] = round(
        0.60 * ev["title_score"]
        + 0.25 * ev["author_score"]
        + 0.08 * (ev["publisher_score"] or 0.0)
        + 0.04 * (1.0 if ev["year_match"] else 0.0)
        + 0.03 * (1.0 if ev["isbn_match"] else 0.0),
        4,
    )

    ev["eligible"] = (
        ev["title_ok"] and not ev["name_only"] and not ev["volume_conflict"]
    )

    # Plain-English reason when a candidate is not eligible.
    if ev["eligible"]:
        ev["rejection_reason"] = ""
    elif ev["name_only"]:
        ev["rejection_reason"] = "matched title is only a person's name"
    elif ev["volume_conflict"]:
        ev["rejection_reason"] = "volume differs from the volume on the spine"
    else:
        ev["rejection_reason"] = (
            f"title too different (score {ev['title_score']:.2f}, "
            f"coverage {ev['title_coverage']:.2f}, "
            f"precision {ev['title_precision']:.2f})"
        )
    return ev


# ============================================================
# GROUPING RECORDS INTO WORKS
# ============================================================

def same_work(a, b):
    """Are two catalogue records the same book (maybe different editions)?"""
    ka, kb = a["title_key"], b["title_key"]
    if ka != kb and SequenceMatcher(None, ka, kb).ratio() < 0.92:
        return False

    ta, tb = author_token_set(a["authors"]), author_token_set(b["authors"])
    if not ta or not tb:        # missing author data cannot split a work
        return True
    return any(token_in(t, tb) for t in ta)


def cluster_works(candidates):
    clusters = []
    for cand in sorted(candidates, key=lambda c: -c["total_score"]):
        for cluster in clusters:
            if same_work(cluster["members"][0], cand):
                cluster["members"].append(cand)
                break
        else:
            clusters.append({"members": [cand]})

    for cluster in clusters:
        cluster["score"] = max(m["total_score"] for m in cluster["members"])
        cluster["has_authors"] = any(m["authors"] for m in cluster["members"])
    clusters.sort(key=lambda c: -c["score"])
    return clusters


def record_sort_key(cand, ocr_volume):
    """Order records inside one work: the best-documented edition first."""
    return (
        ocr_volume is not None and cand["volume"] == ocr_volume,
        cand["isbn_match"],
        cand["publisher_match"],
        cand["year_match"],
        cand["title_score"],
        bool(cand["isbns"]),
        bool(cand["authors"]),
        bool(cand["publishers"]),
        cand["total_score"],
    )


# ============================================================
# DECISION
# ============================================================

def decide(clues, candidates):
    """
    Returns a dict:
      status, reason, selected (cand or None), clusters,
      edition_confirmed, flags
    """
    eligible = [c for c in candidates if c["eligible"]]

    if not eligible:
        return {"status": "NOT IDENTIFIED", "selected": None, "clusters": [],
                "edition_confirmed": "", "flags": [],
                "reason": ("No catalogue record matched the title clue "
                           "well enough." if candidates
                           else "No catalogue candidates were returned.")}

    clusters = cluster_works(eligible)
    winner = clusters[0]
    members = sorted(winner["members"],
                     key=lambda c: record_sort_key(c, clues["ocr_volume"]),
                     reverse=True)
    best = members[0]
    # For the title/author tests use the strongest member evidence.
    lead = max(winner["members"], key=lambda c: c["total_score"])

    flags = []
    author_supported = lead["author_score"] >= AUTHOR_SUPPORT_MIN
    standalone_ok = (
        lead["n_query_tokens"] >= STANDALONE_MIN_TOKENS
        and lead["title_score"] >= STANDALONE_MIN_SCORE
    )

    # Short/generic title + the spine names a different author than the
    # record = most likely a DIFFERENT book that shares the title.
    if lead["author_conflict"] and not standalone_ok:
        return {"status": "NOT IDENTIFIED", "selected": None,
                "clusters": clusters, "edition_confirmed": "", "flags": [],
                "reason": ("The title matches a record, but the author on the "
                           "spine does not match its authors: probably a "
                           "different book with the same title.")}

    # 1. Different works with a comparably good title?
    for other in clusters[1:]:
        if not other["has_authors"]:
            continue                       # cannot contest without author data
        other_lead = max(other["members"], key=lambda c: c["total_score"])
        close = other["score"] >= winner["score"] - COMPETITOR_MARGIN
        dominated = author_supported and (
            other_lead["author_score"] < AUTHOR_SUPPORT_MIN)
        if close and not dominated:
            flags.append(
                "another work has an equally good match: "
                f"'{other_lead['title']}' by "
                f"{'; '.join(other_lead['authors']) or 'unknown'}")
            break

    # 2. Author clue on the spine contradicts the best record.
    if lead["author_conflict"]:
        flags.append("the author on the spine does not match the record's authors")

    # 3. Short / generic title without author support.
    if not author_supported and not standalone_ok:
        flags.append("title is short or generic and has no author support")

    # 4. Volumes.
    volumes = {m["volume"] for m in winner["members"]}
    if clues["ocr_volume"] is None and len(volumes) >= 2:
        shown = ", ".join("unnumbered" if v is None else str(v)
                          for v in sorted(volumes, key=lambda v: (v is None, v or 0)))
        flags.append(f"several volumes exist ({shown}) and none is visible on the spine")
    elif clues["ocr_volume"] is not None and not any(
            m["volume"] == clues["ocr_volume"] for m in winner["members"]):
        flags.append(f"volume {clues['ocr_volume']} on the spine not found in the catalogue")

    # 5. Edition information.
    edition_confirmed = (
        "YES" if (best["isbn_match"] or best["publisher_match"] or best["year_match"])
        else "NO")
    distinct_editions = {(tuple(m["publishers"]), m["year"]) for m in winner["members"]}
    if STRICT_EDITION_MODE and edition_confirmed == "NO" and len(distinct_editions) > 1:
        flags.append("several editions exist and the edition cannot be confirmed")

    if flags:
        return {"status": "AMBIGUOUS", "selected": best, "clusters": clusters,
                "edition_confirmed": edition_confirmed, "flags": flags,
                "reason": "Plausible title match, but " + "; ".join(flags) + "."}

    strength = (
        "High" if lead["title_score"] >= 0.95 and author_supported else "Medium")
    parts = [f"Title match {lead['title_score']:.2f}"]
    parts.append(
        f"author support {lead['author_score']:.2f}" if author_supported
        else "distinctive long title, no competing work")
    return {"status": "IDENTIFIED", "selected": best, "clusters": clusters,
            "edition_confirmed": edition_confirmed, "flags": [],
            "confidence": strength,
            "reason": "; ".join(parts) + "; one work, no conflicts."}


def build_evidence_text(cand, clues):
    if cand is None:
        return ""
    bits = [
        f"title {cand['title_score']:.2f} (OCR '{cand['title_query']}' vs "
        f"{cand['title_view']} title)",
        f"author {cand['author_score']:.2f}"
        + (f" ({cand['author_source']})" if cand["author_source"] else ""),
    ]
    if cand["publisher_score"] is not None:
        bits.append(f"publisher {cand['publisher_score']:.2f}")
    elif clues["publisher_clues"]:
        bits.append("publisher on spine, none in record")
    if cand["year_match"]:
        bits.append("year matches")
    if cand["isbn_match"]:
        bits.append("ISBN matches")
    return "; ".join(bits)


# ============================================================
# IDENTIFY ONE BOOK
# ============================================================

def empty_decision(reason):
    return {"status": "NOT IDENTIFIED", "selected": None, "clusters": [],
            "edition_confirmed": "", "flags": [], "reason": reason}


def identify_one_book(client, image_name, ocr_rows):
    clues = build_clues(ocr_rows)

    print("Best OCR text:", clues["best_ocr_text"])
    print("Title queries :", [q["text"] for q in clues["title_queries"]])
    print("Author clues  :", clues["author_clues"])
    print("Publisher clues:", clues["publisher_clues"])

    outcomes = {OL: Counter(), GB: Counter()}
    candidates = {}

    def run(service, title=None, author=None):
        docs, outcome = client.search(service, title, author)
        outcomes[service][outcome] += 1
        for doc in docs:
            candidates.setdefault(doc["id"], dict(doc))

    if clues["title_queries"]:
        # 1) title-only searches
        for q in clues["title_queries"]:
            run(OL, title=q["text"])
            run(GB, title=q["text"])

        # 2) title + author searches (replaces V4.5's author-only searches,
        #    which could only ever create candidates that were never accepted)
        if clues["author_clues"]:
            combo = [q for q in clues["title_queries"]
                     if q["kind"] != "joined"][:MAX_AUTHOR_COMBO_QUERIES]
            for q in combo:
                run(OL, title=q["text"], author=clues["author_clues"][0])
                run(GB, title=q["text"], author=clues["author_clues"][0])

    status_ol = summarize_service(outcomes[OL])
    status_gb = summarize_service(outcomes[GB])
    used = [s for s in (status_ol, status_gb) if not s.startswith("NOT USED")]
    if used and all(s.startswith("OK") for s in used):
        search_complete = "YES"
    elif any(s.startswith("OK") or s.startswith("PARTIAL") for s in used):
        search_complete = "PARTIAL"
    else:
        search_complete = "NO" if clues["title_queries"] else "NOT NEEDED"

    # ---- evidence ----
    cand_list = []
    for cand in candidates.values():
        cand.update(evaluate_candidate(clues, cand))
        cand["title_key"] = title_key(cand["title_main"])
        cand_list.append(cand)

    # ---- decision ----
    if not clues["title_queries"]:
        decision = empty_decision(
            "No usable title clue in OCR (only a name, a publisher, or no text).")
    else:
        decision = decide(clues, cand_list)
        if (decision["status"] == "NOT IDENTIFIED" and not cand_list
                and search_complete in ("NO", "PARTIAL")):
            decision["reason"] = (
                "No candidates returned, and the catalogue search was "
                "incomplete (service unavailable). This is NOT proof that "
                "the book does not exist.")

    return {
        "image": image_name, "clues": clues, "candidates": cand_list,
        "decision": decision, "status_ol": status_ol, "status_gb": status_gb,
        "search_complete": search_complete,
    }


# ============================================================
# BUILD OUTPUT ROWS
# ============================================================

DETAIL_FIELDS = [
    "image", "final_status", "candidate_type", "candidate_rank", "work_rank",
    "relation_to_selected", "candidate_title", "candidate_subtitle",
    "candidate_authors", "publisher", "publication_year", "isbn", "source",
    "title_score", "title_coverage", "title_precision", "author_score",
    "publisher_score", "total_score", "evidence", "identification_reason",
    "rejection_reason", "edition_confirmed", "open_library_status",
    "google_books_status", "search_complete", "source_url",
]

SUMMARY_FIELDS = [
    "image", "final_status", "confidence", "selected_candidate_title",
    "selected_authors", "publisher", "publication_year", "isbn",
    "title_score", "author_score", "publisher_score", "total_score",
    "evidence", "identification_reason", "edition_confirmed",
    "best_ocr_text", "title_queries", "author_clues", "publisher_clues",
    "open_library_status", "google_books_status", "search_complete",
    "n_candidates", "n_alternatives", "alternative_titles",
]


def fmt_score(value):
    return "" if value is None else round(value, 4)


def candidate_row(result, cand, ctype, rank, work_rank, relation):
    decision = result["decision"]
    return {
        "image": result["image"],
        "final_status": decision["status"],
        "candidate_type": ctype,
        "candidate_rank": rank,
        "work_rank": work_rank,
        "relation_to_selected": relation,
        "candidate_title": cand["title"],
        "candidate_subtitle": cand["subtitle"],
        "candidate_authors": "; ".join(cand["authors"]),
        "publisher": "; ".join(cand["publishers"][:3]),
        "publication_year": cand["year"],
        "isbn": "; ".join(cand["isbns"][:3]),
        "source": cand["source"],
        "title_score": round(cand["title_score"], 4),
        "title_coverage": round(cand["title_coverage"], 4),
        "title_precision": round(cand["title_precision"], 4),
        "author_score": round(cand["author_score"], 4),
        "publisher_score": fmt_score(cand["publisher_score"]),
        "total_score": cand["total_score"],
        "evidence": build_evidence_text(cand, result["clues"]),
        "identification_reason": decision["reason"],
        "rejection_reason": cand["rejection_reason"],
        "edition_confirmed": decision["edition_confirmed"],
        "open_library_status": result["status_ol"],
        "google_books_status": result["status_gb"],
        "search_complete": result["search_complete"],
        "source_url": cand["url"],
    }


def build_rows(result):
    """Detailed candidate rows + one summary row. Marks ONE selected."""
    decision = result["decision"]
    selected = decision["selected"]
    clues = result["clues"]
    rows = []
    rank = 0

    if result["candidates"]:
        eligible_ids = set()
        shown_ids = set()

        # 1) selected / leading candidate
        if selected is not None:
            ctype = ("SELECTED" if decision["status"] == "IDENTIFIED"
                     else "LEADING - UNCONFIRMED")
            rank += 1
            rows.append(candidate_row(result, selected, ctype, rank, 1, "THIS IS THE CHOSEN RECORD"))
            shown_ids.add(selected["id"])

        # 2) other records of the same work, then other works
        alt_count = 0
        for work_rank, cluster in enumerate(decision["clusters"], start=1):
            members = sorted(cluster["members"],
                             key=lambda c: record_sort_key(c, clues["ocr_volume"]),
                             reverse=True)
            is_winner = selected is not None and selected["id"] in {m["id"] for m in members}
            for cand in members:
                eligible_ids.add(cand["id"])
                if cand["id"] in shown_ids or alt_count >= MAX_ALTERNATIVES_PER_BOOK:
                    continue
                rank += 1
                alt_count += 1
                shown_ids.add(cand["id"])
                relation = (
                    "SAME WORK - OTHER RECORD/EDITION" if is_winner
                    else "DIFFERENT WORK" if selected is not None
                    else "POSSIBLE WORK")
                rows.append(candidate_row(
                    result, cand, "ALTERNATIVE - REVIEW", rank, work_rank, relation))

        # 3) a few rejected candidates for error analysis
        rejected = sorted(
            [c for c in result["candidates"] if c["id"] not in eligible_ids],
            key=lambda c: -c["total_score"])[:MAX_REJECTED_PER_BOOK]
        for cand in rejected:
            rank += 1
            rows.append(candidate_row(result, cand, "REJECTED", rank, "", "NOT A MATCH"))
    else:
        rows.append({
            "image": result["image"], "final_status": decision["status"],
            "candidate_type": "NO CANDIDATE", "candidate_rank": "",
            "work_rank": "", "relation_to_selected": "",
            "identification_reason": decision["reason"],
            "open_library_status": result["status_ol"],
            "google_books_status": result["status_gb"],
            "search_complete": result["search_complete"],
        })

    alt_titles = [r["candidate_title"] for r in rows
                  if r.get("candidate_type") == "ALTERNATIVE - REVIEW"]
    summary = {
        "image": result["image"],
        "final_status": decision["status"],
        "confidence": decision.get("confidence", ""),
        "selected_candidate_title": selected["title"] if selected else "",
        "selected_authors": "; ".join(selected["authors"]) if selected else "",
        "publisher": "; ".join(selected["publishers"][:3]) if selected else "",
        "publication_year": selected["year"] if selected else "",
        "isbn": "; ".join(selected["isbns"][:3]) if selected else "",
        "title_score": round(selected["title_score"], 4) if selected else "",
        "author_score": round(selected["author_score"], 4) if selected else "",
        "publisher_score": fmt_score(selected["publisher_score"]) if selected else "",
        "total_score": selected["total_score"] if selected else "",
        "evidence": build_evidence_text(selected, clues),
        "identification_reason": decision["reason"],
        "edition_confirmed": decision["edition_confirmed"],
        "best_ocr_text": clues["best_ocr_text"],
        "title_queries": " | ".join(q["text"] for q in clues["title_queries"]),
        "author_clues": " | ".join(clues["author_clues"]),
        "publisher_clues": " | ".join(clues["publisher_clues"]),
        "open_library_status": result["status_ol"],
        "google_books_status": result["status_gb"],
        "search_complete": result["search_complete"],
        "n_candidates": len(result["candidates"]),
        "n_alternatives": len(alt_titles),
        "alternative_titles": " | ".join(alt_titles[:5]),
    }
    return rows, summary


# ============================================================
# EVALUATION AGAINST MANUAL GROUND TRUTH
# ============================================================

def title_matches_truth(actual_title, actual_authors, cand_title, cand_authors):
    """Does a catalogue title/author agree with the manually verified one?"""
    a_main = actual_title.split(":")[0]
    best = 0.0
    for left in (actual_title, a_main):
        for right in (cand_title, cand_title.split(":")[0]):
            score, cov, prec = token_title_score(
                meaningful_tokens(left), meaningful_tokens(right))
            if cov >= 0.85 and prec >= 0.6:
                best = max(best, score)
    if best < 0.85:
        return False
    if actual_authors and cand_authors:
        return max(
            (author_similarity(a, cand_authors)
             for a in re.split(r"[;|]", actual_authors) if a.strip()),
            default=0.0) >= 0.6
    return True


def write_ground_truth_template(images):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_TEMPLATE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "actual_title", "actual_authors", "publisher",
            "publication_year", "isbn", "verification_source",
            "identifiable_from_photo"])
        for name in images:
            writer.writerow([name, "", "", "", "", "", "", ""])


def evaluate_against_truth(summaries, details, truth_path):
    with open(truth_path, "r", encoding="utf-8-sig", newline="") as f:
        truth = {Path(r["image"]).name: r for r in csv.DictReader(f)
                 if (r.get("image") or "").strip()}

    by_image = defaultdict(list)
    for row in details:
        if row.get("candidate_type") in ("SELECTED", "LEADING - UNCONFIRMED",
                                         "ALTERNATIVE - REVIEW"):
            by_image[row["image"]].append(row)

    counts = Counter()
    top1 = top5 = 0
    evaluated = 0

    for summary in summaries:
        t = truth.get(summary["image"])
        if not t or not (t.get("actual_title") or "").strip():
            continue
        identifiable = (t.get("identifiable_from_photo") or "yes").strip().lower()
        evaluated += 1

        status = summary["final_status"]
        counts[status] += 1

        if identifiable == "no":
            if status == "IDENTIFIED":
                counts["false_positive"] += 1
            continue

        counts["identifiable"] += 1
        actual_t, actual_a = t["actual_title"], t.get("actual_authors", "")

        if status == "IDENTIFIED":
            if title_matches_truth(actual_t, actual_a,
                                   summary["selected_candidate_title"],
                                   [a for a in summary["selected_authors"].split("; ") if a]):
                counts["correct"] += 1
            else:
                counts["false_positive"] += 1

        ranked = sorted(by_image.get(summary["image"], []),
                        key=lambda r: r["candidate_rank"])
        seen_works, hits = [], []
        for row in ranked:
            work = row["work_rank"]
            if work not in seen_works:
                seen_works.append(work)
            if len(seen_works) > 5:
                break
            hits.append((len(seen_works), title_matches_truth(
                actual_t, actual_a, row["candidate_title"],
                [a for a in row["candidate_authors"].split("; ") if a])))
        if hits and hits[0][1] and hits[0][0] == 1:
            top1 += 1
        if any(ok for _, ok in hits):
            top5 += 1

    if not evaluated:
        return None

    identified = counts["IDENTIFIED"]
    return {
        "books_evaluated": evaluated,
        "identifiable_from_photo": counts["identifiable"],
        "identified_correct": counts["correct"],
        "identified_wrong_false_positives": counts["false_positive"],
        "ambiguous": counts["AMBIGUOUS"],
        "not_identified": counts["NOT IDENTIFIED"],
        "correct_identification_rate": round(
            counts["correct"] / max(counts["identifiable"], 1), 4),
        "false_positive_rate_among_identified": round(
            counts["false_positive"] / max(identified, 1), 4),
        "top1_accuracy_candidates": round(top1 / max(counts["identifiable"], 1), 4),
        "top5_accuracy_candidates": round(top5 / max(counts["identifiable"], 1), 4),
        "note": ("AMBIGUOUS and NOT IDENTIFIED are never counted as correct. "
                 "Top-1/Top-5 look at the ranked candidate list regardless of "
                 "status (top-5 = five best distinct works)."),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    global STRICT_EDITION_MODE

    parser = argparse.ArgumentParser(description="ShelfSense AI V4.6")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and do not write the API response cache")
    parser.add_argument("--skip-google", action="store_true",
                        help="do not use Google Books at all")
    parser.add_argument("--strict-edition", action="store_true",
                        help="return AMBIGUOUS when the edition cannot be confirmed")
    parser.add_argument("--ground-truth", default=str(GROUND_TRUTH_CSV),
                        help="path to manually verified ground-truth CSV")
    args = parser.parse_args()

    if args.strict_edition:
        STRICT_EDITION_MODE = True

    print("\nShelfSense AI - Book Identification V4.6")
    print("=" * 70)

    if not OCR_CSV.exists():
        print("ERROR: OCR CSV not found:\n", OCR_CSV)
        return
    if not CROPS_DIR.exists():
        print("ERROR: Cropped books folder not found:\n", CROPS_DIR)
        return

    ocr_data = load_ocr()
    crop_files = sorted(
        p for p in CROPS_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    print("OCR images :", len(ocr_data))
    print("Book crops :", len(crop_files))
    print("Services   :", OL, "+", "(Google Books skipped)" if args.skip_google else GB)
    print("Cache      :", "off" if args.no_cache else CACHE_FILE)
    print("Strict edition mode:", STRICT_EDITION_MODE)

    client = CatalogueClient(use_cache=not args.no_cache,
                             skip_google=args.skip_google)

    all_details, all_summaries = [], []

    for index, crop in enumerate(crop_files, start=1):
        name = crop.name
        print("\n" + "-" * 70)
        print(f"[{index}/{len(crop_files)}] {name}")

        rows = ocr_data.get(name, [])
        if rows:
            result = identify_one_book(client, name, rows)
        else:
            print("No OCR text.")
            result = {
                "image": name, "clues": build_clues([]), "candidates": [],
                "decision": empty_decision("No OCR text for this crop."),
                "status_ol": "NOT SEARCHED", "status_gb": "NOT SEARCHED",
                "search_complete": "NOT NEEDED",
            }

        detail_rows, summary = build_rows(result)
        all_details.extend(detail_rows)
        all_summaries.append(summary)
        client.save_cache()

        d = result["decision"]
        print(f"FINAL STATUS: {d['status']}")
        if d["selected"]:
            sel = d["selected"]
            print(f"  {'SELECTED' if d['status'] == 'IDENTIFIED' else 'LEADING (unconfirmed)'}:"
                  f" {sel['title']} | {'; '.join(sel['authors'])} | {sel['source']}")
        print("  Reason:", d["reason"])
        print(f"  Open Library: {result['status_ol']} | "
              f"Google Books: {result['status_gb']} | "
              f"search complete: {result['search_complete']}")

    # ---- write CSVs ----
    for path, fields, data in (
        (OUTPUT_CSV, DETAIL_FIELDS, all_details),
        (SUMMARY_CSV, SUMMARY_FIELDS, all_summaries),
    ):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)

    # ---- report ----
    status_counts = Counter(s["final_status"] for s in all_summaries)
    api = {}
    for svc, c in client.stats.items():
        real = c["requests"]
        api[svc] = {
            "searches_requested": c["searches"], "real_http_requests": real,
            "ok": c["ok"], "from_cache": c["cached"],
            "http_429_rate_limited": c["http_429"], "errors": c["errors"],
            "skipped_after_disable": c["skipped"], "retries": c["retries"],
            "rate_limit_frequency": round(c["http_429"] / real, 4) if real else 0.0,
            "disabled_for_rest_of_run": client.disabled[svc] or "no",
        }

    report = {
        "version": "4.6",
        "books_processed": len(all_summaries),
        "IDENTIFIED": status_counts["IDENTIFIED"],
        "AMBIGUOUS": status_counts["AMBIGUOUS"],
        "NOT IDENTIFIED": status_counts["NOT IDENTIFIED"],
        "books_with_incomplete_search": sum(
            1 for s in all_summaries if s["search_complete"] in ("NO", "PARTIAL")),
        "api": api,
    }

    truth_path = Path(args.ground_truth)
    if truth_path.exists():
        report["evaluation"] = evaluate_against_truth(
            all_summaries, all_details, truth_path)
    else:
        write_ground_truth_template([c.name for c in crop_files])
        report["evaluation"] = (
            "No ground-truth file yet. A template was written to "
            f"{GROUND_TRUTH_TEMPLATE}. Fill it in and save it as "
            f"{GROUND_TRUTH_CSV.name} to get accuracy numbers.")

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("V4.6 IDENTIFICATION COMPLETED")
    print("=" * 70)
    print("Books processed :", report["books_processed"])
    print("IDENTIFIED      :", report["IDENTIFIED"])
    print("AMBIGUOUS       :", report["AMBIGUOUS"])
    print("NOT IDENTIFIED  :", report["NOT IDENTIFIED"])
    print("Incomplete catalogue searches:", report["books_with_incomplete_search"])
    for svc, a in api.items():
        print(f"{svc}: {a['real_http_requests']} requests, "
              f"{a['from_cache']} cached, {a['http_429_rate_limited']} x HTTP 429, "
              f"{a['errors']} errors, disabled: {a['disabled_for_rest_of_run']}")
    if isinstance(report["evaluation"], dict):
        print("\nEvaluation vs ground truth:")
        for key, value in report["evaluation"].items():
            print(f"  {key}: {value}")
    else:
        print("\n" + report["evaluation"])
    print("\nSaved:")
    print(" ", OUTPUT_CSV)
    print(" ", SUMMARY_CSV)
    print(" ", REPORT_JSON)
    print("\nREMINDER: AMBIGUOUS / LEADING candidates are NOT confirmed "
          "identifications. Review them manually.")


if __name__ == "__main__":
    main()
