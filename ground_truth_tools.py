"""
ShelfSense AI - Ground Truth Tools

This script helps you create a manually verified ground-truth file
for evaluating book identification results.

It does NOT automatically decide what the real book is.
It only prepares a review CSV with system suggestions, then converts
your manually filled review CSV into ground_truth.csv.

Commands:

    python ground_truth_tools.py review
    python ground_truth_tools.py finalize
    python ground_truth_tools.py validate

Files used:

    outputs/book_identification/identified_books_v4_6_1_summary.csv
    outputs/book_identification/identified_books_v4_6_summary.csv

Files created:

    outputs/evaluation/ground_truth_review.csv
    outputs/evaluation/ground_truth.csv
"""

import csv
import sys
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "outputs" / "evaluation"
BOOK_DIR = BASE_DIR / "outputs" / "book_identification"

REVIEW_CSV = EVAL_DIR / "ground_truth_review.csv"
GROUND_TRUTH_CSV = EVAL_DIR / "ground_truth.csv"

V461_SUMMARY = BOOK_DIR / "identified_books_v4_6_1_summary.csv"
V46_SUMMARY = BOOK_DIR / "identified_books_v4_6_summary.csv"

PLACEHOLDER_NOT_IDENTIFIABLE = "NOT IDENTIFIABLE FROM PHOTO"

REVIEW_COLUMNS = [
    # Reference columns from the system. Do not edit these manually.
    "image",
    "system_version",
    "system_status",
    "system_confidence",
    "system_selected_title",
    "system_selected_authors",
    "system_publisher",
    "system_publication_year",
    "system_isbn",
    "system_reason",
    "system_best_ocr_text",
    "system_title_queries",
    "system_author_clues",
    "system_publisher_clues",

    # Manual ground-truth columns. You fill these.
    "actual_title",
    "actual_authors",
    "publisher",
    "publication_year",
    "isbn",
    "verification_source",
    "identifiable_from_photo",
    "notes",
]

GT_COLUMNS = [
    "image",
    "actual_title",
    "actual_authors",
    "publisher",
    "publication_year",
    "isbn",
    "verification_source",
    "identifiable_from_photo",
]


def clean(value):
    return str(value or "").strip()


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def choose_summary_source():
    """
    Prefer V4.6.1 summary. If missing, fall back to V4.6 summary.
    """
    if V461_SUMMARY.exists():
        return V461_SUMMARY, "v4.6.1"

    if V46_SUMMARY.exists():
        return V46_SUMMARY, "v4.6"

    return None, ""


def cmd_review():
    """
    Create ground_truth_review.csv from the latest identification summary.

    The review file contains:
    - system suggestions for reference
    - empty manual columns for you to fill

    IMPORTANT:
    Do not copy system_selected_title into actual_title unless you have
    manually verified that it is the real book.
    """
    source, version = choose_summary_source()

    if source is None:
        print("ERROR: No identification summary CSV found.")
        print("Expected one of:")
        print(" ", V461_SUMMARY)
        print(" ", V46_SUMMARY)
        print("\nRun identify_books_v4_6_1.py first.")
        return 1

    rows = read_csv(source)

    if not rows:
        print("ERROR: Summary CSV exists but has no rows:", source)
        return 1

    out_rows = []

    for row in rows:
        image = clean(row.get("image"))

        if not image:
            continue

        out_rows.append({
            # System reference columns
            "image": image,
            "system_version": version,
            "system_status": clean(row.get("final_status")),
            "system_confidence": clean(row.get("confidence")),
            "system_selected_title": clean(row.get("selected_candidate_title")),
            "system_selected_authors": clean(row.get("selected_authors")),
            "system_publisher": clean(row.get("publisher")),
            "public_systemation_year": clean(row.get("publication_year")),
            "system_publication_year": clean(row.get("publication_year")),
            "system_isbn": clean(row.get("isbn")),
            "system_reason": clean(row.get("identification_reason")),
            "system_best_ocr_text": clean(row.get("best_ocr_text")),
            "system_title_queries": clean(row.get("title_queries")),
            "system_author_clues": clean(row.get("author_clues")),
            "system_publisher_clues": clean(row.get("publisher_clues")),

            # Manual columns, intentionally blank
            "actual_title": "",
            "actual_authors": "",
            "publisher": "",
            "publication_year": "",
            "isbn": "",
            "verification_source": "",
            "identifiable_from_photo": "",
            "notes": "",
        })

    write_csv(REVIEW_CSV, REVIEW_COLUMNS, out_rows)

    print("\nGround truth review file created:")
    print(" ", REVIEW_CSV)
    print("\nSource summary:")
    print(" ", source)
    print("Version:")
    print(" ", version)
    print("\nRows prepared:", len(out_rows))

    print("\nNow open the review CSV and fill ONLY these columns:")
    print("  actual_title")
    print("  actual_authors")
    print("  publisher")
    print("  publication_year")
    print("  isbn")
    print("  verification_source")
    print("  identifiable_from_photo")
    print("  notes")

    print("\nImportant rules:")
    print("  1. Do not use system_selected_title as truth unless verified.")
    print("  2. Do not use Qwen guesses as truth.")
    print("  3. If the photo does not show enough title evidence, set:")
    print("       identifiable_from_photo = no")
    print("  4. For multiple authors, use semicolon separation:")
    print("       Name One; Name Two")
    print("  5. verification_source should say how you verified the book, e.g.:")
    print("       physical book")
    print("       publisher website")
    print("       Open Library record checked manually")
    print("       library catalogue")
    print("       manual inspection")

    print("\nAfter filling the file, save it and close Excel/CSV editor.")
    print("Then run:")
    print("  python ground_truth_tools.py finalize")
    print("  python ground_truth_tools.py validate")

    return 0


def cmd_finalize():
    """
    Convert the manually filled ground_truth_review.csv into
    ground_truth.csv using the schema expected by identify_books_v4_6.py.
    """
    if not REVIEW_CSV.exists():
        print("ERROR: Review file not found:")
        print(" ", REVIEW_CSV)
        print("\nRun this first:")
        print("  python ground_truth_tools.py review")
        return 1

    rows = read_csv(REVIEW_CSV)

    if not rows:
        print("ERROR: Review CSV exists but has no rows:", REVIEW_CSV)
        return 1

    gt_rows = []
    warnings = []
    skipped = []

    for row_number, row in enumerate(rows, start=2):
        image = clean(row.get("image"))

        if not image:
            skipped.append(f"Row {row_number}: missing image name")
            continue

        actual_title = clean(row.get("actual_title"))
        actual_authors = clean(row.get("actual_authors"))
        publisher = clean(row.get("publisher"))
        publication_year = clean(row.get("publication_year"))
        isbn = clean(row.get("isbn"))
        verification_source = clean(row.get("verification_source"))
        identifiable = clean(row.get("identifiable_from_photo")).lower()

        # If user left identifiable blank, infer safely.
        if identifiable not in ("yes", "no"):
            if actual_title:
                identifiable = "yes"
                warnings.append(
                    f"{image}: identifiable_from_photo was blank; "
                    "set to yes because actual_title was filled."
                )
            else:
                identifiable = "no"
                warnings.append(
                    f"{image}: identifiable_from_photo was blank and "
                    "actual_title was empty; set to no."
                )

        if identifiable == "no":
            # Keep a placeholder title so the evaluation script can count
            # this book as evaluated but not identifiable from the photo.
            if not actual_title:
                actual_title = PLACEHOLDER_NOT_IDENTIFIABLE

            if not verification_source:
                verification_source = "manual photo review"
                warnings.append(
                    f"{image}: identifiable_from_photo=no but "
                    "verification_source was empty; set to 'manual photo review'."
                )

        else:
            # identifiable_from_photo = yes
            if not actual_title:
                skipped.append(
                    f"{image}: marked identifiable_from_photo=yes "
                    "but actual_title is empty."
                )
                continue

            if not verification_source:
                verification_source = "manual review"
                warnings.append(
                    f"{image}: identifiable_from_photo=yes but "
                    "verification_source was empty; set to 'manual review'."
                )

        gt_rows.append({
            "image": image,
            "actual_title": actual_title,
            "actual_authors": actual_authors,
            "publisher": publisher,
            "publication_year": publication_year,
            "isbn": isbn,
            "verification_source": verification_source,
            "identifiable_from_photo": identifiable,
        })

    if not gt_rows:
        print("ERROR: No valid ground-truth rows were created.")
        if skipped:
            print("\nSkipped rows:")
            for item in skipped:
                print(" -", item)
        return 1

    # Backup existing ground truth if present.
    if GROUND_TRUTH_CSV.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = EVAL_DIR / f"ground_truth_backup_{stamp}.csv"
        backup.write_bytes(GROUND_TRUTH_CSV.read_bytes())
        print("\nOld ground truth backed up to:")
        print(" ", backup)

    write_csv(GROUND_TRUTH_CSV, GT_COLUMNS, gt_rows)

    print("\nGround truth file created:")
    print(" ", GROUND_TRUTH_CSV)
    print("\nRows written:", len(gt_rows))

    yes_count = sum(1 for r in gt_rows if r["identifiable_from_photo"] == "yes")
    no_count = sum(1 for r in gt_rows if r["identifiable_from_photo"] == "no")

    print("Identifiable from photo: yes =", yes_count)
    print("Identifiable from photo: no  =", no_count)

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(" -", warning)

    if skipped:
        print("\nSkipped:")
        for item in skipped:
            print(" -", item)

    print("\nNext run:")
    print("  python ground_truth_tools.py validate")
    print("  python identify_books_v4_6_1.py")

    return 0


def cmd_validate():
    """
    Check ground_truth.csv for obvious problems before evaluation.
    """
    if not GROUND_TRUTH_CSV.exists():
        print("ERROR: Ground truth file not found:")
        print(" ", GROUND_TRUTH_CSV)
        print("\nRun:")
        print("  python ground_truth_tools.py finalize")
        return 1

    with open(GROUND_TRUTH_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    missing_columns = [col for col in GT_COLUMNS if col not in fields]

    if missing_columns:
        print("ERROR: ground_truth.csv is missing required columns:")
        for col in missing_columns:
            print(" -", col)
        return 1

    images = [clean(row.get("image")) for row in rows]
    duplicates = sorted({img for img in images if images.count(img) > 1})

    invalid_identifiable = []
    yes_missing_title = []
    yes_placeholder_title = []
    no_missing_verification = []
    yes_missing_verification = []

    for row in rows:
        image = clean(row.get("image"))
        actual_title = clean(row.get("actual_title"))
        verification_source = clean(row.get("verification_source"))
        identifiable = clean(row.get("identifiable_from_photo")).lower()

        if identifiable not in ("yes", "no"):
            invalid_identifiable.append(image)

        if identifiable == "yes":
            if not actual_title:
                yes_missing_title.append(image)

            if actual_title.upper() == PLACEHOLDER_NOT_IDENTIFIABLE:
                yes_placeholder_title.append(image)

            if not verification_source:
                yes_missing_verification.append(image)

        if identifiable == "no":
            if not verification_source:
                no_missing_verification.append(image)

    print("\nGround truth validation")
    print("=" * 70)
    print("File:", GROUND_TRUTH_CSV)
    print("Rows:", len(rows))

    yes_count = sum(
        1 for row in rows
        if clean(row.get("identifiable_from_photo")).lower() == "yes"
    )
    no_count = sum(
        1 for row in rows
        if clean(row.get("identifiable_from_photo")).lower() == "no"
    )

    print("identifiable_from_photo=yes:", yes_count)
    print("identifiable_from_photo=no :", no_count)

    problems = False

    if duplicates:
        problems = True
        print("\nDuplicate images:")
        for item in duplicates:
            print(" -", item)

    if invalid_identifiable:
        problems = True
        print("\nInvalid identifiable_from_photo values:")
        for item in invalid_identifiable:
            print(" -", item)

    if yes_missing_title:
        problems = True
        print("\nMarked yes but missing actual_title:")
        for item in yes_missing_title:
            print(" -", item)

    if yes_placeholder_title:
        problems = True
        print("\nMarked yes but uses placeholder title:")
        for item in yes_placeholder_title:
            print(" -", item)

    if yes_missing_verification:
        problems = True
        print("\nMarked yes but missing verification_source:")
        for item in yes_missing_verification:
            print(" -", item)

    if no_missing_verification:
        problems = True
        print("\nMarked no but missing verification_source:")
        for item in no_missing_verification:
            print(" -", item)

    if not problems:
        print("\nValidation passed.")
        print("\nNow run evaluation:")
        print("  python identify_books_v4_6_1.py")
        return 0

    print("\nValidation found issues. Fix ground_truth_review.csv, then run:")
    print("  python ground_truth_tools.py finalize")
    print("  python ground_truth_tools.py validate")

    return 1


def print_usage():
    print("Usage:")
    print("  python ground_truth_tools.py review")
    print("  python ground_truth_tools.py finalize")
    print("  python ground_truth_tools.py validate")
    print()
    print("Recommended order:")
    print("  1. python ground_truth_tools.py review")
    print("  2. Fill outputs/evaluation/ground_truth_review.csv manually")
    print("  3. python ground_truth_tools.py finalize")
    print("  4. python ground_truth_tools.py validate")
    print("  5. python identify_books_v4_6_1.py")


def main():
    if len(sys.argv) < 2:
        print_usage()
        return 1

    command = sys.argv[1].lower()

    if command == "review":
        return cmd_review()

    if command == "finalize":
        return cmd_finalize()

    if command == "validate":
        return cmd_validate()

    print("Unknown command:", command)
    print()
    print_usage()
    return 1


if __name__ == "__main__":
    sys.exit(main())