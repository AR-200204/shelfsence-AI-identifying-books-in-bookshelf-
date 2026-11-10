"""
Compare V4.6 and V4.6.1 results.

Run after:
    python identify_books_v4_6.py
    python identify_books_v4_6_1.py

Inputs:
    outputs/book_identification/identified_books_v4_6_summary.csv
    outputs/book_identification/identified_books_v4_6_1_summary.csv

Output:
    outputs/book_identification/comparison_v4_6_vs_v4_6_1.csv
"""

import csv
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FOLDER = BASE_DIR / "outputs" / "book_identification"

V46_SUMMARY = FOLDER / "identified_books_v4_6_summary.csv"
V461_SUMMARY = FOLDER / "identified_books_v4_6_1_summary.csv"
OUT_CSV = FOLDER / "comparison_v4_6_vs_v4_6_1.csv"


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get(row, key):
    return (row.get(key) or "").strip()


def norm_title(title):
    title = title or ""
    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split())


def classify_change(old_status, new_status, old_title, new_title):
    """
    Return:
        change_type, review_flag
    """
    old_title_n = norm_title(old_title)
    new_title_n = norm_title(new_title)

    if not old_status and not new_status:
        return "MISSING_DATA", "CHECK_FILES"

    if old_status == new_status:
        if old_status in ("IDENTIFIED", "AMBIGUOUS"):
            if old_title_n == new_title_n:
                return "SAME_SELECTION", ""
            return "DIFFERENT_SELECTION", "REVIEW"
        return "UNCHANGED", ""

    if new_status == "IDENTIFIED" and old_status != "IDENTIFIED":
        return "NEW_IDENTIFIED", "VERIFY_MANUALLY"

    if old_status == "IDENTIFIED" and new_status != "IDENTIFIED":
        return "IDENTIFICATION_REMOVED", "CHECK_IF_OLD_WAS_FALSE_POSITIVE"

    if old_status == "AMBIGUOUS" and new_status == "IDENTIFIED":
        return "AMBIGUOUS_TO_IDENTIFIED", "CHECK_VOLUME_EDITION"

    if old_status == "IDENTIFIED" and new_status == "AMBIGUOUS":
        return "IDENTIFIED_TO_AMBIGUOUS", "SAFER_BUT_REVIEW"

    if new_status == "AMBIGUOUS" and old_status == "NOT IDENTIFIED":
        return "NEW_AMBIGUOUS_LEAD", "REVIEW_LEADING_CANDIDATE"

    if old_status == "AMBIGUOUS" and new_status == "NOT IDENTIFIED":
        return "AMBIGUOUS_TO_NOT_IDENTIFIED", "SAFER_REJECTION"

    return "STATUS_CHANGED", "REVIEW"


def main():
    missing = []

    if not V46_SUMMARY.exists():
        missing.append(str(V46_SUMMARY))

    if not V461_SUMMARY.exists():
        missing.append(str(V461_SUMMARY))

    if missing:
        print("Missing required file(s):")
        for p in missing:
            print(" ", p)
        print("\nRun identify_books_v4_6.py first, then identify_books_v4_6_1.py.")
        return

    old_rows = {}
    for row in read_csv(V46_SUMMARY):
        image = get(row, "image")
        if image:
            old_rows[image] = row

    new_rows = {}
    for row in read_csv(V461_SUMMARY):
        image = get(row, "image")
        if image:
            new_rows[image] = row

    all_images = sorted(set(old_rows) | set(new_rows))

    out_rows = []

    for image in all_images:
        old = old_rows.get(image, {})
        new = new_rows.get(image, {})

        old_status = get(old, "final_status") or "(missing)"
        new_status = get(new, "final_status") or "(missing)"

        old_title = get(old, "selected_candidate_title")
        new_title = get(new, "selected_candidate_title")

        change, review_flag = classify_change(
            old_status,
            new_status,
            old_title,
            new_title
        )

        out_rows.append({
            "image": image,

            "v4_6_status": old_status,
            "v4_6_confidence": get(old, "confidence"),
            "v4_6_selected_title": old_title,
            "v4_6_selected_authors": get(old, "selected_authors"),
            "v4_6_reason": get(old, "identification_reason"),
            "v4_6_search_complete": get(old, "search_complete"),

            "v4_6_1_status": new_status,
            "v4_6_1_confidence": get(new, "confidence"),
            "v4_6_1_selected_title": new_title,
            "v4_6_1_selected_authors": get(new, "selected_authors"),
            "v4_6_1_reason": get(new, "identification_reason"),
            "v4_6_1_search_complete": get(new, "search_complete"),

            "change": change,
            "review_flag": review_flag,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print("\nV4.6 vs V4.6.1 comparison")
    print("=" * 110)
    print(
        f"{'image':<14}"
        f"{'V4.6':<16}"
        f"{'V4.6.1':<16}"
        f"{'change':<32}"
        f"{'review':<30}"
    )
    print("-" * 110)

    for r in out_rows:
        print(
            f"{r['image']:<14}"
            f"{r['v4_6_status']:<16}"
            f"{r['v4_6_1_status']:<16}"
            f"{r['change']:<32}"
            f"{r['review_flag']:<30}"
        )

    print("\nSaved:", OUT_CSV)

    print("\nImportant:")
    print("  NEW_IDENTIFIED does not mean correct. It means V4.6.1 found a candidate.")
    print("  You must manually verify new IDENTIFIED results.")
    print("  IDENTIFICATION_REMOVED may be good if V4.6 was a false positive.")
    print("  AMBIGUOUS results are not confirmed identifications.")


if __name__ == "__main__":
    main()