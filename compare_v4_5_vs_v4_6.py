import csv
from pathlib import Path
from collections import defaultdict

# ============================================================
# Compare V4.5 and V4.6 results, one line per book.
# Run AFTER both identify_books_v4_5.py and identify_books_v4_6.py.
# Neither V4.5 nor V4.6 is modified by this script.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FOLDER = BASE_DIR / "outputs" / "book_identification"

V45_CSV = FOLDER / "identified_books_v4_5.csv"
V46_SUMMARY_CSV = FOLDER / "identified_books_v4_6_summary.csv"
OUT_CSV = FOLDER / "comparison_v4_5_vs_v4_6.csv"


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    for path in (V45_CSV, V46_SUMMARY_CSV):
        if not path.exists():
            print("Missing file:", path)
            return

    # V4.5: many rows per image. Collect every ACCEPTED record.
    v45_accepted = defaultdict(list)
    v45_images = set()
    for row in read_csv(V45_CSV):
        v45_images.add(row["image"])
        if row.get("candidate_decision") == "ACCEPTED":
            v45_accepted[row["image"]].append(
                f"{row.get('title', '')} ({row.get('authors', '')})")

    v46 = {row["image"]: row for row in read_csv(V46_SUMMARY_CSV)}

    out_rows = []
    for image in sorted(v45_images | set(v46)):
        old = v45_accepted.get(image, [])
        new = v46.get(image, {})
        status = new.get("final_status", "(missing)")

        if len(old) > 1:
            note = (f"V4.5 accepted {len(old)} records for ONE book "
                    "(problem 2 - check V4.6 picked one)")
        elif old and status == "IDENTIFIED":
            same = (old[0].split(" (")[0].lower()[:25]
                    == new.get("selected_candidate_title", "").lower()[:25])
            note = ("same title in both versions" if same
                    else "DIFFERENT title - review manually")
        elif old and status != "IDENTIFIED":
            note = "V4.5 accepted, V4.6 did NOT identify - was V4.5 a false positive?"
        elif not old and status == "IDENTIFIED":
            note = "NEW in V4.6 - verify manually"
        else:
            note = "neither version identified"

        out_rows.append({
            "image": image,
            "v4_5_accepted_count": len(old),
            "v4_5_accepted": " || ".join(old),
            "v4_6_status": status,
            "v4_6_selected_title": new.get("selected_candidate_title", ""),
            "v4_6_selected_authors": new.get("selected_authors", ""),
            "v4_6_reason": new.get("identification_reason", ""),
            "note": note,
        })

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"{'image':<14}{'V4.5 accepted':>14}  {'V4.6 status':<15} note")
    print("-" * 100)
    for r in out_rows:
        print(f"{r['image']:<14}{r['v4_5_accepted_count']:>14}  "
              f"{r['v4_6_status']:<15} {r['note']}")
    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()
