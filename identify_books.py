
import csv
import os
import re
import time
import requests

# Input from our PaddleOCR V2 experiment
INPUT_CSV = "outputs/ocr_results_paddle_v2/paddleocr_v2_comparison.csv"

# Output folder
OUTPUT_FOLDER = "outputs/book_identification"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_FOLDER,
    "identified_books.csv"
)

API_URL = "https://openlibrary.org/search.json"

session = requests.Session()
session.headers.update({
    "User-Agent": "ShelfSenseAI/1.0 (educational research project)"
})


def clean_text(text):
    """Clean OCR text before searching."""

    text = text.replace("|", " ")
    text = re.sub(r"[^a-zA-Z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_ocr_results():
    """Read OCR results and group them by image."""

    grouped = {}

    with open(INPUT_CSV, "r", encoding="utf-8-sig") as file:

        reader = csv.DictReader(file)

        for row in reader:

            image = row["image"]
            method = row["method"]
            text = row["detected_text"].strip()

            if not text:
                continue

            if image not in grouped:
                grouped[image] = {}

            grouped[image].setdefault(method, []).append(text)

    return grouped


def choose_search_text(methods):
    """Prefer counterclockwise OCR, then original OCR."""

    priority = [
        "counterclockwise",
        "original",
        "counterclockwise_contrast",
        "clockwise"
    ]

    for method in priority:

        texts = methods.get(method, [])

        if texts:
            combined = " ".join(texts)
            return clean_text(combined), method

    return "", ""


def search_open_library(query):

    params = {
        "q": query,
        "limit": 5,
        "fields": (
            "key,title,author_name,first_publish_year,"
            "isbn,publisher,cover_i,edition_count"
        )
    }

    try:

        response = session.get(
            API_URL,
            params=params,
            timeout=25
        )

        response.raise_for_status()

        data = response.json()

        return data.get("docs", [])

    except requests.RequestException as error:

        print("API error:", error)
        return []


def main():

    if not os.path.exists(INPUT_CSV):
        print("OCR CSV file not found:", INPUT_CSV)
        return

    grouped = load_ocr_results()

    identified = []

    print("\nShelfSense AI - Book Identification")
    print("=" * 50)

    for image, methods in sorted(grouped.items()):

        query, selected_method = choose_search_text(methods)

        print(f"\nImage: {image}")
        print("OCR method:", selected_method)
        print("Search query:", query)

        if not query:
            print("No readable OCR text.")
            continue

        books = search_open_library(query)

        if not books:
            print("No matching books found.")

            identified.append({
                "image": image,
                "ocr_text": query,
                "matched_title": "",
                "authors": "",
                "publication_year": "",
                "isbn": "",
                "publisher": "",
                "cover_url": "",
                "match_rank": "",
                "status": "No results"
            })

            continue

        for rank, book in enumerate(books, start=1):

            title = book.get("title", "")
            authors = ", ".join(book.get("author_name", []))
            year = book.get("first_publish_year", "")

            isbns = book.get("isbn", [])
            isbn = isbns[0] if isbns else ""

            publishers = book.get("publisher", [])
            publisher = publishers[0] if publishers else ""

            cover_id = book.get("cover_i")

            cover_url = (
                f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
                if cover_id else ""
            )

            print(f"\nCandidate {rank}")
            print("Title:", title)
            print("Author:", authors)
            print("Year:", year)
            print("ISBN:", isbn)

            identified.append({
                "image": image,
                "ocr_text": query,
                "matched_title": title,
                "authors": authors,
                "publication_year": year,
                "isbn": isbn,
                "publisher": publisher,
                "cover_url": cover_url,
                "match_rank": rank,
                "status": "Candidate - needs verification"
            })

        time.sleep(1)

    # Save results
    fields = [
        "image",
        "ocr_text",
        "matched_title",
        "authors",
        "publication_year",
        "isbn",
        "publisher",
        "cover_url",
        "match_rank",
        "status"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(identified)

    print("\n" + "=" * 50)
    print("Book identification completed!")
    print("Total images:", len(grouped))
    print("Total candidate records:", len(identified))
    print("Saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
