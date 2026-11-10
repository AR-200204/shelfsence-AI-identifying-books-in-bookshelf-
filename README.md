"# shelfsence-AI-identifying-books-in-bookshelf-" 
# ShelfSense AI – AI-Powered Book Detection and Identification

ShelfSense AI is a computer vision project that aims to automate book identification from library bookshelf images.

The system detects individual books on a bookshelf, crops the detected books, extracts text from their spines using OCR, and searches for possible book records using Open Library.

The project combines object detection, Optical Character Recognition (OCR), and book metadata search into one pipeline.

## Features

- **Book Detection:** Uses a custom-trained YOLO11n model to detect books on library shelves.
- **Automatic Cropping:** Extracts individual book images from detected bookshelf regions.
- **Text Recognition:** Uses PaddleOCR to read text from book spines.
- **Book Identification:** Searches Open Library for possible matching book records.
- **Candidate Comparison:** Compares OCR text with available book metadata to rank possible matches.
- **Evaluation:** Includes scripts for comparing identification versions and evaluating results.

> Note: Book identification is an experimental process. OCR errors, incomplete spine text, and similar book titles can produce incorrect or ambiguous matches. Candidate results should be verified.

## Project Pipeline

```text
Bookshelf Image
      |
      v
YOLO11n Book Detection
      |
      v
Individual Book Cropping
      |
      v
PaddleOCR Text Recognition
      |
      v
Open Library Search
      |
      v
Candidate Ranking
      |
      v
Book Identification Results
```

## Technologies Used

- Python
- PyTorch
- Ultralytics YOLO11n
- PaddleOCR
- PaddlePaddle
- Open Library API
- OpenCV
- Pandas

## Dataset

This project uses the **Open Shelves Dataset – Version 9**, published by capjamesg on Roboflow Universe.

**Dataset information:**

- Dataset: Open Shelves
- Version: 9
- Total images: 496
- Training images: 446
- Validation images: 35
- Testing images: 15
- Class: `book`
- License: CC BY 4.0

### Download the dataset

1. Visit the [Open Shelves Dataset – Version 9](https://universe.roboflow.com/capjamesg/open-shelves/dataset/9).
2. Create or sign in to your Roboflow account if required.
3. Download the dataset.
4. Select the YOLO-compatible export format used by this project.
5. Extract the downloaded dataset into the project directory.

Expected directory structure:

```text
ShelfSense-AI/
│
├── dataset/
│   └── external/
│       └── Open Shelves.v9i.yolov8/
│           ├── train/
│           │   ├── images/
│           │   └── labels/
│           ├── valid/
│           │   ├── images/
│           │   └── labels/
│           ├── test/
│           │   ├── images/
│           │   └── labels/
│           └── data.yaml
│
├── train.py
├── test_model.py
├── crop_books.py
├── read_books_paddle_v2.py
├── identify_books_v4_6_1.py
└── README.md
```

The dataset is not stored directly in this GitHub repository. Download it from the original source to reproduce the experiments.

**Dataset attribution:** Open Shelves Dataset by capjamesg, Roboflow Universe, licensed under CC BY 4.0.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AR-200204/shelfsence-AI-identifying-books-in-bookshelf-.git
cd shelfsence-AI-identifying-books-in-bookshelf-
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install ultralytics opencv-python pandas numpy
```

Install PaddlePaddle and PaddleOCR using the official installation instructions for your operating system and hardware:

- [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/install/quick)
- [PaddleOCR Documentation](https://www.paddleocr.ai/)

GPU installation requires compatible hardware, drivers, and package versions. CPU execution may also be possible.

## Running the Project

### 1. Check the dataset

After downloading and extracting the dataset, run:

```bash
python check_dataset.py
```

### 2. Train the book detection model

```bash
python train.py
```

Make sure the dataset path in the training configuration points to your downloaded `data.yaml` file.

### 3. Test the trained model

```bash
python test_model.py
```

Make sure the trained model path and test image path are configured correctly.

### 4. Crop detected books

```bash
python crop_books.py
```

### 5. Recognize book spine text

```bash
python read_books_paddle_v2.py
```

### 6. Identify books

```bash
python identify_books_v4_6_1.py
```

Check each script's configuration and input/output paths before running it. Some scripts represent earlier experimental versions of the pipeline.

## Model Training

The project uses YOLO11n for book detection.

Training configuration used in the initial experiment:

- Model: YOLO11n
- Epochs: 30
- Image size: 416 × 416
- Dataset: Open Shelves v9
- Detection class: Book

The trained model weights are not included in this repository. Developers can train their own model using the dataset and training script.

## Contributing

Contributions and improvements are welcome!

You can contribute by:

- Improving book detection accuracy.
- Improving OCR accuracy on book spines.
- Developing better book identification and matching algorithms.
- Testing the model with different bookshelf images.
- Adding support for additional book metadata sources.
- Improving the overall pipeline.

## Future Improvements

- Support for multiple book detection models.
- Better recognition of rotated and partially hidden book spines.
- Improved OCR and text correction.
- More reliable book identification and duplicate detection.
- Library inventory management integration.
- Web and mobile application integration.

## Acknowledgements

- [Roboflow Universe – Open Shelves Dataset](https://universe.roboflow.com/capjamesg/open-shelves)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Open Library](https://openlibrary.org/)

## License

The Open Shelves dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

Check the licenses of the software, pretrained weights, and other dependencies before redistributing or using the complete project commercially.