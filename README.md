# Multimodal ML Pipeline: Digit Classification & Animal Verification

This repository contains a two-part Machine Learning assignment focusing on Computer Vision and Natural Language Processing.

* **Task 1:** Handwritten digit classification (MNIST) using three different architectures: Convolutional Neural Networks (CNN), Feed-Forward Neural Networks (FFNN), and Random Forest.
* **Task 2:** A multimodal verification pipeline that extracts animal entities from text using a Transformer-based NER model and verifies their presence in an image using a ResNet18 classifier.

---

## Project Structure

```text
.
├── task1/                   # MNIST Classification
│   ├── models/              # Model architectures (CNN, FFNN, RF)
│   ├── classifier.py        # Unified interface for digit classification
│   ├── demo.ipynb           # Gradio web interface for real-time testing
│   └── *_train.py           # Training scripts
├── task2/                   # Animal Verification Pipeline
│   ├── models/              # Inference and training for NER and CV
│   ├── pipeline.py          # Main multimodal verification script
│   └── Exploratory_Data_Analysis_(10animals).ipynb
├── utils/                   # Shared utilities
│   ├── data_loader.py       # MNIST data loading and augmentation
|   ├── data_distribution.py # Calculates classes weight in dataset
│   └── setup_weights.py     # Automated script for downloading models
├── weights/                 # Models weigths
├── requirements.txt         # Project dependencies
└── README.md
```

---

## Installation & Setup

### 1. Clone the Repository
 
```bash
# Clone
git clone https://github.com/tallchildvi/test-assignment.git

# Navigate into the project directory
cd test-assignment
```

### 2. Environment Setup

It is recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
# For PowerShell:
venv\Scripts\Activate.ps1
# For cmd:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Downloading Model Weights

Due to file size restrictions, trained model weights are stored on Google Drive.

```bash
# Run the weight setup script
python -m utils.setup_weights
```
---

## Usage Documentation

### Task 1: MNIST Digit Classification

Test the digit classifiers (CNN, FFNN, RF) using the interactive Gradio demo.

1. Open `task1/demo.ipynb` in a Jupyter environment.
2. Launch the Gradio Sketchpad to draw digits and see real-time predictions.

---

### Task 2: Multimodal Verification Pipeline

The pipeline extracts animal names from text and checks if they match the animal detected in an image.

**Run via Command Line:**

```bash
python -m task2.pipeline --text "There is a butterfly on the flower" --image "task2/test_image.jpg"
```

**Key Components:**

- **NER:** Extracts unique animal entities  using a fine-tuned DistilBERT model.
- **CV:** Classifies the image into one of 10 animal categories using ResNet18.
- **Verification:** Returns `True` if the detected image animal is mentioned in the text and `False` otherwise.

