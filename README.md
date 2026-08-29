# Nutmeg Harvest NCD4 – Nutmeg Condition Classification using MobileNetV2

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Accuracy 99.17%](https://img.shields.io/badge/Test_Accuracy-99.17%25-brightgreen.svg)]()

A complete Deep Learning image-classification system for classifying the harvest stage and condition of nutmeg into 4 distinct categories using **MobileNetV2 Transfer Learning**.

---

## 📌 Problem Statement

Manual visual inspection of harvested nutmeg fruits is labor-intensive, subjective, and prone to human error. Automatically identifying the harvest condition (whether nutmeg is ready for harvest, not harvested, decayed, or intermediate) using computer vision enables automated sorting, quality control, and agricultural yield optimization.

---

## 🎯 Project Objective

Build an end-to-end Deep Learning image classification pipeline using **MobileNetV2** that takes a nutmeg fruit image and classifies it into one of four classes:

1. **Harvested** (`H`)
2. **Not Harvested** (`NH`)
3. **Decayed** (`D`)
4. **Intermediate** (`I`)

Example model output:

```text
Predicted Class : Harvested
Confidence      : 99.92%

Class Probabilities
-------------------
Harvested       : 99.92%
Not Harvested   : 0.00%
Decayed         : 0.06%
Intermediate    : 0.02%
```

---

## 📊 Dataset Specification

- **Total Images**: 800 RGB images
- **Number of Classes**: 4 classes
- **Images per Class**: 200 images per class
  - `Harvested` (`Harvest_IMG`): 200 images
  - `Not Harvested` (`Not_Harvested_IMG`): 200 images
  - `Decayed` (`Decayed_IMG`): 200 images
  - `Intermediate` (`Intermediate_IMG`): 200 images
- **Dataset Split**: Stratified **70% Training** (560 images), **15% Validation** (120 images), and **15% Testing** (120 images).

> ⚠️ **Note**: The 800-image dataset is stored locally and is excluded from the Git repository via `.gitignore`.

---

## ⚙️ Methodology

```text
Image (224 x 224 x 3)
         ↓
MobileNetV2 Preprocessing
         ↓
MobileNetV2 Base (ImageNet Weights)
         ↓
GlobalAveragePooling2D
         ↓
Dropout (0.3) -> Dense (128, ReLU) -> Dropout (0.2)
         ↓
4-Class Softmax Output
```

The training process follows a two-stage strategy:
1. **Stage 1 (Base Frozen)**: Train the top classification head with the MobileNetV2 base frozen using Adam optimizer (`lr=1e-3`).
2. **Stage 2 (Fine-Tuning)**: Unfreeze deeper layers of MobileNetV2 base and train with a very small learning rate (`lr=1e-5`) alongside EarlyStopping and ReduceLROnPlateau callbacks.

---

## 🛠️ Technologies Used

- **Language**: Python 3.12
- **Deep Learning Framework**: TensorFlow / Keras
- **Model Backbone**: MobileNetV2
- **Data Manipulation & Analysis**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib
- **Image Processing**: Pillow (PIL), OpenCV

---

## 📁 Project Structure

```text
Nutmeg_Harvest_NCD4/
│
├── Frames/                      # Dataset subfolders (Local only, ignored by git)
│   └── M1/
│       ├── Harvest_IMG/
│       ├── Not_Harvested_IMG/
│       ├── Decayed_IMG/
│       └── Intermediate_IMG/
│
├── src/
│   ├── dataset.py               # Dataset verification, path resolution, corrupt check, stratified split
│   ├── model.py                 # MobileNetV2 architecture build & fine-tuning layer unfreezing
│   ├── train.py                 # 2-Phase training pipeline & callbacks
│   ├── evaluate.py              # Unseen test set evaluation & report generation
│   ├── predict.py               # CLI single-image prediction script
│   └── utils.py                 # JSON class mapping & plotting utilities
│
├── models/
│   ├── nutmeg_mobilenetv2.keras # Saved best trained model
│   ├── class_indices.json       # JSON mapping for class index resolution
│   └── test_split.json          # Saved test set paths for reproducible evaluation
│
├── results/
│   ├── accuracy.png             # Training & Validation Accuracy curve
│   ├── loss.png                 # Training & Validation Loss curve
│   ├── confusion_matrix.png     # Test set confusion matrix plot
│   └── classification_report.txt# Detailed evaluation metrics
│
├── requirements.txt             # Required Python packages
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <GITHUB_REPOSITORY_URL>
   cd Nutmeg_Harvest_NCD4
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training

To verify the dataset, create stratified splits, train the MobileNetV2 classification head, fine-tune deeper layers, and save the best model:

```bash
python src/train.py
```

Generated plots are saved automatically to `results/accuracy.png` and `results/loss.png`.

---

## 📈 Evaluation

To evaluate the trained model on the completely unseen 15% test dataset (120 images):

```bash
python src/evaluate.py
```

---

## 🔮 Prediction

To classify a single nutmeg image:

```bash
python src/predict.py --image "path/to/your/image.jpg"
```

---

## 📊 Empirical Training & Test Results

### Unseen Test Set Metrics (120 images)

- **Overall Test Accuracy**: **99.17%**
- **Macro F1-Score**: **0.9919**
- **Weighted F1-Score**: **0.9917**

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Harvested** | 1.0000 | 1.0000 | 1.0000 | 30 |
| **Not Harvested** | 0.9677 | 1.0000 | 0.9836 | 30 |
| **Decayed** | 1.0000 | 1.0000 | 1.0000 | 30 |
| **Intermediate** | 1.0000 | 0.9667 | 0.9831 | 30 |
| **Overall Accuracy** | | | **0.9917** | **120** |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
