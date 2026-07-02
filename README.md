# AI for Medicine: Chest X-Ray Classification

This repository contains a reproducible deep learning workflow for classifying chest X-ray images into four categories using the COVID-19 Radiography Database.

The project compares a custom CNN baseline with a MobileNet transfer-learning model. Model selection is performed with stratified 5-fold cross-validation on the development set, and the selected model is evaluated once on an independent hold-out test set.

## Target Classes

- COVID
- Lung_Opacity
- Normal
- Viral Pneumonia

## Repository Contents

```text
.
|-- chest_xray_classification.ipynb
|-- README.md
|-- report.pdf
`-- report_assets/
```

## Dataset

The notebook downloads the COVID-19 Radiography Database at runtime with `kagglehub`:

```text
tawsifurrahman/covid19-radiography-database
```

The dataset is not stored in this repository.

## Methodology

1. Install and import dependencies.
2. Download the Kaggle dataset.
3. Inspect class folders and class distribution.
4. Verify image-mask consistency.
5. Flag corrupted or unreadable images.
6. Remove exact duplicate images before splitting.
7. Create a stratified 80/20 development/test split.
8. Train a custom CNN baseline with 5-fold cross-validation.
9. Train MobileNet with the same 5-fold cross-validation protocol.
10. Select the model with the strongest development-set mean validation accuracy.
11. Retrain the selected model on the full development set.
12. Evaluate once on the independent hold-out test set.
13. Generate a classification report, confusion matrix, ROC/AUC curves, and balanced SHAP explanations.

## Data Quality Summary

| Check | Result |
|---|---:|
| Usable image records before duplicate removal | 21,165 |
| Missing masks | 0 |
| Extra masks without matching images | 0 |
| Corrupted/unopenable images | 0 |
| Exact duplicate groups | 42 |
| Duplicate images removed | 54 |
| Clean dataset size | 21,111 |
| Development set size | 16,888 |
| Hold-out test set size | 4,223 |

## Model Comparison

Development-set 5-fold cross-validation results:

| Model | Mean Validation Accuracy | Std. Accuracy | Mean Validation Loss |
|---|---:|---:|---:|
| CNN baseline | 0.6943 | 0.0083 | 0.7528 |
| MobileNet | 0.8840 | 0.0126 | 0.3199 |

MobileNet achieved the strongest cross-validation performance and was selected for final evaluation.

## Hold-Out Test Results

| Metric | Value |
|---|---:|
| Selected model | MobileNet |
| Test accuracy | 0.8972 |
| Test loss | 0.2679 |
| Macro F1-score | 0.9022 |
| Weighted F1-score | 0.8963 |
| Macro AUC | 0.9836 |

## Class-Level Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| COVID | 0.8935 | 0.8810 | 0.8872 | 714 |
| Lung_Opacity | 0.9166 | 0.8130 | 0.8617 | 1,203 |
| Normal | 0.8839 | 0.9450 | 0.9134 | 2,038 |
| Viral Pneumonia | 0.9377 | 0.9552 | 0.9464 | 268 |

## ROC/AUC Results

| Class | AUC |
|---|---:|
| COVID | 0.9869 |
| Lung_Opacity | 0.9751 |
| Normal | 0.9730 |
| Viral Pneumonia | 0.9993 |

## Explainability

The notebook includes a SHAP analysis using one hold-out test image from each class. This makes the explanation sample class-balanced instead of showing only the first images from the test dataset.

SHAP is included as a qualitative inspection tool. It should not be interpreted as clinical evidence.

## Main Findings

- MobileNet substantially outperformed the custom CNN baseline.
- Final hold-out test accuracy was 0.8972.
- Macro-average AUC was 0.9836, indicating strong one-vs-rest separability.
- Lung_Opacity had the lowest recall at 0.8130 and remains the main class-level weakness.
- Normal and Lung_Opacity confusion may be partly related to visual similarity or inconsistent image framing. This should be tested with lung-mask cropping or lung-only masking.

## How to Run

Recommended environment: Google Colab with GPU.

1. Open Google Colab.
2. Upload `chest_xray_classification.ipynb`.
3. Select `Runtime -> Change runtime type -> GPU`.
4. Run the notebook from top to bottom.

The first code cell installs notebook-only dependencies:

```python
!pip -q install kagglehub shap
```

## Key Dependencies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Pillow
- KaggleHub
- SHAP

## Limitations and Future Work

- The model was evaluated on a single public dataset and is not clinically validated.
- External validation on independent hospital or multi-center datasets is required.
- Lung_Opacity recall should be improved and monitored carefully.
- A future experiment should train and evaluate a lung-mask-cropped or lung-only-masked version of the dataset.
- Additional transfer-learning models such as EfficientNet, DenseNet, and ResNet can be tested.
- Clinical deployment would require calibration analysis, expert review, prospective validation, and regulatory evaluation.

## Disclaimer

This project is for educational and research purposes only. It is not a medical device and must not be used as a substitute for professional clinical judgment.
