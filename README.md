# AI for Medicine: Chest X-Ray Classification

This project develops and evaluates deep learning models for classifying chest X-ray images into four diagnostic categories using the COVID-19 Radiography Database.

## Project Objective

The goal is to compare a custom CNN baseline with a transfer-learning MobileNet model and select the stronger model using development-set cross-validation before evaluating once on an independent hold-out test set.

Target classes:

- COVID
- Lung Opacity
- Normal
- Viral Pneumonia

## Repository Contents

```text
.
├── chest_xray_classification.ipynb
└── README.md
```

## Dataset

The notebook downloads the COVID-19 Radiography Database directly with `kagglehub`:

```text
tawsifurrahman/covid19-radiography-database
```

The dataset is not stored in this repository. It is downloaded at runtime.

## Methodology

The notebook follows this workflow:

1. Install and import required libraries.
2. Download the dataset from Kaggle.
3. Inspect dataset folders and class distribution.
4. Check image-mask consistency.
5. Flag corrupted images.
6. Remove exact duplicate images using SHA-256 hashing.
7. Create a stratified 80/20 development/test split.
8. Train and evaluate a CNN baseline with 5-fold cross-validation.
9. Train and evaluate MobileNet with the same 5-fold cross-validation protocol.
10. Select the best model using development-set mean validation accuracy.
11. Train the selected model on the full development set.
12. Evaluate once on the independent hold-out test set.
13. Report classification metrics, confusion matrix, ROC/AUC, and SHAP explanations.

## Data Quality Summary

Latest executed run:

| Check | Result |
|---|---:|
| Original image records | 21,165 |
| Missing masks | 0 |
| Extra masks | 0 |
| Corrupted images | 0 |
| Exact duplicate groups | 42 |
| Duplicate images removed | 54 |
| Clean dataset size | 21,111 |
| Development set size | 16,888 |
| Hold-out test set size | 4,223 |

## Model Comparison

Development-set 5-fold cross-validation results:

| Model | Mean Validation Accuracy | Std | Mean Validation Loss |
|---|---:|---:|---:|
| CNN baseline | 0.6950 | 0.0085 | 0.7535 |
| MobileNet | 0.8830 | 0.0119 | 0.3204 |

MobileNet was selected as the final model.

## Final Test Results

Hold-out test-set performance:

| Metric | Value |
|---|---:|
| Test accuracy | 0.8965 |
| Test loss | 0.2734 |
| Macro F1-score | 0.9033 |
| Weighted F1-score | 0.8955 |
| Macro AUC | 0.9833 |

Class-level performance:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| COVID | 0.8923 | 0.8936 | 0.8929 | 714 |
| Lung Opacity | 0.9310 | 0.7963 | 0.8584 | 1,203 |
| Normal | 0.8745 | 0.9504 | 0.9109 | 2,038 |
| Viral Pneumonia | 0.9583 | 0.9440 | 0.9511 | 268 |

ROC/AUC:

| Class | AUC |
|---|---:|
| COVID | 0.9876 |
| Lung Opacity | 0.9750 |
| Normal | 0.9711 |
| Viral Pneumonia | 0.9993 |

## How to Run

Recommended environment: Google Colab with GPU.

1. Open Google Colab.
2. Upload `chest_xray_classification.ipynb`.
3. Select `Runtime -> Change runtime type -> GPU`.
4. Run the notebook from top to bottom.

The first code cell installs notebook dependencies:

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

## Notes on Explainability

The notebook includes a small SHAP analysis for model inspection. This is intended as a qualitative explainability aid and should not be interpreted as clinical evidence.

## Limitations

- This project uses one public dataset and is not externally validated.
- Results should not be used for medical diagnosis or clinical decision-making.
- Additional validation on independent datasets is required.
- Class imbalance remains a relevant limitation.
- Clinical deployment would require expert review, calibration analysis, and prospective evaluation.

## Disclaimer

This project is for educational and research purposes only. It is not a medical device and must not be used as a substitute for professional clinical judgment.
