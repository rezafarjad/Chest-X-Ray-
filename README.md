# Chest X-Ray Classification Using Deep Learning

<p align="center">
  <img src="figures/sample_xray.png" width="500">
</p>

<p align="center">
  <b>Automated Classification of COVID-19, Lung Opacity, Normal, and Viral Pneumonia Chest X-Ray Images Using Deep Learning</b>
</p>

---

## Overview

This project investigates the use of deep learning for automated chest X-ray classification. A custom Convolutional Neural Network (CNN) and a transfer learning approach based on MobileNet were implemented and compared using stratified 5-fold cross-validation.

The final MobileNet model achieved:

* **89.15% Test Accuracy**
* **97.95% Macro AUC**
* Strong classification performance across all four disease categories

---

## Key Results

| Model     |                  Accuracy |
| --------- | ------------------------: |
| CNN       |                    67.65% |
| MobileNet | 87.96% (Cross-Validation) |
| MobileNet |         89.15% (Test Set) |

---

## Confusion Matrix

<p align="center">
  <img src="figures/confusion_matrix.png" width="700">
</p>

The confusion matrix shows strong classification performance across all classes, with most predictions concentrated along the diagonal.

---

## ROC Curves and AUC

<p align="center">
  <img src="figures/roc_curve.png" width="700">
</p>

| Class             |        AUC |
| ----------------- | ---------: |
| COVID-19          |     0.9782 |
| Lung Opacity      |     0.9689 |
| Normal            |     0.9715 |
| Viral Pneumonia   |     0.9991 |
| **Macro Average** | **0.9795** |

---
