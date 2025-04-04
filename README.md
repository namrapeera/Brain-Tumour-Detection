# Brain-Tumour-Detection
Brain tumour detection using ML and SVM#

This repository contains the implementation, research, and findings from my final year BSc (Hons) Computer Science (Artificial Intelligence) dissertation, titled:

**"An Intelligent Predictive Model for Brain Tumor Detection Using Deep Learning"**

## 📌 Project Overview

The project presents a predictive model that leverages Convolutional Neural Networks (CNNs) to detect brain tumors in MRI scans. The goal is to assist medical professionals by providing a reliable, automated, and scalable diagnosis aid.

> **Submitted:** December 2024  
> **Author:** Namra Riyaz Peera  
> **Institution:** Gulf College, in affiliation with Cardiff Metropolitan University

## 🎯 Objectives

- Develop a deep learning-based system to classify MRI images as tumorous or non-tumorous.
- Evaluate model performance using key metrics like accuracy, precision, recall, and F1-score.
- Compare results against traditional machine learning techniques.
- Enhance prediction reliability using preprocessing and data augmentation techniques.

## 🧪 Technologies Used

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebooks

## 🧠 Model Architecture

- Preprocessing: Image resizing, normalization
- Feature extraction: CNN layers with ReLU activation
- Classification: Fully connected layers with softmax
- Evaluation: Confusion matrix, classification report, and accuracy metrics

## 📊 Dataset

The dataset consists of MRI images sourced from publicly available medical image repositories and includes both healthy and tumor-affected scans.

> ⚠️ Due to ethical and privacy concerns, datasets used for training/testing are not included in this repo. You can access similar datasets from sources like Kaggle, Medical Segmentation Decathlon, or TCIA.

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection

