# COVID-19 Detection with Convolutional Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org)

Deep learning model that classifies chest X-ray images to detect COVID-19 using a custom CNN architecture.

## Overview

Trained on the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle, this model distinguishes between COVID-19 positive, viral pneumonia, and normal lung X-rays.

## Model Architecture

- Multiple convolutional layers with ReLU activation
- Max pooling for dimensionality reduction
- Dropout for regularization
- Fully connected layers for classification
- Optimized with stochastic gradient descent

## Results

The model achieves reliable multi-class classification across the three categories, demonstrating the potential of deep learning in medical imaging diagnostics.

## Setup

```bash
pip install tensorflow numpy pandas matplotlib opencv-python jupyter
jupyter notebook COVID.ipynb
```

## Dataset

Download from Kaggle: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

---

*Built as part of exploring applied deep learning in healthcare diagnostics.*
