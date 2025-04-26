# Pneumonia Detection from Chest X-rays using AlexNet

This project implements a deep learning solution for pneumonia detection from chest X-ray images using the AlexNet convolutional neural network architecture. The model is trained and tested on the Chest X-Ray Pneumonia dataset from Kaggle.

## Overview

Pneumonia is a potentially serious infection that inflames the air sacs in one or both lungs. This project aims to detect pneumonia from chest X-ray images using deep learning techniques. The implementation uses AlexNet, a classic convolutional neural network architecture that has proven effective for image classification tasks.

## Google Colab

https://colab.research.google.com/drive/1Q4Q62APjnubuuT3ynGXWSYOt8IcoMea7?usp=sharing 

This project can be run directly in Google Colab. Click the link  above to open the notebook in Colab, or create a new Colab notebook and copy the provided code.

## Dataset

The project uses the Chest X-Ray Pneumonia dataset from Kaggle:
- Source: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- Contains chest X-ray images divided into two categories:
  - NORMAL: Healthy patients
  - PNEUMONIA: Patients with pneumonia

## Requirements

- Python 3.x
- TensorFlow 
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Kagglehub

Install dependencies with:
```
pip install tensorflow numpy matplotlib seaborn scikit-learn kagglehub
```

## Implementation Details

The implementation includes:

1. **Data Preparation**:
   - Loading and preprocessing chest X-ray images
   - Applying data augmentation techniques for training data
   - Computing class weights to handle class imbalance

2. **Model Architecture**:
   - AlexNet architecture with modifications for binary classification
   - 5 convolutional layers with max pooling
   - 2 dense layers with dropout (4096 units each)
   - Binary output layer with sigmoid activation

3. **Training**:
   - Binary cross-entropy loss
   - Adam optimizer with learning rate 1e-4
   - Batch size of 32
   - 10 training epochs
   - Class weight balancing

4. **Evaluation**:
   - Accuracy, precision, recall, F1-score metrics
   - Confusion matrix visualization
   - Sensitivity (recall for pneumonia class)
   - Specificity (recall for normal class)

## Performance Results

The model achieves:
- Overall Accuracy: 73%
- Sensitivity (Recall for PNEUMONIA): 99.74%
- Specificity (Recall for NORMAL): 27.78%

The model is highly sensitive to pneumonia cases but has lower specificity, meaning it tends to classify some normal cases as pneumonia (false positives).

## Project Structure

```
├── pneumonia_detection.py       # Main script for training and evaluation
├── pneumonia_alexnet.h5         # Saved model file after training
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

## Usage

1. **Download Dataset**:
   ```python
   path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
   ```

2. **Train the Model**:
   ```python
   history = train_model(model, epochs=10)
   ```

3. **Evaluate the Model**:
   ```python
   evaluate_model(model)
   ```

4. **Make Predictions**:
   ```python
   # Load a saved model
   loaded_model = load_model('pneumonia_alexnet.h5')
   
   # Predict on a single image
   prediction = loaded_model.predict(sample_image_batch)
   ```

## Visualizations

The implementation includes visualizations for:
- Training and validation accuracy/loss curves
- Confusion matrix
- Sample predictions on test images
- Comparison with other architectures (e.g., ResNet)

## Limitations and Future Work

- The model shows high sensitivity but lower specificity
- Future work could focus on:
  - Improving specificity through additional training data or techniques
  - Testing other architectures like DenseNet or EfficientNet
  - Implementing explainable AI techniques to highlight relevant image regions
  - Exploring transfer learning with models pre-trained on medical imaging datasets

## Acknowledgments

- The Chest X-Ray dataset is from Kaggle by Paul Mooney
- The AlexNet architecture is based on the paper by Krizhevsky et al. (2012)

