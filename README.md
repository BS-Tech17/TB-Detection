🩺 Overview
This repository implements a deep learning-based solution for detecting tuberculosis (TB) in chest X-ray images using MobileNetV2 and PyTorch. MobileNetV2, a lightweight and efficient convolutional neural network, is well-suited for medical image analysis tasks, making it an ideal choice for this project.

The model is trained to classify X-ray images as either:

Tuberculosis Positive
Tuberculosis Negative
🚀 Features
Lightweight and efficient architecture using MobileNetV2.
High accuracy in detecting TB from chest X-ray images.
Customizable training pipeline for further fine-tuning.
Data augmentation for improved generalization.
Detailed metrics: confusion matrix, precision, recall, F1-score, and accuracy.
📂 Project Structure
bash
Copy code
├── data
│   ├── train
│   │   ├── TB_Positive
│   │   ├── TB_Negative
│   ├── val
│   │   ├── TB_Positive
│   │   ├── TB_Negative
│   ├── test
│       ├── TB_Positive
│       ├── TB_Negative
├── src
│   ├── tb.py            # Defines the Python code for the model

🔧 Setup Instructions
Clone the Repository

bash
Copy code
git clone 7https://github.com/BS-Tech17/TB-Detection.git
cd tuberculosis-TB-Detection

📊 Results and Metrics
The following metrics are calculated during evaluation:

Confusion Matrix
Accuracy
Precision
Recall
F1-Score
