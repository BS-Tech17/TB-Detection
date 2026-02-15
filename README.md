# 🫁 Tuberculosis Detection from Chest X-Rays using Deep Learning

## 📌 Overview

This project implements a deep learning pipeline for detecting **Tuberculosis (TB)** from chest X-ray images using transfer learning with MobileNetV2 in PyTorch.
It includes preprocessing, training, validation, evaluation via confusion matrices, and an interactive inference loop for real-time prediction.

The system demonstrates how AI can assist in medical image screening workflows by identifying TB-related patterns in radiographic scans.

---

## 🎯 Objectives

* Build a binary classifier (TB vs Normal)
* Use transfer learning for improved performance
* Evaluate model accuracy and confusion matrices
* Enable prediction on unseen X-ray images
* Save and reload trained models for reuse

---

## 🧰 Software Requirements

* Python 3.8+
* PyTorch
* Torchvision
* NumPy
* Pillow
* Scikit-learn

### Installation

```bash
pip install torch torchvision numpy pillow scikit-learn
```

---

## 💻 Hardware Requirements

* CPU (minimum)
* GPU with CUDA support (recommended)
* 8GB RAM or higher

The script automatically selects GPU if available.

---

## 📂 Dataset Structure

```
xyz/
│
├── test/
│   ├── Normal/
│   └── Tuberculosis/
│
├── val/
│   ├── Normal/
│   └── Tuberculosis/
```

Each class folder should contain labeled chest X-ray images.

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing

* Resize images to 224×224
* Augmentation:

  * Horizontal flip
  * Rotation
  * Color jitter
* Normalize using ImageNet statistics

### 2️⃣ Model Architecture

* Backbone: MobileNetV2 (pretrained)
* Modified classifier:

```
Linear Layer → 1 Output Node
```

### 3️⃣ Training Setup

* Loss Function: Binary Cross Entropy with Logits
* Optimizer: Adam
* Epochs: 10
* Batch Size: 32

### 4️⃣ Evaluation

* Accuracy tracking
* Confusion matrix computation
* Validation loss monitoring

### 5️⃣ Inference

* Model weights saved to `.pth`
* Reloaded for predictions
* Interactive CLI input loop

---

## ▶️ Execution Steps

### Train the model

```bash
python main.py
```

### Output

* Epoch metrics printed
* Model saved as:

```
tb_detection_pytorch_model.pth
```

### Run Predictions

After training, the script prompts:

```
Enter the path of the X-ray image
```

Type `exit` to stop.

---

## 📊 Output Metrics

* Training Accuracy
* Validation Loss
* Confusion Matrices
* Prediction Results

---

## 🔬 Applications

* AI-assisted radiology research
* Medical imaging experimentation
* Academic demonstrations

---

## 🚧 Limitations

* Depends on dataset quality
* Binary classification only
* Not clinically validated

---

## 🔮 Future Enhancements

* Web deployment (Streamlit)
* Explainability via Grad-CAM
* Multi-disease classification
* ROC/AUC metrics
* Hyperparameter tuning

---

## ⚠️ Disclaimer

This project is intended strictly for educational and research purposes.
It must not be used for real medical diagnosis.

---

## 👩‍💻 Author

**Bhoomika Saxena**
B.Tech — CSE (IoT & Intelligent Systems)
AI • IoT • Computer Vision
