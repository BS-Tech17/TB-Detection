<h1 align="center">🫁 Deep Learning Tuberculosis Detection from Chest X-Rays</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MobileNetV2-TransferLearning-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MedicalImaging-ComputerVision-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Kaggle-Dataset-orange?style=for-the-badge"/>
</p>

---

<h2>📌 Overview</h2>

This project implements an AI-based diagnostic support system that detects **Tuberculosis (TB)** from chest X-ray images using transfer learning with MobileNetV2 in PyTorch.

The model is trained on a labeled dataset sourced from **Kaggle**, applying augmentation, validation monitoring, and confusion matrix evaluation to assess performance.

It demonstrates integration of:

* Deep Learning
* Medical Image Analysis
* Transfer Learning
* Model Deployment for Inference

The system is designed as an academic exploration of AI-assisted radiological screening workflows.

---

<h2>🏗 System Flow</h2>

Dataset Images
→ Preprocessing & Augmentation
→ MobileNetV2 Training
→ Evaluation Metrics
→ Model Save (.pth)
→ Interactive Prediction Loop

---

<h2>📁 Project Structure</h2>

```
main.py
tb_detection_pytorch_model.pth
README.md
```

---

<h2>🧰 Requirements</h2>

<h3>Hardware</h3>

* CPU (Minimum)
* GPU with CUDA support (Recommended)
* 8GB RAM or higher

<h3>Software</h3>

* Python 3.8+
* PyTorch
* Torchvision
* Pillow
* NumPy
* Scikit-learn

---

<h2>⚙️ Dependency Installation</h2>

Install required packages:

```
pip install torch torchvision numpy pillow scikit-learn
```

Verify installation:

```
python -c "import torch, torchvision, numpy, PIL, sklearn"
```

---

<h2>📊 Dataset Source</h2>

Chest X-ray data used for training and validation was obtained from **Kaggle** for academic experimentation.

Ensure dataset paths inside the script are updated to match your local storage before running.

---

<h2>▶️ Running the Model</h2>

1️⃣ Configure dataset paths in the script

2️⃣ Start training

```
python main.py
```

3️⃣ Model weights saved automatically

```
tb_detection_pytorch_model.pth
```

4️⃣ Enter image path for prediction

Type:

```
exit
```

to stop inference loop.

---

<h2>📈 Evaluation Output</h2>

* Epoch Loss
* Accuracy Tracking
* Training Confusion Matrix
* Validation Confusion Matrix
* Prediction Labels

---

<h2>⚠️ Do's & Don'ts</h2>

✔ Ensure dataset paths are correct
✔ Use GPU for faster training
✔ Maintain balanced dataset if possible
✔ Normalize images consistently
---

<h2>🚀 Future Enhancements</h2>

* Grad-CAM explainability visualization
* Web deployment using Streamlit
* Multi-disease classification
* ROC / AUC metric tracking
* Hyperparameter tuning automation

---

<p align="center">
Built for exploration in AI-driven Medical Imaging & Intelligent Diagnostic Systems
</p>
