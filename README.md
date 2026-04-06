# 🥔 Potato Leaf Disease Classifier Model

> An AI-powered web application that detects potato leaf diseases using deep learning (EfficientNetB7).

![Project Interface](images/webpage.png)

---

## 🤖 What is this Project?

This project is a smart AI system that helps detect diseases in potato leaves automatically using image classification.

Instead of manual inspection, users can simply upload a leaf image, and the model will predict whether the plant is healthy or diseased.

---

## ✨ Features

- **Deep Learning Model** — Built using EfficientNetB7 (state-of-the-art CNN)  
- **3-Class Classification**:
  - Early Blight  
  - Late Blight  
  - Healthy Leaf  
- **Simple Web Interface** — Upload and get instant results  
- **Image Preview** — Displays uploaded image with prediction  
- **Fast & Accurate Predictions**  

---

## 🧠 How It Works

1. User uploads a potato leaf image  
2. Image is preprocessed (resize, normalize)  
3. Passed to trained EfficientNetB7 model  
4. Model predicts the disease category  
5. Result is displayed with the uploaded image  

---

## 🖼️ Sample Results

### 🔹 Upload Interface
![Upload Page](images/upload.png)

### 🔹 Prediction Result
![Result Page](images/late_blight.png)

---

## 🌿 Disease Categories

### 1️⃣ Early Blight
![Early Blight](images/early_blight.png)

### 2️⃣ Late Blight
![Late Blight](images/late_blight.png)

### 3️⃣ Healthy Leaf
![Healthy Leaf](images/healthy.png)

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|-----------|
| Backend | Python, Flask |
| Model | EfficientNetB7 |
| AI Framework | TensorFlow / Keras |
| Frontend | HTML, CSS |
| Image Handling | OpenCV / PIL |

---

## 📁 Project Structure

project/
├── static/
│ └── uploads/ # Uploaded images
├── templates/ # HTML files
├── model/ # Trained model
├── app.py # Flask app
└── README.md


---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/potato-disease-classifier.git
cd potato-disease-classifier

2. Install Dependencies
pip install flask tensorflow keras pillow numpy


3. Run the Application
python app.py

4. Open in Browser
http://localhost:5000
