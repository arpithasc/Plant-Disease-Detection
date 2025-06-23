# 🌱 Plant Disease Detection – Detailed Project Description

## 🧩 Problem Statement

Early and accurate identification of plant diseases is critical to prevent crop damage and economic losses in agriculture. Traditional methods are time-consuming, manual, and require expert intervention. This project automates disease detection using AI to assist farmers and agri-specialists.

---

## 🎯 Objectives

- Detect disease from plant leaf images using deep learning.
- Reduce need for manual inspection or agricultural lab testing.
- Provide a scalable tool for remote farming communities.

---

## 🏗️ System Architecture

1. **Image Input Module**  
   - Accepts plant leaf image (upload or capture).

2. **Preprocessing Pipeline**  
   - Applies resizing, normalization, and augmentation.
   - Extracts relevant features using OpenCV and TensorFlow preprocessing layers.

3. **CNN Model**  
   - Trained on PlantVillage dataset.
   - Classifies images into healthy or diseased categories.

4. **Result Display Interface**  
   - Built using Streamlit.
   - Shows predicted class, confidence level, and possible remedies.
  
   ## 🔮 Future Enhancements

- Real-time detection using mobile phone camera
- Integration with farmer advisory systems for treatment guidance
- Detection of soil/fruit diseases
- Multilingual plant health reports

---

## 🧠 Challenges Faced

- Overfitting due to limited data on certain diseases
- Class imbalance for rare diseases
- Lighting/angle variations in real-world photos
