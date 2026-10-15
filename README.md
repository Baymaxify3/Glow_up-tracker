# 🌟 Posture Glow-Up Tracker

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://glowup-tracker-uzcqxdjmndpvjkcfrucin7.streamlit.app)

An AI-powered web app that analyses your posture from a photo, tracks your progress over time, and builds a posture + nutrition plan to help you reach 100/100.

🔗 **Live demo:** https://glowup-tracker-uzcqxdjmndpvjkcfrucin7.streamlit.app

## ✨ Features

- **Posture Check:** upload a side-view photo (standing or sitting). MediaPipe pose estimation detects ear, shoulder and hip landmarks and measures:
  - Neck angle (forward head)
  - Upper-back hump (kyphosis)
  - Torso lean
  - Estimated neck load, and an overall score out of 100
- **Glow-Up Plan:** enter height, weight, age, gender and daily nutrients (Vitamin D, iron, calcium, magnesium, protein, water) to get a Glow Score.
- **Perfect Posture Blueprint:** see how far you are from 100/100, with a clinical-style exercise and nutrition prescription.
- **Progress Tracking:** score-over-time charts, last-7-day average, and Excel/CSV export.

## 🛠️ Tech Stack

Python, Streamlit, MediaPipe, OpenCV, NumPy, pandas, openpyxl

## 🚀 Run Locally

```bash
git clone https://github.com/Baymaxify3/Glow_up-tracker.git
cd Glow_up-tracker
pip install -r requirements.txt
streamlit run app.py
```

## 📸 How to Take a Good Photo

- Stand sideways to the camera with your full body visible
- Your ear, shoulder and hip must all be visible
- Use good lighting and a plain background

## 🗺️ Roadmap

- [ ] **Perfect Walk (Phase 4):** analyse a walking video for confidence and form
- [ ] Richer exercise plans
- [ ] Mobile app
- [ ] Wearable posture tag for real-time feedback

## ⚠️ Disclaimer

This is a student project for educational use. It is not a medical device and does not replace advice from a doctor or physiotherapist.

## 👤 Author

**Swetwaza Fatin**, BTech CSE, Gitanjali Institute of Technical Studies
