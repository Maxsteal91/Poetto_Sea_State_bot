# 🌊 Poetto Sea State Bot

**Poetto Sea State Bot** is a Telegram bot that uses a deep learning model to estimate the sea state at Poetto Beach (Cagliari, Italy) from images or short videos.
It acts as an automated assistant that analyzes sea conditions and provides real-time feedback directly via Telegram.

---

## 🧠 Model

The bot is powered by a 3D ResNet-18 model trained on real video footage from the Poetto coastline.
Because GitHub limits files to 100 MB, the model is not included in this repository.

👉 **Download the model here:**
[🔗 r3d_18_poetto.pth (Google Drive)](https://drive.google.com/file/d/1ASd7VdbSrXXs9fa3EpHFed0W2NlUugq9/view?usp=sharing)

Once downloaded, place the model file in folder `model`

---

## 📹 Data Source

The images and videos used to train and test the model come from **live webcam footage of Poetto Beach (Cagliari)** provided by:

🌐 [https://panoramicams.com/poetto-cagliari/](https://panoramicams.com/poetto-cagliari/)
