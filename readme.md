# 🚗 Car Damage Detection using Streamlit

This project is a **Streamlit-based web application** that classifies car damage types using a trained deep learning model. It helps identify and label the type and location of vehicle damage from uploaded images.

---

## 📁 Project Structure
streamlit_app/
├── app.py # Main Streamlit application
├── model_helper.py # Contains model loading and prediction functions
├── model/ # Directory containing the saved PyTorch model
├── requirements.txt # Python dependencies
├── .gitignore # Files and folders ignored by Git
└── readme.md # Project documentation (this file)




---

## 📸 Classes Detected

The model classifies images into the following 6 categories:

- `Front Breakage`
- `Front Crushed`
- `Front Normal`
- `Rear Breakage`
- `Rear Crushed`
- `Rear Normal`

---

## 🧠 Model Details

- **Model Used**: ResNet50 (Transfer Learning)
- **Framework**: PyTorch
- **Input Format**: RGB images
- **Output**: Predicted damage class

---

## 🚀 How to Run
```bash
streamlit run app.py
```

### 🔧 Setup

Install the required dependencies:

```bash
pip install -r requirements.txt

```
## Requirements
streamlit==1.42.2
torch==2.5.1+cu121
torchvision==0.20.1+cu121
Pillow==11.3.0
