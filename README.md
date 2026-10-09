#  Ayurvedic Medicine Recommendation System

A web app that predicts a likely disease from the symptoms you select and suggests matching **Ayurvedic medicines** based on your age, gender and severity level. It uses a **Random Forest** classifier served through a **Flask** API, with a responsive HTML/CSS/JavaScript front end.

>  **Disclaimer:** This project is for educational purposes only. It is not a medical diagnosis tool and does not replace advice from a qualified doctor or Ayurvedic practitioner.

##  About

Many people look for natural remedies but don't know which ones suit their condition. This system takes a user's symptoms, predicts the most probable disease with a machine learning model, and then looks up Ayurvedic recommendations from a curated dataset, filtered by the user's profile.

##  Features

- **Symptom-based disease prediction** using a Random Forest classifier
- **Personalised recommendations** filtered by age group, gender and severity
- **Ayurvedic medicine suggestions** with dosage information where available
- **Model accuracy shown** alongside each prediction
- **Clean, responsive interface** built with HTML, CSS and JavaScript
- **REST API** (`/predict`) that can be reused by other apps

##  How It Works

1. The user selects symptoms and enters age, gender and severity.
2. The symptoms are converted into a binary feature vector (1 = present, 0 = absent).
3. The trained Random Forest model predicts the disease.
4. Age is mapped to a group: **child** (≤ 12), **adult** (13–59) or **elderly** (60+).
5. The medicine dataset is filtered by disease, age group, gender and severity.
6. The disease, model accuracy and up to two recommended medicines are returned and displayed.

##  Model Details

| | |
|---|---|
| **Algorithm** | Random Forest Classifier (scikit-learn) |
| **Trees / max depth** | 100 / 5 |
| **Min samples per leaf** | 5 (to reduce overfitting) |
| **Validation** | 80/20 stratified train-test split and 5-fold cross-validation |
| **Data cleaning** | Missing-target rows dropped, leaky features removed (correlation > 0.99) |
| **Saved artifacts** | Trained model, label encoder and feature names (`.pkl`) |

The repo also includes experiments with other classifiers (Naive Bayes, SVM) and an evaluation script for comparing results. Test and cross-validation accuracy are saved in `accuracy.txt`.

##  Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | Python, scikit-learn, pandas, NumPy, joblib |
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |

##  Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/MSushmitha02/Aayurvedic-Medicine-Recommendation-System.git
cd Aayurvedic-Medicine-Recommendation-System
```

**2. Install dependencies**

```bash
pip install flask scikit-learn pandas numpy joblib
```

**3. (Optional) Retrain the model**

```bash
python train.py
```

**4. Run the app**

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

##  API

**`POST /predict`**

Request:
```json
{
  "symptoms": ["fever", "headache"],
  "age": 25,
  "gender": "female",
  "severity": "moderate"
}
```

Response (illustrative):
```json
{
  "disease": "migraine",
  "accuracy": 92.5,
  "medicines": ["Medicine A (dosage)", "Medicine B (dosage)"]
}
```

##  Limitations

- Predictions depend on the symptoms and diseases covered by the training dataset.
- Recommendations come from a fixed lookup table and are not personalised beyond age, gender and severity.
- The model cannot detect serious conditions, drug interactions or allergies.
- Always consult a healthcare professional before taking any medicine.

##  Future Improvements

- [ ] Add more diseases, symptoms and medicine data
- [ ] Show prediction confidence and top-3 likely diseases
- [ ] Add precautions and diet/lifestyle suggestions
- [ ] Add multi-language support
- [ ] Deploy online (Render / Railway / Hugging Face Spaces)

