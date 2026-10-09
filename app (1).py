#! C:\Users\susmi\Downloads\ayurvedaa\projectenv\Scripts\python.exe

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# Directories
BASE_DIR     = os.path.dirname(__file__)
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))
MED_CSV      = os.path.abspath(os.path.join(BASE_DIR, '..', 'data',
                        'full_ayurvedic_medicine_recommendations.csv'))

# Load model artifacts
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
model         = joblib.load(os.path.join(MODEL_DIR, 'model_disease.pkl'))
le_disease    = joblib.load(os.path.join(MODEL_DIR, 'le_disease.pkl'))
accuracy      = float(open(os.path.join(MODEL_DIR, 'accuracy.txt')).read())

# Load full medicines CSV
med_df = pd.read_csv(MED_CSV)

app = Flask(__name__,
            static_folder=FRONTEND_DIR,
            static_url_path='')

def get_age_group(age: int) -> str:
    if age <= 12:   return 'child'
    if age <= 59:   return 'adult'
    return 'elderly'

@app.route('/') 
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data     = request.get_json()
    symptoms = data.get('symptoms', [])
    age      = int(data.get('age', 0))
    gender   = data.get('gender', '').lower().strip()
    severity = data.get('severity', '').lower().strip()

    # 1) Build feature vector
    x = np.array([1 if feat in symptoms else 0
                  for feat in feature_names]).reshape(1, -1)

    # 2) Predict disease
    d_idx   = model.predict(x)[0]
    disease = le_disease.inverse_transform([d_idx])[0]

    # 3) Map age → group
    age_grp = get_age_group(age)

    # 4) Filter medicines CSV
    mask = (
      (med_df['Disease'].str.lower().str.strip() == disease.lower()) &
      (med_df['Age Group'].str.lower() == age_grp) &
      (med_df['Gender'].str.lower() == gender) &
      (med_df['Severity'].str.lower() == severity)
    )
    meds = []
    if mask.any():
        entry = med_df.loc[mask,
                  'Recommended Ayurvedic Medicines (with dosage if applicable)'
                ].iloc[0]
        meds  = [m.strip() for m in entry.split(';')][:2]

    return jsonify({
        'disease':   disease,
        'accuracy':  round(accuracy * 100, 2),
        'medicines': meds
    })

if __name__ == '__main__':
    app.run(debug=True)
