# random_forest_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix

# 1) Load and clean data
df = pd.read_csv(os.path.join("data", "training_data.csv"))
df.columns = df.columns.str.strip()
df['prognosis'] = df['prognosis'].str.strip().str.lower()
df = df[df['prognosis'].notna()]

# 2) Split features/label
y = df['prognosis']
X = df.drop(columns=['prognosis'])

# 3) Drop leaky features
le_temp = LabelEncoder().fit(y)
y_enc_temp = le_temp.transform(y)
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    if abs(X[col].corr(pd.Series(y_enc_temp))) > 0.99:
        X = X.drop(columns=[col])

# 4) Handle missing & categorical
X = X.fillna(0)
X = pd.get_dummies(X, drop_first=True)

# 5) Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 6) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 7) Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# 8) Evaluate
y_pred = model.predict(X_test)
print("=== Random Forest ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2%}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.2%}")
print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.2%}")

