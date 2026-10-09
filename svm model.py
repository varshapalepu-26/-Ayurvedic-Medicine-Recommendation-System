# svm_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1) Load and clean data
df = pd.read_csv(os.path.join("data", "training_data.csv"))
df.columns = df.columns.str.strip()
df['prognosis'] = df['prognosis'].str.strip().str.lower()
df = df[df['prognosis'].notna()]

# 2) Split features/label
y = df['prognosis']
X = df.drop(columns=['prognosis'])

# 3) Drop near‑perfectly correlated (“leaky”) features
le_temp = LabelEncoder().fit(y)
y_enc_temp = le_temp.transform(y)
for col in X.select_dtypes(include=['int64','float64']).columns:
    if abs(X[col].corr(pd.Series(y_enc_temp))) > 0.99:
        X = X.drop(columns=[col])

# 4) Handle missing values and one‑hot encode categoricals
X = X.fillna(0)
X = pd.get_dummies(X, drop_first=True)

# 5) Encode the label
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 6) Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 7) Optional: feature selection to reduce noise but keep more features
selector = SelectKBest(chi2, k=90).fit(X_train, y_train)
X_train_sel = selector.transform(X_train)
X_test_sel  = selector.transform(X_test)

# 8) Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled  = scaler.transform(X_test_sel)

# 9) Train a higher-capacity SVM (RBF kernel)
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=42)
model.fit(X_train_scaled, y_train)

# 10) Evaluate
y_pred = model.predict(X_test_scaled)
print("=== SVM (RBF Kernel, C=1.0) ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2%}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.2%}")
print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.2%}\n")
